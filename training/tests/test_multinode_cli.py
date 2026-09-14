"""CPU previews and real GLOO tests of the external-torchrun control plane."""
from contextlib import ExitStack, redirect_stdout
from datetime import timedelta
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from wf_training import cli
from wf_training.config import ConfigError
from wf_training.utils import multinode


def _control_plane_checks(rank, temporary):
    """Two CPU ranks emulate two launcher participants, not GPU/HSDP kernels."""
    torch.set_num_threads(1)
    root = Path(temporary)
    scenarios = ('success', 'resume', 'rank_zero_preflight_error', 'peer_arguments',
                 'peer_installation', 'peer_assets', 'second_stage_error')
    for scenario in scenarios:
        run_root = root / scenario
        torch_env = {'RANK': str(rank), 'LOCAL_RANK': '0', 'WORLD_SIZE': '2',
                     'LOCAL_WORLD_SIZE': '1'}

        def initialize():
            dist.init_process_group('gloo', init_method=f'file://{root / (scenario + ".rendezvous")}',
                                    rank=rank, world_size=2, timeout=timedelta(seconds=30))

        configs = [OmegaConf.create({'recipe': '14b-hsdp-smoke', 'stage': stage,
                                    'world_size': 2, 'gpus_per_node': 1, 'num_nodes': 2,
                                    'max_steps': 1, 'logdir': str(run_root / stage)})
                   for stage in ('s1', 's2')]
        resume_checkpoint = run_root / 's1/checkpoint_model_000001'
        if scenario == 'resume':
            configs = configs[:1]
            configs[0].max_steps = 2
            if rank == 0:
                resume_checkpoint.mkdir(parents=True)
                OmegaConf.save(configs[0], run_root / 's1/resolved_config.yaml')
                for name in ('model.pt', 'trainer_state_rank00.pt', 'trainer_state_rank01.pt'):
                    (resume_checkpoint / name).write_text('original checkpoint')
                (resume_checkpoint / 'resume_complete.json').write_text('{}')
        observed_stages = []
        original_write_json = cli._write_json

        def preflight(config, *, allow_pending_init=False):
            assert dist.get_rank() == 0, 'preflight must run on global rank zero only'
            if scenario == 'rank_zero_preflight_error':
                raise OSError('rank-zero assets unavailable')
            if scenario == 'second_stage_error' and config.stage == 's2' and not allow_pending_init:
                raise OSError('second-stage preparation failed')
            return {'assets': {'initial_checkpoint': {'path': '/fixture/model.pt'}},
                    'package_versions': {}, 'python_version': 'fixture'}

        def write_json(path, data):
            assert dist.get_rank() == 0, 'only global rank zero may write control-plane records'
            original_write_json(path, data)

        def worker(args, *, keep_process_group):
            assert keep_process_group
            assert dist.is_initialized()
            config = OmegaConf.load(args.config)
            observed_stages.append(config.stage)
            if config.stage == 's2':
                previous = json.loads((run_root / 's1/status.json').read_text())
                assert previous['status'] == 'completed'
            assert bool(args.resume_from) == (scenario == 'resume')
            # Exercise a collective inside each stage on the reused WORLD.
            gradient = torch.tensor(float(rank + 1))
            dist.all_reduce(gradient)
            assert gradient.item() == 3
            endpoint = Path(config.logdir) / f'checkpoint_model_{config.max_steps:06d}'
            if rank == 0:
                endpoint.mkdir()
            dist.barrier()
            (endpoint / f'trainer_state_rank{rank:02d}.pt').write_text('cpu fixture')
            dist.barrier()
            if rank == 0:
                (endpoint / 'model.pt').write_text('cpu fixture')

        def checkpoint_complete(path, config, *, expected_step=None):
            assert dist.get_rank() == 0
            assert expected_step in (None, config.max_steps)
            assert (Path(path) / 'model.pt').is_file()
            assert all((Path(path) / f'trainer_state_rank{peer:02d}.pt').is_file() for peer in range(2))
            return {'rank_state_files': ['trainer_state_rank00.pt', 'trainer_state_rank01.pt']}

        def load_resume(run, checkpoint):
            assert dist.get_rank() == 0, 'resume preparation must run only on global rank zero'
            assert run == str(run_root / 's1')
            assert checkpoint == str(resume_checkpoint)
            return configs[0], run_root / 's1/resolved_config.yaml', str(resume_checkpoint)

        def installation(reports):
            if scenario == 'peer_installation' and rank == 1:
                return {'source_sha256': 'different source'}
            if scenario == 'peer_assets' and rank == 1:
                raise OSError('peer cannot access shared asset')
            return {'source_sha256': 'fixture-source'}

        args = SimpleNamespace(command='distributed-run', world_size=None, gpus_per_node=None,
                               output=str(run_root), dry_run=False)
        if scenario == 'resume':
            args.command = 'distributed-resume'
            args.run = str(run_root / 's1')
            args.checkpoint = str(resume_checkpoint)
        if scenario == 'peer_arguments' and rank == 1:
            args.output += '-different'
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, torch_env))
            stack.enter_context(patch('wf_training.utils.distributed.launch_distributed_job',
                                      side_effect=initialize))
            stack.enter_context(patch.object(cli, 'build_plan', return_value=configs))
            stack.enter_context(patch.object(cli, 'preflight', side_effect=preflight))
            stack.enter_context(patch.object(cli, '_write_json', side_effect=write_json))
            stack.enter_context(patch.object(cli, '_worker', side_effect=worker))
            stack.enter_context(patch.object(cli, 'checkpoint_complete', side_effect=checkpoint_complete))
            stack.enter_context(patch.object(cli, 'load_resume', side_effect=load_resume))
            stack.enter_context(patch.object(multinode, '_installation_identity', side_effect=installation))
            stack.enter_context(patch.object(multinode, '_shared_asset_identity', return_value='assets'))
            stack.enter_context(patch.object(multinode, 'source_provenance',
                                             return_value={'source_sha256': 'fixture-source'}))
            error = None
            try:
                multinode.distributed_main(args)
            except ConfigError as caught:
                error = str(caught)
            assert not dist.is_initialized(), 'WORLD must be closed after the whole plan'
        if scenario in ('success', 'resume'):
            assert error is None, error
            assert observed_stages == (['s1'] if scenario == 'resume' else ['s1', 's2'])
            for config in configs:
                status = json.loads((Path(config.logdir) / 'status.json').read_text())
                assert status['status'] == 'completed'
        elif scenario == 'second_stage_error':
            assert error and 'second-stage preparation failed' in error
            assert observed_stages == ['s1']
            assert json.loads((run_root / 's1/status.json').read_text())['status'] == 'completed'
            assert not (run_root / 's2').exists()
        else:
            assert error is not None, scenario
            assert observed_stages == []
            assert not run_root.exists()


class MultiNodeCLITest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='wf-multinode-cli-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / 'assets.yaml'
        OmegaConf.save(OmegaConf.create({'model_root': '/missing/models',
                                        'prompts': '/missing/prompts.txt',
                                        'distill_init_14b': '/missing/init.pt',
                                        'paired_train': '/missing/train.jsonl',
                                        'paired_val': '/missing/val.jsonl'}), self.assets)

    def args(self, command='show-config', *extra):
        return cli.build_parser().parse_args([
            command, '--recipe', '14b-hsdp-smoke', '--assets', str(self.assets),
            '--output', str(self.root / 'run'), *extra])

    def test_external_topology_uses_global_and_local_counts(self):
        args = self.args('distributed-run')
        with patch.dict(os.environ, {'WORLD_SIZE': '64', 'LOCAL_WORLD_SIZE': '8'}):
            self.assertEqual(multinode.apply_torchrun_topology(args), (64, 8))
        configs = cli.build_plan(args)
        self.assertEqual([config.stage for config in configs], ['s1', 's2', 's3'])
        for config in configs:
            self.assertEqual((config.world_size, config.gpus_per_node, config.num_nodes), (64, 8, 8))

    def test_explicit_counts_must_match_torchrun(self):
        for option, value in (('--world-size', '8'), ('--gpus-per-node', '4')):
            args = self.args('distributed-run', option, value)
            with self.subTest(option=option), patch.dict(
                    os.environ, {'WORLD_SIZE': '64', 'LOCAL_WORLD_SIZE': '8'}):
                with self.assertRaisesRegex(ConfigError, 'conflicts with torchrun'):
                    multinode.apply_torchrun_topology(args)

    def test_external_topology_requires_valid_launcher_environment(self):
        for environment in ({}, {'WORLD_SIZE': '64', 'LOCAL_WORLD_SIZE': '0'},
                            {'WORLD_SIZE': '63', 'LOCAL_WORLD_SIZE': '8'},
                            {'WORLD_SIZE': 'eight', 'LOCAL_WORLD_SIZE': '8'}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ConfigError):
                    multinode.apply_torchrun_topology(self.args('distributed-run'))

    def test_cpu_preview_shows_external_launcher_without_creating_directories(self):
        args = self.args('show-config', '--world-size', '64', '--gpus-per-node', '8')
        output = io.StringIO()
        with redirect_stdout(output):
            cli._display_plan(cli.build_plan(args))
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['stages'][0]['command']['nproc_per_node'], 8)
        self.assertEqual(plan['stages'][0]['command']['nnodes'], 8)
        self.assertFalse((self.root / 'run').exists())

    def test_local_run_rejects_multinode_before_preflight_or_directory_writes(self):
        configs = cli.build_plan(self.args('run', '--world-size', '64', '--gpus-per-node', '8'))
        with patch.object(cli, 'preflight') as preflight:
            with self.assertRaisesRegex(ConfigError, 'external torchrun'):
                cli._run(configs)
            preflight.assert_not_called()
        self.assertFalse((self.root / 'run').exists())

    def test_existing_local_recipe_keeps_eight_process_launcher(self):
        args = cli.build_parser().parse_args([
            'run', '--recipe', '14b-fsdp8-smoke', '--stage', 's1',
            '--assets', str(self.assets), '--output', str(self.root / 'run')])
        config = cli.build_plan(args)[0]
        self.assertEqual(config.world_size, 8)
        self.assertIn('--nproc_per_node=8', cli.worker_command(config, self.root / 'resolved.yaml'))

    def test_explicit_local_count_prevents_a_global_world_size_local_spawn(self):
        config = OmegaConf.create({'world_size': 64, 'gpus_per_node': 8,
                                   'logdir': str(self.root / 'run')})
        with self.assertRaisesRegex(ConfigError, 'external torchrun'):
            cli.worker_command(config, self.root / 'resolved.yaml')

    def test_asset_visibility_and_small_content_identity_are_checked(self):
        path = self.root / 'asset.json'
        path.write_text('{"value": 1}')
        stat = path.stat()
        reports = [{'assets': {'asset': {'path': str(path.resolve()), 'bytes': stat.st_size,
                                         'mtime_ns': stat.st_mtime_ns}}}]
        first = multinode._shared_asset_identity(reports)
        path.write_text('{"value": 2}')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(first, multinode._shared_asset_identity(reports))
        path.unlink()
        with self.assertRaises(OSError):
            multinode._shared_asset_identity(reports)

    def test_worker_failure_does_not_enter_any_further_control_collective(self):
        args = self.args('distributed-run', '--stage', 's1')
        fake_dist = Mock()
        fake_dist.get_rank.return_value = 0
        fake_dist.get_world_size.return_value = 8
        fake_dist.is_initialized.return_value = True
        fake_dist.all_gather_object.side_effect = lambda packets, value: packets.__setitem__(slice(None), [value] * 8)
        report = {'assets': {'initial_checkpoint': {'path': '/fixture/init.pt'}}}
        counts_at_failure = None

        def fail_worker(*args, **kwargs):
            nonlocal counts_at_failure
            counts_at_failure = [getattr(fake_dist, name).call_count for name in
                                 ('broadcast_object_list', 'all_gather_object', 'barrier')]
            raise RuntimeError('simulated CUDA OOM')

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8'}))
            stack.enter_context(patch('wf_training.utils.distributed.launch_distributed_job'))
            stack.enter_context(patch.object(multinode, '_dist', return_value=fake_dist))
            stack.enter_context(patch.object(cli, 'preflight', return_value=report))
            stack.enter_context(patch.object(cli, '_worker', side_effect=fail_worker))
            stack.enter_context(patch.object(multinode, '_installation_identity',
                                             return_value={'source_sha256': 'fixture'}))
            stack.enter_context(patch.object(multinode, '_shared_asset_identity', return_value='assets'))
            stack.enter_context(patch.object(multinode, 'source_provenance',
                                             return_value={'source_sha256': 'fixture'}))
            with self.assertRaisesRegex(RuntimeError, 'simulated CUDA OOM'):
                multinode.distributed_main(args)
        self.assertEqual(counts_at_failure, [getattr(fake_dist, name).call_count for name in
                                            ('broadcast_object_list', 'all_gather_object', 'barrier')])
        directory = self.root / 'run/s1'
        self.assertEqual(json.loads((directory / 'status.json').read_text())['status'], 'failed')
        self.assertIn('simulated CUDA OOM', json.loads(
            (directory / 'worker_failure_rank00.json').read_text())['error'])

    def test_worker_releases_stage_resources_but_resets_sp_only_after_success(self):
        import wf_training.trainer as trainer_package

        config = cli.build_plan(self.args('run', '--stage', 's1'))[0]
        directory = Path(config.logdir)
        directory.mkdir(parents=True)
        config_path = directory / 'resolved_config.yaml'
        OmegaConf.save(config, config_path)
        args = SimpleNamespace(config=str(config_path), resume_from='')
        for fails in (False, True):
            events = []

            def train():
                events.append('train')
                if fails:
                    raise RuntimeError('worker failure')

            trainer = SimpleNamespace(
                device='cpu', world_size=8, train=train,
                writer=SimpleNamespace(close=lambda: events.append('writer.close')),
                model=SimpleNamespace(denoising_step_list=torch.tensor([1000, 750, 500, 250])),
            )
            module = SimpleNamespace(Trainer=lambda config: trainer, __file__=str(self.root / 'fixture.py'))
            with self.subTest(fails=fails), ExitStack() as stack:
                stack.enter_context(patch.object(trainer_package, 'distillation', module, create=True))
                stack.enter_context(patch('wf_training.assets.configure_reference_runtime'))
                stack.enter_context(patch.object(dist, 'get_rank', return_value=0))
                destroy = stack.enter_context(patch.object(dist, 'destroy_process_group'))
                stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
                                                 return_value=SimpleNamespace(name='CPU fixture', total_memory=0)))
                stack.enter_context(patch('gc.collect', side_effect=lambda: events.append('gc')))
                stack.enter_context(patch.object(torch.cuda, 'synchronize',
                                                 side_effect=lambda: events.append('synchronize')))
                stack.enter_context(patch.object(torch.cuda, 'memory_allocated', return_value=0))
                clear = stack.enter_context(patch(
                    'wf_training.utils.distributed.clear_completed_fsdp_saved_views',
                    side_effect=lambda model: events.append('clear_views') or
                    {'handles': 2, 'cleared_saved_views': 4}))
                stack.enter_context(patch.object(torch.cuda, 'empty_cache', side_effect=lambda: events.append('empty')))
                stack.enter_context(patch('wf_training.utils.sequence_parallel.reset_sequence_parallel',
                                          side_effect=lambda: events.append('reset_sp')))
                if fails:
                    with self.assertRaisesRegex(RuntimeError, 'worker failure'):
                        cli._worker(args, keep_process_group=True)
                else:
                    cli._worker(args, keep_process_group=True)
                destroy.assert_not_called()
                if fails:
                    clear.assert_not_called()
                else:
                    clear.assert_called_once_with(trainer.model)
                    cleanup = json.loads((directory / 'stage_cleanup_rank00.json').read_text())
                    self.assertEqual(cleanup['cleared_saved_views'], 4)
                    self.assertEqual(cleanup['after_gc_bytes'], 0)
            self.assertEqual(events, ['train', 'writer.close'] +
                             ([] if fails else ['synchronize', 'clear_views']) +
                             ['gc', 'empty'] + ([] if fails else ['reset_sp']))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'GLOO is unavailable')
    def test_collective_stage_progression_and_failures_on_two_cpu_ranks(self):
        mp.start_processes(_control_plane_checks, args=(str(self.root),), nprocs=2,
                           start_method='fork', join=True)


if __name__ == '__main__':
    unittest.main()
