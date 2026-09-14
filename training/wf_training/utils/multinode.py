"""Collective control plane for a training plan launched by external torchrun.

Only global rank zero creates run directories and advances stage status. Every
rank checks its local installation and shared files before model construction.
Torch imports remain lazy so configuration previews are CPU-only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace

from omegaconf import OmegaConf

from wf_training.config import ConfigError, plain
from wf_training.utils.source import source_provenance


def apply_torchrun_topology(args) -> tuple[int, int]:
    """Use torchrun's global/local counts, rejecting conflicting CLI values."""
    try:
        world = int(os.environ['WORLD_SIZE'])
        local = int(os.environ['LOCAL_WORLD_SIZE'])
    except (KeyError, ValueError) as error:
        raise ConfigError('distributed commands require WORLD_SIZE and LOCAL_WORLD_SIZE from torchrun') from error
    if world < 1 or local < 1 or world % local:
        raise ConfigError('torchrun WORLD_SIZE must be a positive multiple of LOCAL_WORLD_SIZE')
    for name, expected in (('world_size', world), ('gpus_per_node', local)):
        value = getattr(args, name, None)
        if value is not None and value != expected:
            raise ConfigError(f'--{name.replace("_", "-")} conflicts with torchrun topology')
        setattr(args, name, expected)
    return world, local


def _dist():
    import torch.distributed as dist
    return dist


def rank_zero_call(label, function):
    """Broadcast preparation failures as well as values; peers never poll files."""
    dist = _dist()
    packet = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {'value': function(), 'error': None}
        except (Exception, KeyboardInterrupt) as error:
            packet[0] = {'value': None, 'error': f'{type(error).__name__}: {error}'}
    dist.broadcast_object_list(packet, src=0)
    if packet[0]['error']:
        raise ConfigError(f'{label} failed on rank 0: {packet[0]["error"]}')
    return packet[0]['value']


def all_rank_call(label, function, *, require_equal=False):
    """Collect local checks/errors before proceeding to model collectives."""
    dist = _dist()
    try:
        local = {'value': function(), 'error': None}
    except (Exception, KeyboardInterrupt) as error:
        local = {'value': None, 'error': f'{type(error).__name__}: {error}'}
    packets = [None] * dist.get_world_size()
    dist.all_gather_object(packets, local)
    errors = [f'rank {rank}: {packet["error"]}' for rank, packet in enumerate(packets)
              if packet['error']]
    if errors:
        raise ConfigError(f'{label} failed: ' + '; '.join(errors))
    values = [packet['value'] for packet in packets]
    if require_equal and any(value != values[0] for value in values[1:]):
        mismatches = [str(rank) for rank, value in enumerate(values) if value != values[0]]
        raise ConfigError(f'{label} differs from rank 0 on ranks ' + ', '.join(mismatches))
    return values


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _shared_asset_identity(reports):
    """Stat every recorded shard/pair; hash small assets without reading big weights.

    Device/inode numbers can differ across mounts of the same NFS export, so
    identity uses canonical paths, lengths, timestamps and small-file contents.
    """
    files = {}

    def visit(value):
        if isinstance(value, dict):
            if 'path' in value and 'bytes' in value and 'mtime_ns' in value:
                files[value['path']] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for report in reports:
        visit(report['assets'])
    identities = []
    for name, expected in sorted(files.items()):
        path = Path(name)
        stat = path.stat()
        if (not path.is_file() or str(path.resolve()) != name
                or stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']):
            raise ConfigError(f'shared asset differs from rank-zero preflight: {name}')
        content = hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size <= 4 * 1024**2 else None
        identities.append([name, stat.st_size, stat.st_mtime_ns, content])
    return _digest(identities)


def _installation_identity(reports):
    versions = dict(sorted(
        (distribution.metadata['Name'].lower().replace('_', '-'), distribution.version)
        for distribution in importlib.metadata.distributions() if distribution.metadata['Name']
    ))
    for report in reports:
        if versions != report['package_versions'] or platform.python_version() != report['python_version']:
            raise ConfigError('local dependency/Python versions differ from rank-zero preflight')
    return {'source_sha256': source_provenance()['source_sha256'],
            'packages_sha256': _digest(versions), 'python_version': platform.python_version(),
            'assets_sha256': _shared_asset_identity(reports)}


def _check_topology(configs, world, local):
    for config in configs:
        if (config.world_size != world or getattr(config, 'gpus_per_node', config.world_size) != local):
            raise ConfigError('saved/resolved configuration differs from torchrun topology')
        if getattr(config, 'recipe', '') not in ('14b-hsdp', '14b-hsdp-smoke'):
            raise ConfigError('distributed commands require a 14b-hsdp recipe')


def distributed_main(args):
    from wf_training import cli
    from wf_training.utils.distributed import launch_distributed_job

    for name, value in (('WANDB_MODE', 'offline'), ('HF_HUB_OFFLINE', '1'),
                        ('TRANSFORMERS_OFFLINE', '1'), ('OMP_NUM_THREADS', '1')):
        os.environ.setdefault(name, value)
    launch_distributed_job()
    dist = _dist()
    current_directory = None
    current_status = None
    stage_can_write = False
    training_started = False
    try:
        world, local = all_rank_call('torchrun topology', lambda: apply_torchrun_topology(args),
                                    require_equal=True)[0]
        all_rank_call('launch arguments', lambda: vars(args), require_equal=True)
        if args.command == 'distributed-resume':
            def prepare_resume():
                config, path, checkpoint = cli.load_resume(args.run, args.checkpoint)
                return {'config': plain(config), 'path': str(path), 'checkpoint': checkpoint}
            prepared = rank_zero_call('resume preparation', prepare_resume)
            configs = [OmegaConf.create(prepared['config'])]
            resume_from = prepared['checkpoint']
        else:
            local_configs = None
            def prepare_local_plan():
                nonlocal local_configs
                local_configs = cli.build_plan(args)
                return [plain(config) for config in local_configs]
            all_rank_call('resolved training plan', prepare_local_plan, require_equal=True)
            configs = local_configs
            resume_from = ''
        all_rank_call('resolved topology', lambda: _check_topology(configs, world, local))

        def initial_preflight():
            reports = []
            for index, config in enumerate(configs):
                if not resume_from and Path(config.logdir).exists():
                    raise ConfigError(f'refusing existing stage directory; use distributed-resume: {config.logdir}')
                report = cli.preflight(config, allow_pending_init=index > 0)
                if resume_from:
                    # Compare visibility of all rank states before any rank
                    # enters FSDP/model loading during a restart.
                    marker = cli.checkpoint_complete(resume_from, config)
                    checkpoint_files = ['model.pt', 'resume_complete.json', *marker['rank_state_files']]
                    report['assets']['resume_checkpoint'] = []
                    for name in checkpoint_files:
                        path = (Path(resume_from) / name).resolve()
                        stat = path.stat()
                        report['assets']['resume_checkpoint'].append(
                            {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
                reports.append(report)
            return reports
        reports = rank_zero_call('plan preflight', initial_preflight)
        identities = all_rank_call('installation and shared assets',
                                   lambda: _installation_identity(reports), require_equal=True)

        for config in configs:
            current_directory = Path(config.logdir)
            config_path = current_directory / 'resolved_config.yaml'
            current_status = {'status': 'running', 'started_at': cli._now(),
                              'launcher': 'external torchrun', 'world_size': world,
                              'gpus_per_node': local, 'resume_from': resume_from}

            def prepare_stage():
                nonlocal stage_can_write
                report = cli.preflight(config)
                if not resume_from:
                    current_directory.mkdir(parents=True, exist_ok=False)
                    stage_can_write = True
                    OmegaConf.save(config, config_path)
                    cli._write_json(current_directory / 'preflight.json', report)
                    cli._write_json(current_directory / 'run_manifest.json', {
                        'schema_version': 1, 'created_at': cli._now(), 'stage': config.stage,
                        'source_path': str(Path(cli.__file__).resolve().parent),
                        'source_provenance': source_provenance(), 'python': cli.sys.executable,
                        'initial_checkpoint': report['assets']['initial_checkpoint'],
                        'environment': {key: os.environ[key] for key in cli._ENV_WHITELIST if key in os.environ},
                        'distributed_identity': identities[0], 'verified_ranks': world,
                    })
                else:
                    stage_can_write = True
                cli._write_json(current_directory / 'status.json', current_status)
                return report

            report = rank_zero_call(f'{config.stage} preparation', prepare_stage)
            all_rank_call(f'{config.stage} shared assets',
                          lambda: _shared_asset_identity([report]), require_equal=True)

            def verify_config():
                loaded = OmegaConf.load(config_path)
                if plain(loaded) != plain(config):
                    raise ConfigError('shared resolved_config.yaml differs from the agreed plan')
                if source_provenance()['source_sha256'] != identities[0]['source_sha256']:
                    raise ConfigError('training source changed after distributed preflight')
            all_rank_call(f'{config.stage} shared configuration', verify_config)
            dist.barrier()
            # Do not wrap model execution in an object collective. If one
            # rank OOMs, its peers may still be inside FSDP/NCCL: exit promptly
            # and let the external torchrun agents terminate the worker gang.
            training_started = True
            cli._worker(
                SimpleNamespace(config=str(config_path), resume_from=resume_from),
                keep_process_group=True)
            training_started = False
            dist.barrier()

            def finish_stage():
                endpoint = current_directory / f'checkpoint_model_{config.max_steps:06d}'
                cli.checkpoint_complete(endpoint, config, expected_step=config.max_steps)
                current_status.update(status='completed', finished_at=cli._now(), checkpoint=str(endpoint))
                cli._write_json(current_directory / 'status.json', current_status)
            rank_zero_call(f'{config.stage} completion', finish_stage)
            dist.barrier()
            current_directory = None
            current_status = None
            stage_can_write = False
    except (Exception, KeyboardInterrupt) as error:
        if current_directory is not None and current_status is not None:
            try:
                if training_started:
                    cli._write_json(current_directory / f'worker_failure_rank{dist.get_rank():02d}.json',
                                    {'rank': dist.get_rank(), 'failed_at': cli._now(),
                                     'error': f'{type(error).__name__}: {error}'})
                if dist.get_rank() == 0 and stage_can_write and current_directory.exists():
                    current_status.update(status='failed', finished_at=cli._now(), error=str(error))
                    cli._write_json(current_directory / 'status.json', current_status)
            except OSError:
                # Preserve the training error even if the filesystem failed.
                pass
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
