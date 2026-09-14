"""CPU checks for SP sampling, actual trainer microbackwards, and exact resume.

Tiny differentiable losses replace Wan, while Trainer's accumulation, clipping,
optimizer/EMA control flow, cache cleanup, and checkpoint files remain real.
"""

import copy
import json
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import test_trainer_state as state_helpers


Trainer = state_helpers.Trainer
trainer_module = state_helpers.trainer_module


class PairCycle:
    def __init__(self):
        self.count = 0

    def __next__(self):
        index = self.count % 2
        self.count += 1
        return {"prompts": [f"prompt {index}"], "y_ref": torch.tensor([index + 1.0])}


class SPTrainerTest(unittest.TestCase):
    def setUp(self):
        # Reuse the existing CPU-only distributed/RNG/checkpoint fixture.
        self.fixture = state_helpers.TrainerStateTest(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)

    def make_trainer(self, *, paired=False, accumulation=4):
        trainer = self.fixture.make_trainer()
        trainer.config.gradient_accumulation_steps = accumulation
        trainer.config.sequence_parallel_size = 1
        trainer.config.max_steps = 1
        trainer.config.image_or_video_shape = [1, 2, 1, 1, 1]
        trainer.config.negative_prompt = "negative"
        trainer.config.distribution_loss = "paired_coupled_dmd" if paired else "dmd"
        trainer.dtype = torch.float32
        trainer.max_grad_norm_generator = 0.7
        trainer.max_grad_norm_critic = 0.9
        trainer.generator_optimizer = torch.optim.SGD(trainer.model.generator.parameters(), lr=0.03)
        trainer.critic_optimizer = torch.optim.SGD(trainer.model.fake_score.parameters(), lr=0.02)
        trainer.model.eval = Mock()
        trainer.model.lpips_weight = 0.25 if paired else 0
        trainer.model.inference_pipeline = SimpleNamespace(release_caches=Mock())
        trainer.model._offp_pipeline = SimpleNamespace(release_caches=Mock())
        trainer.model.text_encoder = lambda text_prompts: {
            "value": torch.tensor(float(text_prompts[0].split()[-1]) + 1)
            if text_prompts[0] != "negative" else torch.tensor(0.0)
        }
        trainer.paired_dataloader = PairCycle()
        trainer.losses = {"generator": [], "critic": []}
        trainer.backward_calls = {"generator": [], "critic": []}
        trainer.grads_at_clip = {"generator": [], "critic": []}

        for name, module in (("generator", trainer.model.generator),
                             ("critic", trainer.model.fake_score)):
            def clip(instance, max_norm, name=name):
                trainer.grads_at_clip[name].append(
                    [parameter.grad.detach().clone() for parameter in instance.parameters()])
                return torch.nn.utils.clip_grad_norm_(instance.parameters(), max_norm)

            module.clip_grad_norm_ = MethodType(clip, module)
            # Any accidental FSDP no_sync use fails this CPU fixture as well.
            module.no_sync = Mock(side_effect=AssertionError("must retain sharded gradients"))

        def loss(train_generator, **kwargs):
            name = "generator" if train_generator else "critic"
            module = trainer.model.generator if train_generator else trainer.model.fake_score
            value = (kwargs["clean_latent"].mean() if paired
                     else kwargs["conditional_dict"]["value"])
            prediction = module(torch.stack((value, value.square())))
            dmd = prediction.square()
            lpips = prediction.abs()
            total = dmd + trainer.model.lpips_weight * lpips if train_generator else dmd
            total.register_hook(lambda grad: trainer.backward_calls[name].append(grad.detach().clone()))
            trainer.losses[name].append(total.detach())
            # Keep a graph in these diagnostics to ensure Trainer detaches it.
            return total, {"dmdtrain_gradient_norm": prediction,
                           "lpips_loss": lpips, "critic_timestep": value}

        trainer.model.generator_loss = lambda **kwargs: loss(True, **kwargs)
        trainer.model.critic_loss = lambda **kwargs: loss(False, **kwargs)
        return trainer

    def test_four_microbatches_match_mean_gradient_then_one_clip_step_and_ema(self):
        trainer = self.make_trainer()
        expected_generator = copy.deepcopy(trainer.model.generator)
        expected_critic = copy.deepcopy(trainer.model.fake_score)
        expected_grads = {}
        for name, model, values, max_norm, lr in (
                ("generator", expected_generator, [1, 2, 3, 1], 0.7, 0.03),
                ("critic", expected_critic, [2, 3, 1, 2], 0.9, 0.02)):
            loss = torch.stack([model(torch.tensor([float(v), float(v * v)])).square()
                                for v in values]).mean()
            loss.backward()
            expected_grads[name] = [p.grad.detach().clone() for p in model.parameters()]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            torch.optim.SGD(model.parameters(), lr=lr).step()

        phases = []
        original_accumulate = trainer._accumulate_phase

        def capture_phase(*args):
            result = original_accumulate(*args)
            phases.append(result)
            return result

        with patch.object(trainer, "_accumulate_phase", side_effect=capture_phase), \
                patch.object(trainer.generator_optimizer, "step", wraps=trainer.generator_optimizer.step) as g_step, \
                patch.object(trainer.critic_optimizer, "step", wraps=trainer.critic_optimizer.step) as c_step, \
                patch.object(trainer.generator_ema, "update", wraps=trainer.generator_ema.update) as ema, \
                patch.object(trainer, "_restore_generator_mask", wraps=trainer._restore_generator_mask) as restore:
            trainer.train()

        self.assertEqual((g_step.call_count, c_step.call_count, ema.call_count), (1, 1, 1))
        self.assertEqual((trainer.step, trainer.data_batches_seen), (1, 8))
        self.assertEqual(restore.call_count, 8)
        self.assertEqual(trainer.model.inference_pipeline.release_caches.call_count, 8)
        self.assertEqual(trainer.model._offp_pipeline.release_caches.call_count, 8)
        for name, actual, expected in (("generator", trainer.model.generator, expected_generator),
                                       ("critic", trainer.model.fake_score, expected_critic)):
            self.assertEqual(len(trainer.grads_at_clip[name]), 1)
            self.assertEqual(len(trainer.backward_calls[name]), 4)
            for scale in trainer.backward_calls[name]:
                torch.testing.assert_close(scale, torch.tensor(0.25))
            for grad, expected_grad in zip(trainer.grads_at_clip[name][0], expected_grads[name]):
                torch.testing.assert_close(grad, expected_grad)
            for parameter, expected_parameter in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(parameter, expected_parameter)
                self.assertIsNone(parameter.grad)
            actual.no_sync.assert_not_called()
        for phase, name in zip(phases, ("generator", "critic")):
            torch.testing.assert_close(phase[f"{name}_loss"], torch.stack(trainer.losses[name]).mean())
            for value in phase.values():
                if torch.is_tensor(value):
                    self.assertFalse(value.requires_grad)
                    self.assertIsNone(value.grad_fn)

    def test_paired_dmd_and_lpips_both_accumulate_and_count_actual_cursors(self):
        trainer = self.make_trainer(paired=True)
        phases = []
        original = trainer._accumulate_phase

        def capture(*args):
            result = original(*args)
            phases.append(result)
            return result

        with patch.object(trainer, "_accumulate_phase", side_effect=capture):
            trainer.train()
        self.assertEqual((trainer.data_batches_seen, trainer.paired_batches_seen), (4, 8))
        self.assertEqual([len(trainer.backward_calls[key]) for key in ("generator", "critic")], [4, 4])
        self.assertEqual([len(trainer.grads_at_clip[key]) for key in ("generator", "critic")], [1, 1])
        torch.testing.assert_close(phases[0]["generator_loss"],
                                   phases[0]["paired_dmd_loss"] + 0.25 * phases[0]["lpips_loss"])
        self.assertEqual(trainer.model._offp_pipeline.release_caches.call_count, 8)

    def test_update_ratio_counts_optimizer_iterations_with_accumulation(self):
        trainer = self.make_trainer()
        trainer.config.max_steps = 3
        with patch.object(trainer.generator_optimizer, "step", wraps=trainer.generator_optimizer.step) as g_step, \
                patch.object(trainer.critic_optimizer, "step", wraps=trainer.critic_optimizer.step) as c_step, \
                patch.object(trainer.generator_ema, "update", wraps=trainer.generator_ema.update) as ema:
            trainer.train()
        self.assertEqual((g_step.call_count, c_step.call_count, ema.call_count), (2, 3, 2))
        self.assertEqual((trainer.step, trainer.data_batches_seen), (3, 20))

    def test_path_logging_counts_every_microbatch_in_a_mixed_update(self):
        for paths, expected_path, coverage in (([0, 2, 0, 2], -1, (0.5, 0.0, 0.5)),
                                               ([1], 1, (0.0, 1.0, 0.0))):
            with self.subTest(paths=paths):
                trainer = self.make_trainer(accumulation=len(paths))
                original = trainer.model.generator_loss
                remaining_paths = iter(paths)

                def choose_path(**kwargs):
                    trainer.model._last_train_path = next(remaining_paths)
                    return original(**kwargs)

                trainer.model.generator_loss = choose_path
                trainer.train()
                logged = {call.args[0]: call.args[1]
                          for call in trainer.writer.add_scalar.call_args_list}
                self.assertEqual(logged["train_path"], expected_path)
                self.assertEqual(tuple(logged[f"path_{name}"] for name in ("rf", "crf", "sf")), coverage)

    def test_one_nonfinite_microbatch_skips_whole_phase_without_adam_or_ema_update(self):
        for bad_phase in ("generator", "critic"):
            with self.subTest(phase=bad_phase):
                trainer = self.make_trainer()
                trainer.generator_optimizer = torch.optim.Adam(trainer.model.generator.parameters(), lr=0.01)
                trainer.critic_optimizer = torch.optim.Adam(trainer.model.fake_score.parameters(), lr=0.01)
                model = trainer.model.generator if bad_phase == "generator" else trainer.model.fake_score
                optimizer = trainer.generator_optimizer if bad_phase == "generator" else trainer.critic_optimizer
                before = state_helpers.snapshot(model)
                ema_before = copy.deepcopy(trainer.generator_ema.state_dict())
                original = getattr(trainer.model, f"{bad_phase}_loss")
                calls = []

                def poison(**kwargs):
                    result, log = original(**kwargs)
                    calls.append(1)
                    if len(calls) == 2:
                        result = result * float("nan")
                    return result, log

                setattr(trainer.model, f"{bad_phase}_loss", poison)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step, \
                        patch.object(trainer.generator_ema, "update", wraps=trainer.generator_ema.update) as ema:
                    trainer.train()
                self.assertEqual(len(calls), 4)
                step.assert_not_called()
                self.assertEqual(optimizer.state_dict()["state"], {})
                self.fixture.assert_nested_equal(state_helpers.snapshot(model), before)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                if bad_phase == "generator":
                    ema.assert_not_called()
                    self.fixture.assert_nested_equal(trainer.generator_ema.state_dict(), ema_before)
                    self.assertEqual(trainer.nan_skip_count, 1)
                else:
                    self.assertEqual(ema.call_count, 1)
                    self.assertEqual(trainer.critic_nan_skip_count, 1)

    def test_remote_nonfinite_flag_skips_each_rank_even_with_finite_local_gradients(self):
        trainer = self.make_trainer()
        reductions = []

        def remote_skip(flag, op):
            reductions.append(int(flag.item()))
            flag.fill_(1)

        with patch.object(trainer_module.dist, "is_initialized", return_value=True), \
                patch.object(trainer_module.dist, "all_reduce", side_effect=remote_skip), \
                patch.object(trainer.generator_optimizer, "step") as g_step, \
                patch.object(trainer.critic_optimizer, "step") as c_step, \
                patch.object(trainer.generator_ema, "update") as ema:
            trainer.train()
        self.assertEqual(reductions, [0, 0])
        g_step.assert_not_called()
        c_step.assert_not_called()
        ema.assert_not_called()
        self.assertEqual((trainer.nan_skip_count, trainer.critic_nan_skip_count), (1, 1))

    def test_nonfinite_loss_with_finite_gradients_still_skips_the_phase(self):
        trainer = self.make_trainer()
        original = trainer.model.generator_loss
        calls = []

        def poison_loss_only(**kwargs):
            loss, log = original(**kwargs)
            calls.append(1)
            if len(calls) == 2:
                loss = loss + float("nan")
            return loss, log

        trainer.model.generator_loss = poison_loss_only
        result = trainer._accumulate_phase(True, False)
        self.assertEqual(len(calls), 4)
        self.assertTrue(torch.isfinite(result["generator_grad_norm"]))
        self.assertEqual(result["grad_nonfinite"], 1.0)

    def test_cleanup_runs_when_forward_or_backward_raises_for_either_phase(self):
        for paired in (False, True):
            for generator in (False, True):
                for failure in ("forward", "backward"):
                    with self.subTest(paired=paired, generator=generator, failure=failure):
                        trainer = self.make_trainer(paired=paired)
                        name = "generator_loss" if generator else "critic_loss"
                        original = getattr(trainer.model, name)

                        def fail(**kwargs):
                            if failure == "forward":
                                raise RuntimeError("injected failure")
                            loss, log = original(**kwargs)

                            def backward_failure(gradient):
                                raise RuntimeError("injected failure")

                            loss.register_hook(backward_failure)
                            return loss, log

                        setattr(trainer.model, name, fail)
                        with patch.object(trainer, "_restore_generator_mask") as restore:
                            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                                trainer._accumulate_phase(generator, paired)
                        restore.assert_called_once()
                        trainer.model.inference_pipeline.release_caches.assert_called_once()
                        trainer.model._offp_pipeline.release_caches.assert_called_once()

    def test_resume_persists_sp_accumulation_identity_and_rejects_changes(self):
        source = self.fixture.make_trainer(populate=True)
        source.world_size = 8
        source.config.sequence_parallel_size = 4
        source.config.gradient_accumulation_steps = 4
        directory = self.fixture.save_fixture(source)
        marker = json.loads((directory / "resume_complete.json").read_text())
        state_path = directory / "trainer_state_rank00.pt"
        state = copy.deepcopy(state_helpers.load_model_checkpoint(state_path))
        for saved in (marker, state):
            self.assertEqual(saved["sequence_parallel_size"], 4)
            self.assertEqual(saved["gradient_accumulation_steps"], 4)
            self.assertEqual(saved["data_parallel_world_size"], 2)
        self.assertEqual((state["sequence_parallel_rank"], state["data_parallel_rank"]), (0, 0))
        restored = self.fixture.make_trainer()
        restored.world_size = 8
        restored.config.sequence_parallel_size = 4
        restored.config.gradient_accumulation_steps = 4
        restored._load_resume_state(str(directory))
        self.assertEqual((restored.data_batches_seen, restored.paired_batches_seen), (7, 5))
        for key, value in (("sequence_parallel_size", 2), ("gradient_accumulation_steps", 1)):
            with self.subTest(key=key), patch.object(restored.config, key, value):
                with self.assertRaisesRegex(ValueError, key + " mismatch"):
                    restored._load_resume_state(str(directory))
        for key in ("sequence_parallel_size", "gradient_accumulation_steps", "data_parallel_rank"):
            bad = dict(state)
            bad[key] += 1
            torch.save(bad, state_path)
            with self.assertRaisesRegex(ValueError, key + " mismatch"):
                restored._load_resume_state(str(directory))

    def test_legacy_resume_without_identity_is_sp1_accumulation1_only(self):
        source = self.fixture.make_trainer(populate=True)
        directory = self.fixture.save_fixture(source)
        marker_path = directory / "resume_complete.json"
        state_path = directory / "trainer_state_rank00.pt"
        marker = json.loads(marker_path.read_text())
        state = copy.deepcopy(state_helpers.load_model_checkpoint(state_path))
        identity_keys = source._training_topology(0).keys()
        for saved in (marker, state):
            for key in identity_keys:
                saved.pop(key, None)
        marker_path.write_text(json.dumps(marker))
        torch.save(state, state_path)
        restored = self.fixture.make_trainer()
        restored._load_resume_state(str(directory))
        restored.config.gradient_accumulation_steps = 4
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps mismatch"):
            restored._load_resume_state(str(directory))

    def test_accumulated_training_resume_matches_uninterrupted_adam_and_ema(self):
        def make():
            trainer = self.make_trainer()
            trainer.generator_optimizer = torch.optim.Adam(trainer.model.generator.parameters(), lr=0.01)
            trainer.critic_optimizer = torch.optim.Adam(trainer.model.fake_score.parameters(), lr=0.02)
            return trainer

        source = make()
        source.train()
        directory = self.fixture.save_fixture(source)
        source.config.max_steps = 3
        source.train()

        restored = make()
        raw = state_helpers.load_model_checkpoint(directory / "model.pt")
        restored.model.generator.load_state_dict(raw["generator"])
        restored.model.fake_score.load_state_dict(raw["critic"])
        restored._load_resume_state(str(directory))
        restored._restore_ema(restored._resume_ema_shard)
        restored.config.max_steps = 3
        restored.train()
        self.assertEqual((restored.step, restored.data_batches_seen), (3, 20))
        for name in ("generator", "fake_score"):
            self.fixture.assert_nested_equal(
                state_helpers.snapshot(getattr(restored.model, name)),
                state_helpers.snapshot(getattr(source.model, name)))
        for name in ("generator_optimizer", "critic_optimizer", "generator_ema"):
            self.fixture.assert_nested_equal(
                getattr(restored, name).state_dict(), getattr(source, name).state_dict())

    def test_init_creates_sp_before_model_and_samplers_share_dp_rank(self):
        config = SimpleNamespace(
            world_size=8, sequence_parallel_size=4, gradient_accumulation_steps=4,
            mixed_precision=False, seed=123, disable_wandb=True, logdir=str(self.fixture.root),
            distribution_loss="paired_coupled_dmd", paired_only=True, paired_manifest="unused",
            num_training_frames=21, lr=0.01, beta1=0.9, beta2=0.99, weight_decay=0.0,
            beta1_critic=0.9, beta2_critic=0.99, data_path="unused", batch_size=1,
            ema_weight=0, ema_start_step=0,
        )
        tiny = self.fixture.make_trainer().model
        tiny.fsdp_initialized = True
        vae = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter([
            SimpleNamespace(device=torch.device("cuda", 0), dtype=torch.float32)])))
        vae.to = lambda **kwargs: vae
        tiny.vae = vae
        loader = Mock()
        loader.__len__ = Mock(return_value=8)
        for paired in (False, True):
            with self.subTest(paired=paired):
                config.distribution_loss = "paired_coupled_dmd" if paired else "dmd"
                events = []

                def model_init(*args, **kwargs):
                    events.append("model")
                    return tiny

                with patch.object(trainer_module, "launch_distributed_job"), \
                        patch.object(trainer_module.dist, "get_rank", return_value=5), \
                        patch.object(trainer_module.dist, "get_world_size", return_value=8), \
                        patch.object(torch.cuda, "current_device", return_value=0), \
                        patch.object(trainer_module, "initialize_fsdp_topology", side_effect=lambda cfg: events.append("fsdp")), \
                        patch.object(trainer_module, "initialize_sequence_parallel", side_effect=lambda size: events.append(("sp", size))), \
                        patch.object(trainer_module, "get_data_parallel_rank", return_value=1), \
                        patch.object(trainer_module, "get_data_parallel_world_size", return_value=2), \
                        patch.object(trainer_module, "set_seed") as seed, \
                        patch.object(trainer_module, "DMD", side_effect=model_init), \
                        patch.object(trainer_module, "OffPolicyDMD", side_effect=model_init), \
                        patch.object(trainer_module, "TextDataset", return_value=list(range(16))), \
                        patch("wf_training.utils.paired_dmd_dataset.PairedDMDLatentDataset", return_value=list(range(16))), \
                        patch.object(torch.utils.data.distributed, "DistributedSampler", wraps=torch.utils.data.distributed.DistributedSampler) as sampler, \
                        patch.object(torch.utils.data, "DataLoader", return_value=loader):
                    Trainer(config)
                self.assertEqual(events, ["fsdp", ("sp", 4), "model"])
                seed.assert_called_once_with(124)
                self.assertEqual(sampler.call_count, 2 if paired else 1)
                for call in sampler.call_args_list:
                    self.assertEqual(call.kwargs["num_replicas"], 2)
                    self.assertEqual(call.kwargs["rank"], 1)


if __name__ == "__main__":
    unittest.main()
