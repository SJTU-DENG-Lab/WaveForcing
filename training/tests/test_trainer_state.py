"""Exercise real Trainer checkpoint/resume and optimizer control flow on CPU.

Only distributed gathers/collectives, CUDA RNG, and unused GPU model imports
are substituted. Checkpoint files, tiny models, Adam, EMA and Trainer methods
are real; these tests do not exercise FSDP/NCCL itself.
"""

import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from wf_training.utils.checkpoint import load_model_checkpoint
from wf_training.utils.ema import ShardedEMA


def load_trainer_module():
    # Trainer.__new__ skips model construction. Avoid importing Wan/FlashAttention
    # merely to access its persistence and optimizer-loop methods.
    model_stub = ModuleType("wf_training.model")
    model_stub.DMD = model_stub.OffPolicyDMD = object
    path = Path(__file__).parents[1] / "wf_training/trainer/distillation.py"
    spec = importlib.util.spec_from_file_location("cpu_trainer_state", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("wf_training.model")
    sys.modules["wf_training.model"] = model_stub
    try:
        spec.loader.exec_module(module)
    finally:
        # Preserve dependencies loaded during import. Restoring all sys.modules
        # would cause PyTorch's custom-op registrations to execute a second time.
        if previous is None:
            sys.modules.pop("wf_training.model", None)
        else:
            sys.modules["wf_training.model"] = previous
    return module


trainer_module = load_trainer_module()
Trainer = trainer_module.Trainer


class TinyModel(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25 + offset, 0.5 + offset]))
        self.bias = nn.Parameter(torch.tensor([0.1 + offset]))

    def forward(self, value):
        return (self.weight * value).sum() + self.bias.sum()


class CountingCycle:
    def __init__(self, length):
        self.length = length
        self.count = 0

    def __next__(self):
        index = self.count % self.length
        self.count += 1
        # Cursor replay may consume randomness; resume must restore RNG after it.
        random.random()
        np.random.random()
        torch.rand(1)
        return {"prompts": [f"prompt {index}"], "index": index}


def snapshot(module):
    # Simulate the detached CPU snapshots produced by FSDP's full state gather.
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


class TrainerStateTest(unittest.TestCase):
    def setUp(self):
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(trainer_module.dist, "get_rank", return_value=0))
        self.patches.enter_context(patch.object(trainer_module.dist, "get_world_size", return_value=1))
        self.patches.enter_context(patch.object(trainer_module.dist, "barrier"))
        self.gather = self.patches.enter_context(
            patch.object(trainer_module, "fsdp_state_dict", side_effect=snapshot))
        self.cuda_state = torch.tensor([3, 1, 4], dtype=torch.uint8)
        self.patches.enter_context(patch.object(torch.cuda, "get_rng_state", return_value=self.cuda_state))
        self.set_cuda_rng = self.patches.enter_context(patch.object(torch.cuda, "set_rng_state"))
        self.patches.enter_context(patch.object(torch.cuda, "empty_cache"))

    def tearDown(self):
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)

    def make_trainer(self, *, populate=False):
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            ema_weight=0.5, ema_start_step=0, ema_mode="sharded",
            profile_memory=False, resume_save_iters=1, distribution_loss="dmd",
            max_steps=3, dfake_gen_update_ratio=2, no_save=True,
            log_iters=100, gc_interval=100,
        )
        trainer.model = SimpleNamespace(generator=TinyModel(0), fake_score=TinyModel(0.3), lpips_weight=0)
        trainer.generator_optimizer = torch.optim.Adam(trainer.model.generator.parameters(), lr=0.01)
        trainer.critic_optimizer = torch.optim.Adam(trainer.model.fake_score.parameters(), lr=0.02)
        trainer.generator_ema = ShardedEMA(trainer.model.generator, decay=trainer.config.ema_weight)
        trainer.device = "cpu"
        trainer.world_size = 1
        trainer.is_main_process = True
        trainer.output_path = str(self.root)
        trainer.step = 0
        trainer.data_batches_seen = 0
        trainer.paired_batches_seen = 0
        trainer._dataloader_len = 3
        trainer._paired_dataloader_len = 2
        trainer.dataloader = CountingCycle(3)
        trainer.paired_dataloader = CountingCycle(2)
        trainer._resume_ema_shard = None
        trainer._wandb = None
        trainer.writer = Mock()
        trainer.previous_time = None
        if populate:
            for module, optimizer in ((trainer.model.generator, trainer.generator_optimizer),
                                      (trainer.model.fake_score, trainer.critic_optimizer)):
                optimizer.zero_grad(set_to_none=True)
                module(torch.ones(2)).square().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            trainer.generator_ema.update(trainer.model.generator)
            trainer.step = 3
            trainer.data_batches_seen = 7
            trainer.paired_batches_seen = 5
            trainer.nan_skip_count = 2
            trainer.critic_nan_skip_count = 1
        return trainer

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def save_fixture(self, trainer):
        trainer.save(force_resume=True)
        return self.root / f"checkpoint_model_{trainer.step:06d}"

    def test_save_contains_raw_exported_ema_local_state_and_complete_marker(self):
        trainer = self.make_trainer(populate=True)
        raw = snapshot(trainer.model.generator)
        critic = snapshot(trainer.model.fake_score)
        ema = copy.deepcopy(trainer.generator_ema.shadow)
        directory = self.save_fixture(trainer)
        model_state = load_model_checkpoint(directory / "model.pt")
        rank_state = load_model_checkpoint(directory / "trainer_state_rank00.pt")
        marker = json.loads((directory / "resume_complete.json").read_text())
        self.assert_nested_equal(model_state["generator"], raw)
        self.assert_nested_equal(model_state["critic"], critic)
        self.assert_nested_equal(model_state["generator_ema"], ema)
        self.assert_nested_equal(snapshot(trainer.model.generator), raw)
        self.assertEqual(self.gather.call_count, 3)
        self.assertEqual(model_state["ema_mode"], "sharded")
        self.assert_nested_equal(rank_state["generator_ema_shard"], trainer.generator_ema.state_dict())
        self.assert_nested_equal(rank_state["generator_optimizer"], trainer.generator_optimizer.state_dict())
        self.assertEqual(marker["rank_state_files"], ["trainer_state_rank00.pt"])
        self.assertEqual(marker["step"], 3)
        self.assertEqual(marker["world_size"], 1)
        self.assertEqual(trainer._last_full_saved_step, 3)
        self.assertFalse(list(directory.glob("*.tmp")))

    def test_save_restores_raw_weights_when_ema_export_fails(self):
        trainer = self.make_trainer(populate=True)
        raw = snapshot(trainer.model.generator)
        self.gather.side_effect = [raw, snapshot(trainer.model.fake_score), RuntimeError("gather failed")]
        with self.assertRaisesRegex(RuntimeError, "gather failed"):
            trainer.save(force_resume=True)
        self.assert_nested_equal(snapshot(trainer.model.generator), raw)
        self.assertFalse(list(self.root.rglob("resume_complete.json")))

    def test_each_rank_saves_and_restores_its_own_local_ema(self):
        rank_zero = self.make_trainer(populate=True)
        rank_zero.world_size = 2
        directory = self.save_fixture(rank_zero)
        rank_one = self.make_trainer(populate=True)
        rank_one.world_size = 2
        rank_one.is_main_process = False
        with torch.no_grad():
            rank_one.model.generator.weight.add_(10)
        rank_one.generator_ema.update(rank_one.model.generator)
        with patch.object(trainer_module.dist, "get_rank", return_value=1):
            rank_one._save_resume_state(str(directory))
            restored = self.make_trainer()
            restored.world_size = 2
            restored._load_resume_state(str(directory))
            restored._restore_ema(restored._resume_ema_shard)
        self.assert_nested_equal(restored.generator_ema.state_dict(), rank_one.generator_ema.state_dict())
        self.assertFalse(torch.equal(restored.generator_ema.shadow["weight"], rank_zero.generator_ema.shadow["weight"]))
        marker = json.loads((directory / "resume_complete.json").read_text())
        self.assertEqual(marker["rank_state_files"], ["trainer_state_rank00.pt", "trainer_state_rank01.pt"])
        self.assertTrue(all((directory / name).is_file() for name in marker["rank_state_files"]))
        self.assertEqual(load_model_checkpoint(directory / "trainer_state_rank01.pt")["rank"], 1)

    def test_resume_restores_adam_ema_cursors_skip_counts_and_rng(self):
        source = self.make_trainer(populate=True)
        directory = self.save_fixture(source)
        expected_draws = (random.random(), np.random.random(), torch.rand(4))
        random.seed(193)
        np.random.seed(193)
        torch.random.default_generator.manual_seed(193)
        restored = self.make_trainer()
        raw = load_model_checkpoint(directory / "model.pt")
        restored.model.generator.load_state_dict(raw["generator"])
        restored.model.fake_score.load_state_dict(raw["critic"])
        self.assertEqual(restored.generator_optimizer.state_dict()["state"], {})
        restored._load_resume_state(str(directory))
        restored._restore_ema(restored._resume_ema_shard)
        actual_draws = (random.random(), np.random.random(), torch.rand(4))
        self.assertEqual(actual_draws[:2], expected_draws[:2])
        torch.testing.assert_close(actual_draws[2], expected_draws[2], rtol=0, atol=0)
        self.assert_nested_equal(restored.generator_optimizer.state_dict(), source.generator_optimizer.state_dict())
        self.assert_nested_equal(restored.critic_optimizer.state_dict(), source.critic_optimizer.state_dict())
        self.assert_nested_equal(restored.generator_ema.state_dict(), source.generator_ema.state_dict())
        self.assertEqual((restored.step, restored.data_batches_seen, restored.paired_batches_seen), (3, 7, 5))
        self.assertEqual((restored.dataloader.count, restored.paired_dataloader.count), (1, 1))
        self.assertEqual((restored.nan_skip_count, restored.critic_nan_skip_count), (2, 1))
        self.set_cuda_rng.assert_called_once()
        torch.testing.assert_close(self.set_cuda_rng.call_args.args[0], self.cuda_state)
        self.assertEqual(self.set_cuda_rng.call_args.args[1], "cpu")

    def test_missing_sharded_ema_rejected_at_start_zero_and_threshold(self):
        for start, step in ((0, 0), (0, 1), (2, 2), (2, 3)):
            with self.subTest(start=start, step=step):
                trainer = self.make_trainer()
                trainer.config.ema_start_step = start
                trainer.step = step
                valid = copy.deepcopy(trainer.generator_ema.state_dict())
                with self.assertRaisesRegex(KeyError, "no EMA"):
                    trainer._restore_ema(None)
                trainer._restore_ema(valid)
                self.assert_nested_equal(trainer.generator_ema.state_dict(), valid)
        trainer = self.make_trainer()
        trainer.config.ema_start_step = 2
        trainer.step = 1
        valid = copy.deepcopy(trainer.generator_ema.state_dict())
        trainer._restore_ema(None)
        self.assertIsNone(trainer.generator_ema)
        with self.assertRaisesRegex(ValueError, "before its configured start"):
            trainer._restore_ema(valid)
        trainer.step = 2
        trainer._restore_ema(valid)
        self.assert_nested_equal(trainer.generator_ema.state_dict(), valid)
        trainer.config.ema_weight = 0
        with self.assertRaisesRegex(ValueError, "EMA is disabled"):
            trainer._restore_ema(valid)

    def test_resume_missing_local_ema_cannot_fall_back_to_exported_full_ema(self):
        source = self.make_trainer(populate=True)
        directory = self.save_fixture(source)
        rank_path = directory / "trainer_state_rank00.pt"
        state = copy.deepcopy(load_model_checkpoint(rank_path))
        del state["generator_ema_shard"]
        torch.save(state, rank_path)
        restored = self.make_trainer()
        restored._load_resume_state(str(directory))
        with self.assertRaisesRegex(KeyError, "no EMA"):
            restored._restore_ema(restored._resume_ema_shard)

    def test_train_loop_preserves_generator_frequency_and_skips_nonfinite_critic(self):
        trainer = self.make_trainer()
        trainer.is_main_process = False
        events = []

        def tiny_forward_backward(instance, batch, train_generator):
            events.append((instance.step, train_generator))
            model = instance.model.generator if train_generator else instance.model.fake_score
            loss = model(torch.ones(2)).square()
            if not train_generator and instance.step == 1:
                loss = loss * float("nan")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if train_generator:
                return {"generator_loss": loss.detach(), "generator_grad_norm": norm, "grad_nonfinite": 0.0}
            return {"critic_loss": loss.detach(), "critic_grad_norm": norm}

        trainer.fwdbwd_one_step = MethodType(tiny_forward_backward, trainer)
        with patch.object(trainer.generator_optimizer, "step", wraps=trainer.generator_optimizer.step) as g_step, \
                patch.object(trainer.critic_optimizer, "step", wraps=trainer.critic_optimizer.step) as c_step, \
                patch.object(trainer.generator_ema, "update", wraps=trainer.generator_ema.update) as ema_update:
            trainer.train()
        self.assertEqual(events, [(0, True), (0, False), (1, False), (2, True), (2, False)])
        self.assertEqual((g_step.call_count, c_step.call_count, ema_update.call_count), (2, 2, 2))
        self.assertEqual(trainer.critic_nan_skip_count, 1)
        self.assertEqual(trainer.step, 3)
        for optimizer in (trainer.generator_optimizer, trainer.critic_optimizer):
            for state in optimizer.state.values():
                self.assertEqual(state["step"].item(), 2)
                self.assertTrue(torch.isfinite(state["exp_avg"]).all())
                self.assertTrue(torch.isfinite(state["exp_avg_sq"]).all())
        for model in (trainer.model.generator, trainer.model.fake_score):
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
