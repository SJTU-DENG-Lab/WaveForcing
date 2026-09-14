"""HSDP owner-only tensor persistence with real CPU Adam, EMA and rank files.

This simulates two eight-rank nodes sequentially. It verifies persistence and
resume semantics, not FSDP/NCCL communication or CUDA RNG behavior.
"""

import copy
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_trainer_state as state_helpers
from wf_training.config import checkpoint_complete
from wf_training.utils.checkpoint import load_model_checkpoint


trainer_module = state_helpers.trainer_module


class HSDPCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.fixture = state_helpers.TrainerStateTest(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)

    def make_trainer(self, rank, *, populate=False):
        trainer = self.fixture.make_trainer(populate=populate)
        trainer.config.recipe = "14b-hsdp-smoke"
        trainer.config.world_size = trainer.world_size = 16
        trainer.config.gpus_per_node = 8
        trainer.config.sharding_strategy = "hybrid_full"
        trainer.config.sequence_parallel_size = 4
        trainer.config.gradient_accumulation_steps = 4
        trainer.is_main_process = rank == 0
        return trainer

    def write_snapshot(self):
        directory = self.fixture.root / "checkpoint_model_000003"
        directory.mkdir()
        self.sources = {}
        self.expected_draws = {}
        # Rank zero publishes the completion marker only after every file exists.
        for rank in reversed(range(16)):
            source = self.make_trainer(rank, populate=True)
            source.data_batches_seen = 7 + rank
            source.paired_batches_seen = 5 + rank
            source.nan_skip_count = rank
            source.critic_nan_skip_count = rank + 1
            # Distinct local shard values detect accidental loading of rank zero
            # for all non-owner ranks. Replicas retain identical tensor states.
            for optimizer in (source.generator_optimizer, source.critic_optimizer):
                for state in optimizer.state.values():
                    state["exp_avg"].add_(rank % 8 + 1)
            for value in source.generator_ema.shadow.values():
                value.add_(rank % 8 + 1)
            self.sources[rank] = source
            random.seed(1000 + rank)
            np.random.seed(2000 + rank)
            torch.random.default_generator.manual_seed(3000 + rank)
            cuda_rng = torch.tensor([rank, rank + 1], dtype=torch.uint8)
            with patch.object(trainer_module.dist, "get_rank", return_value=rank), \
                    patch.object(torch.cuda, "get_rng_state", return_value=cuda_rng):
                if rank >= 8:
                    # Non-owners must not even construct large tensor snapshots.
                    with patch.object(source.generator_optimizer, "state_dict", side_effect=AssertionError("duplicate generator Adam")), \
                            patch.object(source.critic_optimizer, "state_dict", side_effect=AssertionError("duplicate critic Adam")), \
                            patch.object(source.generator_ema, "state_dict", side_effect=AssertionError("duplicate EMA")):
                        source._save_resume_state(str(directory))
                else:
                    source._save_resume_state(str(directory))
            self.expected_draws[rank] = (random.random(), np.random.random(), torch.rand(4), cuda_rng)
        torch.save({"generator": state_helpers.snapshot(self.sources[0].model.generator),
                    "critic": state_helpers.snapshot(self.sources[0].model.fake_score)}, directory / "model.pt")
        return directory

    def assert_nested_equal(self, actual, expected):
        self.fixture.assert_nested_equal(actual, expected)

    def restore(self, directory, rank):
        restored = self.make_trainer(rank)
        owner = self.sources[rank % 8]
        restored.model.generator.load_state_dict(state_helpers.snapshot(owner.model.generator))
        restored.model.fake_score.load_state_dict(state_helpers.snapshot(owner.model.fake_score))
        with patch.object(trainer_module.dist, "get_rank", return_value=rank):
            restored._load_resume_state(str(directory))
        restored._restore_ema(restored._resume_ema_shard)
        return restored

    def test_saves_exactly_one_optimizer_ema_payload_per_shard_and_all_rank_rng_files(self):
        directory = self.write_snapshot()
        marker = checkpoint_complete(directory, self.sources[0].config, expected_step=3)
        self.assertEqual(marker["version"], 2)
        self.assertEqual(marker["optimizer_state_ranks"], list(range(8)))
        self.assertEqual(marker["rank_state_files"], [f"trainer_state_rank{r:02d}.pt" for r in range(16)])
        self.assertEqual(marker["fsdp_replica_size"], 2)
        for rank in range(16):
            state = load_model_checkpoint(directory / f"trainer_state_rank{rank:02d}.pt")
            self.assertEqual(state["rank"], rank)
            self.assertEqual(state["version"], 2)
            self.assertEqual(state["optimizer_state_rank"], rank % 8)
            self.assertEqual(state["fsdp_shard_rank"], rank % 8)
            self.assertEqual(state["fsdp_replica_rank"], rank // 8)
            self.assertEqual(state["data_batches_seen"], 7 + rank)
            self.assertEqual(state["paired_batches_seen"], 5 + rank)
            torch.testing.assert_close(state["rng"]["torch_cuda"], self.expected_draws[rank][3])
            for name in ("generator_optimizer", "critic_optimizer", "generator_ema_shard"):
                if rank < 8:
                    self.assertIsNotNone(state[name])
                else:
                    self.assertIsNone(state[name])
        self.assertFalse(list(directory.glob("*.tmp")))

    def test_resume_uses_owner_adam_ema_but_rank_local_cursors_rng_and_skip_counts(self):
        directory = self.write_snapshot()
        for rank in (0, 7, 8, 15):
            with self.subTest(rank=rank):
                restored = self.restore(directory, rank)
                draws = (random.random(), np.random.random(), torch.rand(4))
                expected = self.expected_draws[rank]
                self.assertEqual(draws[:2], expected[:2])
                torch.testing.assert_close(draws[2], expected[2], rtol=0, atol=0)
                torch.testing.assert_close(self.fixture.set_cuda_rng.call_args.args[0], expected[3])
                self.assertEqual((restored.step, restored.data_batches_seen, restored.paired_batches_seen),
                                 (3, 7 + rank, 5 + rank))
                self.assertEqual((restored.dataloader.count, restored.paired_dataloader.count),
                                 ((7 + rank) % 3, (5 + rank) % 2))
                self.assertEqual((restored.nan_skip_count, restored.critic_nan_skip_count), (rank, rank + 1))
                source = self.sources[rank % 8]
                self.assert_nested_equal(restored.generator_optimizer.state_dict(), source.generator_optimizer.state_dict())
                self.assert_nested_equal(restored.critic_optimizer.state_dict(), source.critic_optimizer.state_dict())
                self.assert_nested_equal(restored.generator_ema.state_dict(), source.generator_ema.state_dict())

    def test_resumed_nonowner_next_adam_and_ema_update_matches_owner_state(self):
        directory = self.write_snapshot()
        restored = self.restore(directory, 15)
        source = self.sources[7]
        for trainer in (restored, source):
            for module, optimizer in ((trainer.model.generator, trainer.generator_optimizer),
                                      (trainer.model.fake_score, trainer.critic_optimizer)):
                optimizer.zero_grad(set_to_none=True)
                module(torch.tensor([0.3, 0.7])).square().backward()
                optimizer.step()
            trainer.generator_ema.update(trainer.model.generator)
        self.assert_nested_equal(state_helpers.snapshot(restored.model.generator), state_helpers.snapshot(source.model.generator))
        self.assert_nested_equal(state_helpers.snapshot(restored.model.fake_score), state_helpers.snapshot(source.model.fake_score))
        self.assert_nested_equal(restored.generator_optimizer.state_dict(), source.generator_optimizer.state_dict())
        self.assert_nested_equal(restored.critic_optimizer.state_dict(), source.critic_optimizer.state_dict())
        self.assert_nested_equal(restored.generator_ema.state_dict(), source.generator_ema.state_dict())

    def test_resume_rejects_corrupted_rank_and_owner_identity_step_schema_or_strategy(self):
        directory = self.write_snapshot()
        paths = {rank: directory / f"trainer_state_rank{rank:02d}.pt" for rank in (7, 15)}
        # The loader may mmap tensor storage; retain detached copies before
        # deliberately replacing these files in corruption cases.
        original = {rank: copy.deepcopy(load_model_checkpoint(path)) for rank, path in paths.items()}
        cases = [
            (15, "optimizer_state_rank", 0), (15, "version", 1),
            (15, "step", 4), (15, "rank", 14), (15, "world_size", 8),
            (15, "sharding_strategy", "full"), (15, "fsdp_replica_rank", 0),
            (7, "optimizer_state_rank", 0), (7, "version", 1),
            (7, "step", 4), (7, "rank", 0), (7, "world_size", 8),
            (7, "sharding_strategy", "full"), (7, "fsdp_shard_rank", 0),
            (7, "ema_mode", "full"),
        ]
        for rank, field, value in cases:
            state = copy.deepcopy(original[rank])
            state[field] = value
            torch.save(state, paths[rank])
            with self.subTest(rank=rank, field=field), self.assertRaises(ValueError):
                self.restore(directory, 15)
            torch.save(original[rank], paths[rank])

    def test_resume_rejects_marker_topology_change_and_missing_owner_payload(self):
        directory = self.write_snapshot()
        marker_path = directory / "resume_complete.json"
        marker = json.loads(marker_path.read_text())
        for field, value in (("sharding_strategy", "full"), ("gpus_per_node", 16),
                             ("fsdp_group_layout", "other"), ("step", 4)):
            marker_path.write_text(json.dumps({**marker, field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.restore(directory, 15)
        marker_path.write_text(json.dumps(marker))
        (directory / "trainer_state_rank07.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.restore(directory, 15)


if __name__ == "__main__":
    unittest.main()
