"""CPU contracts for fixed-node HSDP recipes and checkpoint placement."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from wf_training.config import (
    ConfigError, checkpoint_complete, fsdp_topology_metadata, resolve_config,
    validate_checkpoint_fsdp_topology, validate_config,
)


class HSDPConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.assets = {key: str(self.root / key) for key in (
            "model_root", "prompts", "distill_init_14b", "s2_init_14b", "s3_init_14b",
            "paired_train", "paired_val",
        )}

    def config(self, *, stage="s1", recipe="14b-hsdp", world_size=64, **kwargs):
        return resolve_config(stage, self.assets, self.root / "run", recipe=recipe,
                              world_size=world_size, **kwargs)

    def test_stage_budgets_and_default_batch_scale_with_world_not_step_division(self):
        for recipe, budgets, accum in (
                ("14b-hsdp", (3000, 1000, 2000), 4),
                ("14b-hsdp-smoke", (6, 2, 6), 4),
                ("14b-hsdp-fast", (1500, 500, 1000), 1)):
            for world in (8, 16, 64):
                for stage, budget in zip(("s1", "s2", "s3"), budgets):
                    with self.subTest(recipe=recipe, world=world, stage=stage):
                        config = self.config(recipe=recipe, stage=stage, world_size=world)
                        self.assertEqual(config.max_steps, budget)
                        self.assertEqual(config.gpus_per_node, 8)
                        self.assertEqual(config.num_nodes, world // 8)
                        self.assertEqual(config.fsdp_shard_size, 8)
                        self.assertEqual(config.fsdp_replica_size, world // 8)
                        self.assertEqual(config.sharding_strategy, "hybrid_full")
                        self.assertEqual(config.sequence_parallel_size, 4)
                        self.assertEqual(config.gradient_accumulation_steps, accum)
                        self.assertEqual(config.data_parallel_size, world // 4)
                        self.assertEqual(config.effective_batch_size, (world // 4) * accum)
                        self.assertIn(f"hsdp8x{world // 8}_sp4", config.recipe_id)
                        tag = {"14b-hsdp-smoke": "smoke_", "14b-hsdp-fast": "fast_"}.get(recipe, "")
                        if tag:
                            self.assertIn(tag, config.recipe_id)
                        else:
                            self.assertNotIn("smoke_", config.recipe_id)
                            self.assertNotIn("fast_", config.recipe_id)

    def test_64_gpu_sp_accumulation_matrix(self):
        for sp, accumulation, expected in ((4, "auto", 64), (4, 1, 16),
                                           (8, 1, 8), (8, "auto", 64), (2, 1, 32)):
            with self.subTest(sp=sp, accumulation=accumulation):
                config = self.config(sequence_parallel_size=sp,
                                     overrides=[f"gradient_accumulation_steps={accumulation}"])
                self.assertEqual(config.effective_batch_size, expected)
                self.assertEqual(config.data_parallel_size * config.batch_size, 64 // sp)
                self.assertEqual(config.max_steps, 3000)

    def test_invalid_node_sp_strategy_and_override_combinations_are_rejected(self):
        cases = [
            {"world_size": 12}, {"world_size": 0}, {"world_size": True},
            {"gpus_per_node": 4}, {"gpus_per_node": 16}, {"gpus_per_node": 0},
            {"sequence_parallel_size": 16}, {"sequence_parallel_size": 3},
            {"overrides": ["sharding_strategy=full"]},
            {"overrides": ["sharding_strategy=hybrid_zero2"]},
            {"overrides": ["fsdp_shard_size=4"]},
            {"overrides": ["num_nodes=4"]},
            {"gpus_per_node": 8, "overrides": ["gpus_per_node=4"]},
            {"recipe": "14b-fsdp8", "world_size": 16},
            {"recipe": "14b-fsdp8", "world_size": 8,
             "overrides": ["sharding_strategy=hybrid_full"]},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ConfigError):
                self.config(**arguments)

    def test_saved_derived_topology_cannot_be_changed_independently(self):
        for name, value in (("num_nodes", 4), ("fsdp_shard_size", 16),
                            ("fsdp_replica_size", 4), ("fsdp_group_layout", "other")):
            config = self.config()
            config[name] = value
            with self.subTest(field=name), self.assertRaisesRegex(ConfigError, name):
                validate_config(config)

    def test_rank_metadata_and_explicit_world_override(self):
        config = SimpleNamespace(world_size=8, gpus_per_node=8, sharding_strategy="hybrid_full")
        topology = fsdp_topology_metadata(config, rank=63, world_size=64)
        self.assertEqual(topology["num_nodes"], 8)
        self.assertEqual(topology["fsdp_shard_rank"], 7)
        self.assertEqual(topology["fsdp_replica_rank"], 7)
        for rank in (-1, 64, True, 1.5):
            with self.subTest(rank=rank), self.assertRaises(ConfigError):
                fsdp_topology_metadata(config, rank=rank, world_size=64)
        with self.assertRaises(ConfigError):
            fsdp_topology_metadata(config, world_size=12)

    def test_same_world_different_sharding_or_placement_cannot_resume(self):
        saved_config = SimpleNamespace(world_size=64, gpus_per_node=8,
                                       sharding_strategy="hybrid_full")
        saved = fsdp_topology_metadata(saved_config, rank=17)
        validate_checkpoint_fsdp_topology(saved, saved_config, rank=17)
        for local, strategy in ((16, "hybrid_full"), (8, "full"), (8, "hybrid_zero2")):
            changed = SimpleNamespace(world_size=64, gpus_per_node=local,
                                      sharding_strategy=strategy)
            with self.subTest(local=local, strategy=strategy), self.assertRaises(ConfigError):
                validate_checkpoint_fsdp_topology(saved, changed, rank=17)
        with self.assertRaisesRegex(ConfigError, "fsdp_shard_rank"):
            validate_checkpoint_fsdp_topology(saved, saved_config, rank=18)

    def test_missing_hsdp_metadata_is_rejected_including_single_node(self):
        for world in (8, 64):
            with self.subTest(world=world), self.assertRaisesRegex(ConfigError, "missing FSDP topology"):
                validate_checkpoint_fsdp_topology({}, self.config(world_size=world))
        # Legacy reference used single-node HYBRID_SHARD with only one replica;
        # its physical optimizer shards remain readable without the new metadata.
        hybrid = SimpleNamespace(world_size=8, gpus_per_node=8, sharding_strategy="hybrid_full")
        validate_checkpoint_fsdp_topology({}, hybrid)
        validate_checkpoint_fsdp_topology({}, SimpleNamespace(world_size=8))

    def make_checkpoint(self, config):
        directory = self.root / "checkpoint"
        directory.mkdir()
        names = [f"trainer_state_rank{rank:02d}.pt" for rank in range(config.world_size)]
        for name in ("model.pt", *names):
            (directory / name).write_bytes(b"metadata fixture")
        marker = {
            "version": 2, "world_size": config.world_size, "step": 3,
            **fsdp_topology_metadata(config),
            "sequence_parallel_size": config.sequence_parallel_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "rank_state_files": names, "optimizer_state_ranks": list(range(8)),
        }
        (directory / "resume_complete.json").write_text(json.dumps(marker))
        return directory, marker

    def test_complete_marker_requires_all_rank_files_and_exact_shard_owner_set(self):
        config = self.config()
        directory, marker = self.make_checkpoint(config)
        checkpoint_complete(directory, config, expected_step=3)
        marker_path = directory / "resume_complete.json"
        corruptions = [
            ("version", 1), ("version", None),
            ("optimizer_state_ranks", list(range(64))),
            ("optimizer_state_ranks", list(range(7))),
            ("optimizer_state_ranks", list(reversed(range(8)))),
            ("rank_state_files", marker["rank_state_files"][:-1]),
            ("rank_state_files", list(reversed(marker["rank_state_files"]))),
            ("fsdp_replica_size", 4), ("step", 4),
        ]
        for name, value in corruptions:
            marker_path.write_text(json.dumps({**marker, name: value}))
            with self.subTest(field=name, value=value), self.assertRaises(ConfigError):
                checkpoint_complete(directory, config, expected_step=3)
        marker_path.write_text(json.dumps(marker))
        # Even a non-owner's RNG/cursor file is required for exact restart.
        (directory / "trainer_state_rank63.pt").unlink()
        with self.assertRaises(ConfigError):
            checkpoint_complete(directory, config)


if __name__ == "__main__":
    unittest.main()
