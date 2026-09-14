"""Static contracts for eight-rank FSDP with two spatial SP4 groups."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf

from wf_training.cli import build_parser, build_plan
from wf_training.config import ConfigError, checkpoint_complete, resolve_config, validate_config


class SpatialRecipeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = {key: str(self.root / key) for key in (
            "model_root", "prompts", "distill_init_14b", "s2_init_14b", "s3_init_14b",
            "paired_train", "paired_val",
        )}

    def config(self, **kwargs):
        return resolve_config("s1", self.assets, self.root / "run",
                              recipe="14b-fsdp8-sp4-smoke", **kwargs)

    def test_complete_plans_keep_recipe_and_eight_sample_updates(self):
        asset_file = self.root / "assets.yaml"
        OmegaConf.save(OmegaConf.create(self.assets), asset_file)
        for recipe, budgets in (("14b-fsdp8-sp4", [3000, 1000, 2000]),
                                ("14b-fsdp8-sp4-smoke", [6, 2, 6])):
            args = build_parser().parse_args([
                "run", "--recipe", recipe, "--assets", str(asset_file),
                "--output", str(self.root / "run"),
            ])
            configs = build_plan(args)
            self.assertEqual([c.stage for c in configs], ["s1", "s2", "s3"])
            self.assertEqual([c.max_steps for c in configs], budgets)
            for config in configs:
                self.assertEqual(config.world_size, 8)
                self.assertEqual(config.sequence_parallel_size, 4)
                self.assertEqual(config.data_parallel_size, 2)
                self.assertEqual(config.gradient_accumulation_steps, 4)
                self.assertEqual(config.effective_batch_size, 8)
                self.assertIn("sp4", config.recipe_id)
                self.assertEqual(list(config.denoising_step_list), [1000, 750, 500, 250])
                self.assertEqual(config.sharding_strategy, "full")
                self.assertFalse(config.no_save)

    def test_topology_and_accumulation_cannot_silently_change(self):
        for override in ("sequence_parallel_size=1", "sequence_parallel_size=8",
                         "sequence_parallel_size=3", "sequence_parallel_size=true",
                         "gradient_accumulation_steps=1", "gradient_accumulation_steps=0",
                         "gradient_accumulation_steps=2.5", "gradient_accumulation_steps=bad"):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                self.config(overrides=[override])
        config = self.config()
        config.effective_batch_size = 32
        with self.assertRaisesRegex(ConfigError, "effective batch"):
            validate_config(config)
        config = self.config()
        config.data_parallel_size = 8
        with self.assertRaisesRegex(ConfigError, "data_parallel_size"):
            validate_config(config)

    def test_sp1_recipe_does_not_accept_unimplemented_topology_override(self):
        with self.assertRaisesRegex(ConfigError, "sequence_parallel_size"):
            resolve_config("s1", self.assets, self.root / "run", recipe="14b-fsdp8-smoke",
                           overrides=["sequence_parallel_size=4", "gradient_accumulation_steps=4"])

    def test_resume_rejects_old_or_different_sp_topology(self):
        checkpoint = self.root / "checkpoint_model_000001"
        checkpoint.mkdir()
        names = [f"trainer_state_rank{rank:02d}.pt" for rank in range(8)]
        for name in ["model.pt", *names]:
            (checkpoint / name).write_bytes(b"metadata fixture")
        marker = {"world_size": 8, "step": 1, "rank_state_files": names}
        path = checkpoint / "resume_complete.json"
        path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ConfigError, "sequence_parallel_size"):
            checkpoint_complete(checkpoint, self.config())
        marker.update(sequence_parallel_size=4, gradient_accumulation_steps=4)
        path.write_text(json.dumps(marker))
        checkpoint_complete(checkpoint, self.config())
        marker["gradient_accumulation_steps"] = 1
        path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ConfigError, "gradient_accumulation_steps"):
            checkpoint_complete(checkpoint, self.config())

    def test_dry_run_remains_cpu_only(self):
        asset_file = self.root / "assets.yaml"
        OmegaConf.save(OmegaConf.create(self.assets), asset_file)
        command = [sys.executable, "-c", (
            "import sys; from wf_training.cli import main; "
            "code=main(sys.argv[1:]); assert 'torch' not in sys.modules; sys.exit(code)"
        ), "run", "--recipe", "14b-fsdp8-sp4-smoke", "--assets", str(asset_file),
            "--output", str(self.root / "dry"), "--dry-run"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(len(json.loads(result.stdout)["stages"]), 3)
        self.assertFalse((self.root / "dry").exists())


if __name__ == "__main__":
    unittest.main()
