"""CPU contracts for runtime-selected spatial sequence parallelism."""

from contextlib import redirect_stderr
import io
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
            "paired_train", "paired_val", "ode_init", "rf_init", "s3_init",
        )}
        self.asset_file = self.root / "assets.yaml"
        OmegaConf.save(OmegaConf.create(self.assets), self.asset_file)

    def config(self, stage="s1", recipe="14b-fsdp8-smoke", **kwargs):
        return resolve_config(stage, self.assets, self.root / "run", recipe=recipe, **kwargs)

    def plan(self, *arguments):
        args = build_parser().parse_args([
            "run", "--assets", str(self.asset_file), "--output", str(self.root / "run"),
            *arguments,
        ])
        return build_plan(args)

    def assert_topology(self, config, sp, accumulation=None):
        accumulation = sp if accumulation is None else accumulation
        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.sequence_parallel_size, sp)
        self.assertEqual(config.data_parallel_size, 8 // sp)
        self.assertEqual(config.gradient_accumulation_steps, accumulation)
        self.assertEqual(config.effective_batch_size, 8 // sp * accumulation)

    def test_all_stages_recipes_degrees_and_cli_spellings(self):
        for recipe, budgets in (("14b-fsdp8", [3000, 1000, 2000]),
                                ("14b-fsdp8-smoke", [6, 2, 6])):
            for flag in ("--sp", "-sp"):
                for sp in (1, 2, 4, 8):
                    with self.subTest(recipe=recipe, flag=flag, sp=sp):
                        configs = self.plan("--recipe", recipe, flag, str(sp))
                        self.assertEqual([c.stage for c in configs], ["s1", "s2", "s3"])
                        self.assertEqual([c.max_steps for c in configs], budgets)
                        for index, config in enumerate(configs):
                            self.assert_topology(config, sp)
                            prefix = "wf_14b_fsdp8" + (f"_sp{sp}" if sp != 1 else "")
                            expected = prefix + "_experimental_"
                            expected += "smoke_" if recipe.endswith("-smoke") else ""
                            self.assertEqual(config.recipe_id, expected + config.stage)
                            self.assertEqual(config.config_name, config.recipe_id)
                            self.assertEqual(list(config.denoising_step_list), [1000, 750, 500, 250])
                            self.assertEqual(config.sharding_strategy, "full")
                            self.assertFalse(config.no_save)
                            if index:
                                previous = configs[index - 1]
                                checkpoint = Path(previous.logdir) / (
                                    f"checkpoint_model_{previous.max_steps:06d}/model.pt")
                                self.assertEqual(config.generator_ckpt, str(checkpoint))

    def test_canonical_defaults_preserve_sp1_and_reference_id(self):
        for recipe in ("14b-fsdp8", "14b-fsdp8-smoke"):
            for config in self.plan("--recipe", recipe):
                self.assert_topology(config, 1)
                self.assertNotIn("_sp1", config.recipe_id)
        reference = self.config(recipe="reference")
        self.assert_topology(reference, 1)
        self.assertEqual(reference.recipe_id, "wf_s1")

    def test_removed_recipe_names_are_rejected_by_cli_and_saved_config(self):
        for recipe in ("14b-fsdp8-sp4", "14b-fsdp8-sp4-smoke"):
            with self.subTest(recipe=recipe):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    self.plan("--recipe", recipe, "--sp", "4")
                self.assertEqual(error.exception.code, 2)
                with self.assertRaisesRegex(ConfigError, "unsupported recipe"):
                    self.config(recipe=recipe, sequence_parallel_size=4)
                saved = self.config(sequence_parallel_size=4)
                saved.recipe = recipe
                with self.assertRaisesRegex(ConfigError, "unsupported recipe"):
                    validate_config(saved)

    def test_set_sp_and_explicit_accumulation_are_independent(self):
        for sp in (1, 2, 4, 8):
            for accumulation in (1, 3, 8):
                with self.subTest(sp=sp, accumulation=accumulation):
                    configs = self.plan(
                        "--recipe", "14b-fsdp8-smoke", "--set", f"sequence_parallel_size={sp}",
                        "--set", f"gradient_accumulation_steps={accumulation}",
                    )
                    for config in configs:
                        self.assert_topology(config, sp, accumulation)
            self.assert_topology(self.config(overrides=[f"sequence_parallel_size={sp}"]), sp)
            self.assert_topology(self.config(
                sequence_parallel_size=sp, overrides=["gradient_accumulation_steps=auto"]), sp)

    def test_stage_overrides_do_not_leak(self):
        configs = self.plan(
            "--recipe", "14b-fsdp8-smoke", "--set", "s1.sequence_parallel_size=2",
            "--set", "s2.sequence_parallel_size=8", "--set", "s3.sequence_parallel_size=4",
            "--set", "s2.gradient_accumulation_steps=3",
        )
        self.assert_topology(configs[0], 2)
        self.assert_topology(configs[1], 8, 3)
        self.assert_topology(configs[2], 4)
        self.assert_topology(self.config(
            sequence_parallel_size=4, overrides=["s2.sequence_parallel_size=8"]), 4)

    def test_cli_conflicts_with_applicable_set_only(self):
        for key in ("sequence_parallel_size", "s1.sequence_parallel_size"):
            with self.subTest(key=key):
                self.assert_topology(self.config(sequence_parallel_size=4, overrides=[f"{key}=4"]), 4)
                with self.assertRaisesRegex(ConfigError, "sequence_parallel_size|--sp"):
                    self.config(sequence_parallel_size=4, overrides=[f"{key}=2"])
        with self.assertRaises(ConfigError):
            self.plan("--recipe", "14b-fsdp8-smoke", "--sp", "4",
                      "--set", "s2.sequence_parallel_size=8")
        self.assert_topology(self.config(
            sequence_parallel_size=4, overrides=["s2.sequence_parallel_size=invalid"]), 4)

    def test_invalid_sp_and_accumulation_fail_before_model_loading(self):
        for value in (0, -1, 3, 16, True, 2.5, "bad", float("inf"), float("nan")):
            with self.subTest(sp=value), self.assertRaises(ConfigError):
                self.config(sequence_parallel_size=value)
        for field in ("sequence_parallel_size", "gradient_accumulation_steps"):
            for value in ("0", "-1", "true", "2.5", "bad", "null", ".inf", ".nan"):
                with self.subTest(field=field, value=value), self.assertRaises(ConfigError):
                    self.config(overrides=[f"{field}={value}"])
        with self.assertRaises(ConfigError):
            self.config(overrides=["sequence_parallel_size=3"])

    def test_cli_accepts_one_integer_per_launch_not_implicit_sweep(self):
        for arguments in (("--sp", "2.5"), ("--sp", "bad"), ("-sp", "1", "2", "4", "8")):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args([
                        "run", "--assets", str(self.asset_file), "--output", str(self.root / "run"),
                        *arguments,
                    ])
                self.assertEqual(error.exception.code, 2)

    def test_reference_checks_student_and_critic_head_divisibility(self):
        for sp in (1, 2, 4):
            self.assert_topology(self.config(recipe="reference", sequence_parallel_size=sp), sp)
        # The 14B teacher has 40 heads, but the 1.3B student/critic only have 12.
        with self.assertRaises(ConfigError):
            self.config(recipe="reference", sequence_parallel_size=8)

    def test_saved_config_preserves_explicit_sp_and_accumulation(self):
        config = self.config(sequence_parallel_size=4)
        saved = self.root / "resolved_config.yaml"
        OmegaConf.save(config, saved)
        restored = OmegaConf.load(saved)
        self.assertEqual(restored.gradient_accumulation_steps, 4)
        self.assertEqual(restored.recipe_id, "wf_14b_fsdp8_sp4_experimental_smoke_s1")
        validate_config(restored)
        restored.effective_batch_size = 32
        with self.assertRaisesRegex(ConfigError, "effective batch"):
            validate_config(restored)
        restored.effective_batch_size = 8
        restored.data_parallel_size = 8
        with self.assertRaisesRegex(ConfigError, "data_parallel_size"):
            validate_config(restored)

    def test_resume_rejects_different_topology_or_accumulation(self):
        checkpoint = self.root / "checkpoint_model_000001"
        checkpoint.mkdir()
        names = [f"trainer_state_rank{rank:02d}.pt" for rank in range(8)]
        for name in ["model.pt", *names]:
            (checkpoint / name).write_bytes(b"metadata fixture")
        marker = {"world_size": 8, "step": 1, "rank_state_files": names}
        path = checkpoint / "resume_complete.json"
        path.write_text(json.dumps(marker))
        checkpoint_complete(checkpoint, self.config(sequence_parallel_size=1))
        with self.assertRaisesRegex(ConfigError, "sequence_parallel_size"):
            checkpoint_complete(checkpoint, self.config(sequence_parallel_size=4))
        for sp in (1, 2, 4, 8):
            marker.update(sequence_parallel_size=sp, gradient_accumulation_steps=sp)
            path.write_text(json.dumps(marker))
            checkpoint_complete(checkpoint, self.config(sequence_parallel_size=sp))
            for other in (1, 2, 4, 8):
                if other != sp:
                    with self.assertRaisesRegex(ConfigError, "sequence_parallel_size"):
                        checkpoint_complete(checkpoint, self.config(sequence_parallel_size=other))
            marker["gradient_accumulation_steps"] = sp + 1
            path.write_text(json.dumps(marker))
            with self.assertRaisesRegex(ConfigError, "gradient_accumulation_steps"):
                checkpoint_complete(checkpoint, self.config(sequence_parallel_size=sp))

    def run_cpu_cli(self, *arguments):
        command = [sys.executable, "-c", (
            "import sys; from wf_training.cli import main; "
            "code=main(sys.argv[1:]); assert 'torch' not in sys.modules; sys.exit(code)"
        ), *arguments, "--assets", str(self.asset_file), "--output", str(self.root / "dry")]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertFalse((self.root / "dry").exists())
        return json.loads(result.stdout)

    def test_dry_run_and_show_config_cpu_only_for_each_degree(self):
        for sp in (1, 2, 4, 8):
            for command in (("run", "--dry-run"), ("show-config",)):
                with self.subTest(sp=sp, command=command):
                    report = self.run_cpu_cli(*command, "--recipe", "14b-fsdp8-smoke", "-sp", str(sp))
                    self.assertEqual(len(report["stages"]), 3)
                    for stage in report["stages"]:
                        self.assert_topology(OmegaConf.create(stage["config"]), sp)

    def test_preflight_sp8_does_not_import_torch_or_write_run(self):
        model = Path(self.assets["model_root"]) / "Wan2.1-T2V-14B"
        (model / "google/umt5-xxl").mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({
            "model_type": "t2v", "dim": 5120, "ffn_dim": 13824,
            "num_heads": 40, "num_layers": 40, "in_dim": 16, "out_dim": 16,
        }))
        for name in ("diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth",
                     "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer_config.json",
                     "google/umt5-xxl/spiece.model"):
            (model / name).write_text("metadata-only fixture; no tensor loading")
        Path(self.assets["prompts"]).write_text("\n".join(f"prompt {i}" for i in range(8)))
        Path(self.assets["distill_init_14b"]).write_text("checkpoint metadata fixture")
        report = self.run_cpu_cli("preflight", "--recipe", "14b-fsdp8-smoke", "--stage", "s1", "--sp", "8")
        self.assertEqual(report[0]["status"], "static_checks_passed")
        self.assertFalse(report[0]["assets"]["Wan2.1-T2V-14B"]["architecture"]["weights_loaded"])


if __name__ == "__main__":
    unittest.main()
