"""CPU contract checks for experimental 14B recipes; no model imports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from wf_training.cli import build_parser, build_plan
from wf_training.config import ConfigError, plain, preflight, resolve_config, validate_config


class Recipe14BTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = {
            "model_root": str(self.root / "models"),
            "prompts": str(self.root / "prompts.txt"),
            "ode_init": str(self.root / "ode.pt"),
            "rf_init": str(self.root / "rf.pt"),
            "s3_init": str(self.root / "reference_s2.pt"),
            "distill_init_14b": str(self.root / "distill14b.pt"),
            "s2_init_14b": str(self.root / "s1_14b.pt"),
            "s3_init_14b": str(self.root / "s2_14b.pt"),
            "paired_train": str(self.root / "train.jsonl"),
            "paired_val": str(self.root / "val.jsonl"),
        }
        self.asset_file = self.root / "assets.yaml"
        OmegaConf.save(OmegaConf.create(self.assets), self.asset_file)

    def config(self, stage="s1", recipe="14b-fsdp8", **kwargs):
        return resolve_config(stage, self.assets, self.root / "run", recipe=recipe, **kwargs)

    def plan(self, *arguments):
        args = build_parser().parse_args([
            "run", "--assets", str(self.asset_file), "--output", str(self.root / "run"),
            *arguments,
        ])
        return build_plan(args)

    def test_reference_defaults_and_stage_selection_are_preserved(self):
        configs = self.plan()
        self.assertEqual([item.stage for item in configs], ["s2", "s3"])
        self.assertEqual([item.max_steps for item in configs], [1000, 2000])
        for config in configs:
            self.assertEqual(config.generator_name, "Wan2.1-T2V-1.3B")
            self.assertEqual(config.fake_name, "Wan2.1-T2V-1.3B")
            self.assertEqual(config.real_name, "Wan2.1-T2V-14B")
            self.assertEqual(list(config.denoising_step_list), [1000, 800, 600, 400, 200])
            self.assertEqual(config.recipe_id, f"wf_{config.stage}")
            self.assertEqual(config.fsdp_init_mode, "replicated")
            self.assertEqual(config.ema_mode, "full")
            self.assertFalse(config.profile_memory)
        self.assertEqual(configs[0].generator_ckpt, self.assets["rf_init"])
        # Saved configurations from before recipe selection remain valid.
        legacy = OmegaConf.create(plain(configs[0]))
        for field in ("recipe", "fsdp_init_mode", "ema_mode", "profile_memory"):
            del legacy[field]
        validate_config(legacy)

    def test_formal_and_smoke_default_to_complete_raw_handoff(self):
        for recipe, budgets in (("14b-fsdp8", [3000, 1000, 2000]),
                                ("14b-fsdp8-smoke", [6, 2, 6])):
            with self.subTest(recipe=recipe):
                configs = self.plan("--recipe", recipe)
                self.assertEqual([item.stage for item in configs], ["s1", "s2", "s3"])
                self.assertEqual([item.max_steps for item in configs], budgets)
                self.assertEqual(configs[0].generator_ckpt, self.assets["distill_init_14b"])
                for index, config in enumerate(configs):
                    self.assertEqual(config.init_checkpoint_key, "generator")
                    self.assertEqual(config.world_size, 8)
                    self.assertEqual(config.effective_batch_size, 8)
                    self.assertEqual(list(config.denoising_step_list), [1000, 750, 500, 250])
                    self.assertEqual(config.sharding_strategy, "full")
                    self.assertEqual(config.fsdp_init_mode, "rank0")
                    self.assertEqual(config.ema_mode, "sharded")
                    self.assertTrue(config.profile_memory)
                    self.assertTrue(config.mixed_precision)
                    self.assertTrue(config.gradient_checkpointing)
                    self.assertFalse(config.no_save)
                    self.assertIn("experimental", config.recipe_id)
                    self.assertEqual(config.ema_start_step, 0 if recipe.endswith("smoke") else 200)
                    if index:
                        previous = configs[index - 1]
                        self.assertEqual(config.generator_ckpt, str(
                            Path(previous.logdir) / f"checkpoint_model_{previous.max_steps:06d}" / "model.pt"
                        ))
                self.assertEqual(list(configs[0].mix_final_probs), [0.5, 0.0, 0.5])
                self.assertEqual(configs[1].distribution_loss, "paired_coupled_dmd")
                self.assertEqual(configs[1].lpips_weight, 0.25)
                self.assertEqual(configs[2].distribution_loss, "dmd")
                self.assertEqual(configs[2].lpips_weight, 0)

    def test_each_standalone_stage_uses_its_14b_asset_key(self):
        for stage, key in (("s1", "distill_init_14b"), ("s2", "s2_init_14b"),
                           ("s3", "s3_init_14b")):
            self.assertEqual(self.config(stage).generator_ckpt, self.assets[key])

    def test_sequential_plan_does_not_require_standalone_init_assets(self):
        minimal = {key: value for key, value in self.assets.items()
                   if key not in ("s2_init_14b", "s3_init_14b")}
        OmegaConf.save(OmegaConf.create(minimal), self.asset_file)
        self.assertEqual(len(self.plan("--recipe", "14b-fsdp8")), 3)

    def test_stage_overrides_update_handoff_without_changing_other_budgets(self):
        configs = self.plan("--recipe", "14b-fsdp8-smoke", "--set", "s1.max_steps=7")
        self.assertEqual([item.max_steps for item in configs], [7, 2, 6])
        self.assertIn("checkpoint_model_000007", configs[1].generator_ckpt)

    def test_topology_and_runtime_contract_cannot_silently_drift(self):
        for ranks in (1, 4, 16, 32):
            with self.subTest(ranks=ranks), self.assertRaisesRegex(ConfigError, "exactly 8 ranks"):
                self.config(world_size=ranks)
        overrides = [
            "generator_name=Wan2.1-T2V-1.3B", "fake_name=Wan2.1-T2V-1.3B",
            "real_name=Wan2.1-T2V-1.3B", "batch_size=2", "sharding_strategy=hybrid_full",
            "fsdp_init_mode=replicated", "ema_mode=full", "mixed_precision=false",
            "gradient_checkpointing=false", "no_save=true", "warp_denoising_step=false",
            "denoising_step_list=[1000,800,600,400,200]", "timestep_shift=6",
            "ts_schedule=true", "image_or_video_shape=[1,21,16,90,160]",
        ]
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ConfigError):
                self.config(overrides=[override])

    def test_invalid_runtime_modes_fail_for_reference_too(self):
        for override in ("ema_mode=replicate", "fsdp_init_mode=all_ranks", "profile_memory=1"):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                self.config(recipe="reference", overrides=[override])

    def test_ema_validation_rejects_bad_values_before_model_loading(self):
        for recipe in ("reference", "14b-fsdp8", "14b-fsdp8-smoke"):
            for value in (-1, 0.5, float("inf"), float("nan"), True, "later"):
                with self.subTest(recipe=recipe, start=value):
                    config = self.config(recipe=recipe)
                    config.ema_start_step = value
                    with self.assertRaisesRegex(ConfigError, "ema_start_step"):
                        validate_config(config)
            for value in (-0.01, 1, float("inf"), float("nan"), True, "disabled"):
                with self.subTest(recipe=recipe, decay=value):
                    config = self.config(recipe=recipe)
                    config.ema_weight = value
                    with self.assertRaisesRegex(ConfigError, "ema_weight"):
                        validate_config(config)
            for value in (None, 0, 0.99):
                with self.subTest(recipe=recipe, valid_decay=value):
                    config = self.config(recipe=recipe)
                    config.ema_weight = value
                    config.ema_start_step = 0
                    validate_config(config)

    def test_cpu_dry_run_needs_no_assets_and_imports_no_torch(self):
        command = [sys.executable, "-c", (
            "import sys; from wf_training.cli import main; "
            "code=main(sys.argv[1:]); assert 'torch' not in sys.modules; sys.exit(code)"
        ), "run", "--recipe", "14b-fsdp8-smoke", "--assets", str(self.asset_file),
            "--output", str(self.root / "run"), "--dry-run"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(len(json.loads(result.stdout)["stages"]), 3)
        self.assertFalse((self.root / "run").exists())

    def native_assets(self):
        model = Path(self.assets["model_root"]) / "Wan2.1-T2V-14B"
        tokenizer = model / "google/umt5-xxl"
        tokenizer.mkdir(parents=True)
        config = {"model_type": "t2v", "dim": 5120, "ffn_dim": 13824,
                  "num_heads": 40, "num_layers": 40, "in_dim": 16, "out_dim": 16}
        native = model / "config.json"
        native.write_text(json.dumps(config))
        for name in ("diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth",
                     "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer_config.json",
                     "google/umt5-xxl/spiece.model"):
            (model / name).write_text("metadata-only test fixture")
        Path(self.assets["prompts"]).write_text("\n".join(f"prompt {index}" for index in range(8)))
        Path(self.assets["distill_init_14b"]).write_text("checkpoint placeholder")
        return native, config

    def test_preflight_checks_native_architecture_without_reading_weights(self):
        native, _ = self.native_assets()
        read_text = Path.read_text

        def guarded_read(path, *args, **kwargs):
            self.assertNotIn(path.suffix, (".pt", ".pth", ".safetensors"))
            return read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_read), patch(
                "wf_training.config.importlib.metadata.distributions", return_value=[]):
            report = preflight(self.config())
        arch = report["assets"]["Wan2.1-T2V-14B"]["architecture"]
        self.assertEqual(arch["num_layers"], 40)
        self.assertEqual(arch["patch_size"], [1, 2, 2])
        self.assertFalse(arch["weights_loaded"])
        self.assertEqual(report["assets"]["Wan2.1-T2V-14B"]["config"]["path"], str(native))

    def test_preflight_rejects_wrong_model_or_incompatible_latents(self):
        native, valid = self.native_assets()
        for key, value in (("num_layers", 30), ("num_heads", 12), ("dim", 1536),
                           ("ffn_dim", 8960), ("in_dim", 36), ("out_dim", 32),
                           ("model_type", "i2v"), ("patch_size", [1, 4, 4]),
                           ("text_dim", 2048)):
            with self.subTest(key=key):
                native.write_text(json.dumps({**valid, key: value}))
                with self.assertRaisesRegex(ConfigError, "14B native config"):
                    preflight(self.config())

    def test_cli_reports_missing_14b_initialization_clearly(self):
        del self.assets["distill_init_14b"]
        with self.assertRaisesRegex(ConfigError, "missing assets: distill_init_14b"):
            self.config()


if __name__ == "__main__":
    unittest.main()
