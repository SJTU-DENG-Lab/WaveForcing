"""Checkpoint validation uses small CPU tensors and never initializes CUDA."""
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch
from torch import nn

from wf_training.utils.checkpoint import (
    generator_state, load_model_checkpoint, validate_load,
)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="wf-checkpoint-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_load_mapped_checkpoint_on_cpu(self):
        path = self.directory / "model.pt"
        torch.save({"generator": {"model.weight": torch.arange(6)}}, path)
        with mock.patch.object(torch, "load", wraps=torch.load) as loader:
            checkpoint = load_model_checkpoint(path)
        tensor = checkpoint["generator"]["model.weight"]
        self.assertEqual(tensor.device.type, "cpu")
        torch.testing.assert_close(tensor, torch.arange(6))
        self.assertEqual([call.kwargs["mmap"] for call in loader.call_args_list], [True])
        self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")

    def test_legacy_checkpoint_falls_back_without_mmap(self):
        path = self.directory / "legacy.pt"
        torch.save({"weight": torch.arange(3)}, path, _use_new_zipfile_serialization=False)
        with mock.patch.object(torch, "load", wraps=torch.load) as loader:
            checkpoint = load_model_checkpoint(path)
        torch.testing.assert_close(checkpoint["weight"], torch.arange(3))
        self.assertEqual([call.kwargs["mmap"] for call in loader.call_args_list], [True, False])
        self.assertTrue(all(call.kwargs["map_location"] == "cpu" for call in loader.call_args_list))

    def test_other_load_failures_are_not_retried(self):
        with mock.patch.object(torch, "load", side_effect=RuntimeError("corrupt archive")) as loader:
            with self.assertRaisesRegex(RuntimeError, "corrupt archive"):
                load_model_checkpoint("unused.pt")
        loader.assert_called_once()

    def test_load_rejects_non_mapping_and_empty_files(self):
        for value in (torch.ones(1), [], {}):
            with self.subTest(value=type(value).__name__):
                path = self.directory / "bad.pt"
                torch.save(value, path)
                with self.assertRaisesRegex(ValueError, "nonempty mapping"):
                    load_model_checkpoint(path)

    def test_generator_formats_keep_storage_and_remove_nested_wrappers(self):
        for container in (None, "generator", "model"):
            with self.subTest(container=container):
                weight = torch.ones(2, 3)
                state = OrderedDict({
                    "_fsdp_wrapped_module.model.blocks.0._checkpoint_wrapped_module._orig_mod.weight": weight,
                })
                checkpoint = state if container is None else {container: state}
                result = generator_state(checkpoint)
                self.assertEqual(list(result), ["model.blocks.0.weight"])
                self.assertIs(result["model.blocks.0.weight"], weight)

    def test_native_wan_export_gains_wrapper_prefix(self):
        state = {"patch_embedding.weight": torch.ones(2), "head.weight": torch.zeros(2)}
        result = generator_state(state)
        self.assertEqual(set(result), {"model.patch_embedding.weight", "model.head.weight"})
        self.assertIs(result["model.head.weight"], state["head.weight"])

    def test_full_resume_prefers_raw_generator_over_ema(self):
        raw = torch.ones(2)
        result = generator_state({
            "generator": {"model.weight": raw},
            "generator_ema": {"model.weight": torch.zeros(2)},
            "critic": {"model.weight": torch.full((2,), 3.0)},
            "metadata": {"stage": "s2"},
        })
        self.assertIs(result["model.weight"], raw)

    def test_generator_rejects_unsafe_or_ambiguous_states(self):
        cases = (
            ({"generator_ema": {"weight": torch.ones(1)}}, "EMA-only"),
            ({"generator": {}, "model": {}}, "ambiguous"),
            ({"generator": {"weight": "invalid"}}, "named tensors"),
            ({"generator": {}}, "nonempty tensor mapping"),
            ({"model.weight": torch.ones(1), "_orig_mod.model.weight": torch.zeros(1)}, "collision"),
            ({"model.head.weight": torch.ones(1), "blocks.0.weight": torch.ones(1)}, "mixes native"),
        )
        for checkpoint, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    generator_state(checkpoint)

    def test_validate_accepts_meta_model_and_bf16_into_fp32(self):
        module = nn.Linear(3, 2, device="meta", dtype=torch.float32)
        validate_load(module, {
            "weight": torch.ones(2, 3, dtype=torch.bfloat16),
            "bias": torch.zeros(2, dtype=torch.bfloat16),
        })
        self.assertTrue(module.weight.is_meta)

    def test_validate_reports_all_incompatibilities_without_mutating_model(self):
        module = nn.Linear(3, 2)
        original = {key: value.clone() for key, value in module.state_dict().items()}
        with self.assertRaises(ValueError) as captured:
            validate_load(module, {"weight": torch.zeros(2, 4), "extra": torch.ones(1)})
        message = str(captured.exception)
        self.assertIn("missing keys: ['bias']", message)
        self.assertIn("unexpected keys: ['extra']", message)
        self.assertIn("weight: checkpoint (2, 4) != model (2, 3)", message)
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, original[key])


if __name__ == "__main__":
    unittest.main()
