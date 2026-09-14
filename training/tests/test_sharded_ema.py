"""CPU checks for local-shard EMA arithmetic, recovery, and temporary exports."""

import copy
from contextlib import ExitStack
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from wf_training.utils.ema import ShardedEMA, local_parameters


class TinyShard(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())


def tiny_model(dtype=torch.float32):
    return TinyShard(torch.tensor([1.0, 2.0, 3.0], dtype=dtype),
                     torch.tensor([4.0], dtype=dtype))


class ShardedEMATest(unittest.TestCase):
    def test_local_updates_equal_full_fp32_ema_including_empty_shards(self):
        weights = torch.linspace(-1, 1, 8).to(torch.bfloat16)
        biases = torch.tensor([2.0, -3.0], dtype=torch.bfloat16)
        modules = [
            TinyShard(weights[:3], biases[:1]),
            TinyShard(weights[3:6], biases[1:]),
            TinyShard(weights[6:], biases[:0]),
        ]
        decay = 0.75
        averages = [ShardedEMA(module, decay=decay) for module in modules]
        reference = {"weight": weights.float().clone(), "bias": biases.float().clone()}
        self.assertEqual(averages[2].shadow["bias"].numel(), 0)
        for step in range(1, 5):
            with torch.no_grad():
                for rank, module in enumerate(modules):
                    for parameter in module.parameters():
                        parameter.add_((step + rank) / 16)
            for name in reference:
                full_parameter = torch.cat([getattr(module, name).detach().float()
                                            for module in modules])
                reference[name].mul_(decay).add_(full_parameter, alpha=1 - decay)
            for module, average in zip(modules, averages):
                average.update(module)
            for name in reference:
                combined = torch.cat([average.shadow[name] for average in averages])
                torch.testing.assert_close(combined, reference[name], rtol=0, atol=0)
                for average in averages:
                    self.assertEqual(average.shadow[name].dtype, torch.float32)
                    self.assertEqual(average.shadow[name].device.type, "cpu")
                    self.assertFalse(average.shadow[name].requires_grad)

    def test_serialized_state_restores_independent_fp32_shards(self):
        module = tiny_model(torch.bfloat16)
        average = ShardedEMA(module, decay=0.5)
        with torch.no_grad():
            module.weight.add_(2)
        average.update(module)
        stream = io.BytesIO()
        torch.save(average.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = ShardedEMA(module, decay=0.5, initialize=False)
        self.assertEqual(restored.shadow, {})
        restored.load_state_dict(saved, module)
        for name in average.shadow:
            torch.testing.assert_close(restored.shadow[name], average.shadow[name])
            self.assertNotEqual(restored.shadow[name].data_ptr(), saved["shadow"][name].data_ptr())
        saved["shadow"]["weight"].fill_(99)
        torch.testing.assert_close(restored.shadow["weight"], average.shadow["weight"])

    def test_state_validation_rejects_wrong_format_decay_names_shapes_and_dtype(self):
        module = tiny_model()
        valid = ShardedEMA(module, decay=0.5).state_dict()
        bad_states = []
        wrong = copy.deepcopy(valid)
        wrong["format"] = "full_ema"
        bad_states.append((wrong, "local sharded"))
        wrong = copy.deepcopy(valid)
        wrong["decay"] = 0.9
        bad_states.append((wrong, "decay"))
        wrong = copy.deepcopy(valid)
        del wrong["shadow"]["bias"]
        bad_states.append((wrong, "names"))
        wrong = copy.deepcopy(valid)
        wrong["shadow"]["weight"] = torch.zeros(2)
        bad_states.append((wrong, "shape mismatch"))
        wrong = copy.deepcopy(valid)
        wrong["shadow"]["weight"] = wrong["shadow"]["weight"].to(torch.bfloat16)
        bad_states.append((wrong, "FP32"))
        wrong = copy.deepcopy(valid)
        wrong["shadow"]["weight"] = [1, 2, 3]
        bad_states.append((wrong, "FP32"))
        for state, message in bad_states:
            with self.subTest(message=message):
                average = ShardedEMA(module, decay=0.5, initialize=False)
                with self.assertRaisesRegex(ValueError, message):
                    average.load_state_dict(state, module)

    def test_update_rejects_a_changed_local_shard_topology(self):
        module = tiny_model()
        average = ShardedEMA(module)
        module.weight = nn.Parameter(torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            average.update(module)
        module.weight = nn.Parameter(torch.zeros(3))
        module.register_parameter("extra", nn.Parameter(torch.zeros(1)))
        with self.assertRaisesRegex(ValueError, "names"):
            average.update(module)

    def test_applied_to_restores_raw_weights_on_success_and_failure(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                module = tiny_model(torch.bfloat16)
                average = ShardedEMA(module, decay=0.5)
                with torch.no_grad():
                    for parameter in module.parameters():
                        parameter.add_(3)
                raw = {name: parameter.detach().clone()
                       for name, parameter in module.named_parameters()}
                try:
                    with average.applied_to(module):
                        for name, parameter in module.named_parameters():
                            torch.testing.assert_close(parameter, average.shadow[name].to(torch.bfloat16))
                            self.assertTrue(parameter.requires_grad)
                        if raises:
                            raise RuntimeError("export failed")
                except RuntimeError as error:
                    self.assertTrue(raises)
                    self.assertEqual(str(error), "export failed")
                for name, parameter in module.named_parameters():
                    torch.testing.assert_close(parameter, raw[name], rtol=0, atol=0)
                    self.assertEqual(parameter.dtype, torch.bfloat16)
                    self.assertEqual(average.shadow[name].dtype, torch.float32)

    def test_applied_to_restores_refreshed_parameter_views_and_empty_shards(self):
        module = TinyShard(torch.ones(3), torch.empty(0))
        average = ShardedEMA(module)
        with torch.no_grad():
            module.weight.fill_(5)
        with average.applied_to(module):
            # A full FSDP state_dict export can replace its local parameter views.
            module.weight = nn.Parameter(module.weight.detach().clone())
            module.bias = nn.Parameter(module.bias.detach().clone())
            torch.testing.assert_close(module.weight, torch.ones(3))
        torch.testing.assert_close(module.weight, torch.full((3,), 5.0))
        self.assertEqual(module.bias.numel(), 0)

    def test_normalizes_fsdp_names_and_rejects_ambiguous_names(self):
        wrapper = nn.Module()
        wrapper.add_module("_fsdp_wrapped_module", tiny_model())
        self.assertEqual(set(local_parameters(wrapper)), {"weight", "bias"})
        wrapper.register_parameter("weight", nn.Parameter(torch.zeros(1)))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            local_parameters(wrapper)

    def test_initialization_update_and_apply_never_gather_full_parameters(self):
        module = tiny_model()
        with ExitStack() as stack:
            for target in (
                "torch.distributed.all_gather",
                "torch.distributed.all_gather_into_tensor",
                "torch.distributed.all_reduce",
                "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
            ):
                stack.enter_context(patch(target, side_effect=AssertionError("unexpected collective")))
            stack.enter_context(patch.object(module, "state_dict", side_effect=AssertionError("unexpected gather")))
            average = ShardedEMA(module)
            average.update(module)
            restored = ShardedEMA(module, initialize=False)
            restored.load_state_dict(average.state_dict(), module)
            with restored.applied_to(module):
                pass


if __name__ == "__main__":
    unittest.main()
