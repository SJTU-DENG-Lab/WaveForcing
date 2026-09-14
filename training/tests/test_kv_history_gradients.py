"""Real causal-attention regression for detached rolling-forcing history.

Only the attention kernel and RoPE are replaced with CPU implementations. The
production Q/K/V projections, cache writes, cache indices, history assembly,
output projection, and checkpoint recomputation remain intact. Two-token
blocks keep this test small; the production three-frame block policy is not
under test here.
"""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wf_training.wan.modules import causal_model


def cpu_attention(query, key, value, **kwargs):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)


def identity_rope(value, *args, **kwargs):
    return value


class DetachedKVHistoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.base = causal_model.CausalWanSelfAttention(
            dim=8, num_heads=2, qk_norm=False
        ).double()
        self.base.block_length = 2
        self.base.frame_length = 2
        self.base.max_attention_size = 8
        self.cache = {
            "k": torch.randn(1, 8, 2, 4, dtype=torch.float64),
            "v": torch.randn(1, 8, 2, 4, dtype=torch.float64),
            "global_end_index": torch.tensor([2]),
            "local_end_index": torch.tensor([2]),
        }
        self.inputs = [torch.randn(1, 4, 8, dtype=torch.float64) for _ in range(2)]
        patches = patch.multiple(
            causal_model,
            attention=cpu_attention,
            causal_rope_apply=identity_rope,
            _RF_BLOCK_CAUSAL=False,
        )
        patches.start()
        self.addCleanup(patches.stop)

    @staticmethod
    def detached_reference(model, value, cache, current_start):
        query = model.q(value).reshape(1, 4, 2, 4)
        key = model.k(value).reshape(1, 4, 2, 4)
        val = model.v(value).reshape(1, 4, 2, 4)
        # The selected examples use an anchor and at most one history block,
        # so the complete prefix is exactly the production attention history.
        history_key = cache["k"][:, :current_start].detach()
        history_val = cache["v"][:, :current_start].detach()
        result = cpu_attention(
            query,
            torch.cat([history_key, key], dim=1),
            torch.cat([history_val, val], dim=1),
        )
        return model.o(result.flatten(2))

    def run_rollout(self, *, reference, checkpointed, windows):
        model = copy.deepcopy(self.base)
        cache = copy.deepcopy(self.cache)
        inputs = [value.clone().requires_grad_() for value in self.inputs[:windows]]
        outputs = []
        cache_requires_grad = []
        for index, value in enumerate(inputs):
            current_start = 2 + 2 * index
            kwargs = {
                "seq_lens": torch.tensor([4]),
                "grid_sizes": torch.tensor([[2, 1, 2]]),
                "freqs": torch.zeros(1),
                "block_mask": None,
                "current_start": current_start,
                "kv_cache": cache,
            }
            if reference:
                output = self.detached_reference(model, value, cache, current_start)
            elif checkpointed:
                output = checkpoint(model, value, **kwargs, use_reentrant=False)
            else:
                output = model(value, **kwargs)
            outputs.append(output)
            cache_requires_grad.extend(cache[key].requires_grad for key in ("k", "v"))

            # Production immediately reruns the first block with a clean
            # latent under no_grad. Do this before any backward, including
            # after the final window, to exercise checkpoint cache replay.
            with torch.no_grad():
                model(
                    output[:, :2].detach(),
                    seq_lens=torch.tensor([2]),
                    grid_sizes=torch.tensor([[1, 1, 2]]),
                    freqs=torch.zeros(1),
                    block_mask=None,
                    current_start=current_start,
                    kv_cache=cache,
                    updating_cache=True,
                )
            cache_requires_grad.extend(cache[key].requires_grad for key in ("k", "v"))

        sum(output.square().sum() for output in outputs).backward()
        cache_requires_grad.extend(cache[key].requires_grad for key in ("k", "v"))
        return {
            "outputs": [value.detach() for value in outputs],
            "input_grads": [value.grad for value in inputs],
            "parameter_grads": {name: value.grad for name, value in model.named_parameters()},
            "cache_requires_grad": cache_requires_grad,
            "cache": cache,
        }

    def assert_matches_detached_history(self, *, windows, checkpointed):
        expected = self.run_rollout(reference=True, checkpointed=False, windows=windows)
        actual = self.run_rollout(reference=False, checkpointed=checkpointed, windows=windows)
        self.assertFalse(any(actual["cache_requires_grad"]))
        for key in ("outputs", "input_grads"):
            for result, target in zip(actual[key], expected[key], strict=True):
                torch.testing.assert_close(result, target, rtol=1e-12, atol=1e-12)
        self.assertEqual(actual["parameter_grads"].keys(), expected["parameter_grads"].keys())
        for name, result in actual["parameter_grads"].items():
            self.assertIsNotNone(result, name)
            torch.testing.assert_close(
                result, expected["parameter_grads"][name], rtol=1e-12, atol=1e-12
            )
        # The current key/value projections must still receive gradients.
        for name in ("k.weight", "v.weight"):
            self.assertGreater(actual["parameter_grads"][name].abs().sum().item(), 0)
        # These bounded rollouts intentionally avoid eviction. Index equality
        # confirms that clean writes/recomputation retain the same cache range.
        for key in ("global_end_index", "local_end_index"):
            torch.testing.assert_close(actual["cache"][key], expected["cache"][key])

    def test_single_window_matches_detached_history(self):
        self.assert_matches_detached_history(windows=1, checkpointed=False)

    def test_single_checkpointed_window_survives_clean_cache_overwrite(self):
        self.assert_matches_detached_history(windows=1, checkpointed=True)

    def test_multiple_windows_do_not_backpropagate_through_history(self):
        self.assert_matches_detached_history(windows=2, checkpointed=False)

    def test_multiple_checkpointed_windows_match_detached_history(self):
        self.assert_matches_detached_history(windows=2, checkpointed=True)


if __name__ == "__main__":
    unittest.main()
