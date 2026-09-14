"""CPU tests for storage reuse and checkpoint-safe prediction boundaries."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from torch.utils.checkpoint import checkpoint

from wf_training.utils.cache import TrainingKVCache, offpolicy_cache_frames


class TrainingKVCacheTest(unittest.TestCase):
    shape = (1, 6, 2, 4)

    def make_cache(self):
        cache = TrainingKVCache()
        cache.reset_kv(2, self.shape, torch.float32, "cpu")
        cache.reset_crossattn(2)
        return cache

    def test_reuses_storage_and_resets_all_layers_between_no_grad_predictions(self):
        cache = self.make_cache()
        pointers = [layer[key].data_ptr() for layer in cache.kv for key in ("k", "v")]
        for layer in cache.kv:
            layer["k"].fill_(7)
            layer["v"].fill_(8)
            layer["global_end_index"].fill_(9)
            layer["local_end_index"].fill_(6)
        cache.reserve_for_backward(torch.ones(1))
        # Repeated critic/no-grad rollouts need no backward/release call.
        with patch.object(torch, "zeros", side_effect=AssertionError("unexpected allocation")):
            cache.reset_kv(2, self.shape, torch.float32, "cpu")
        self.assertEqual(
            pointers, [layer[key].data_ptr() for layer in cache.kv for key in ("k", "v")])
        for layer in cache.kv:
            for value in layer.values():
                self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_reset_is_blocked_until_checkpointed_backward_finishes(self):
        cache = self.make_cache()
        cache.kv[0]["k"].fill_(2)
        original_pointer = cache.kv[0]["k"].data_ptr()
        x = torch.linspace(0, 1, cache.kv[0]["k"].numel()).reshape(self.shape).requires_grad_()
        loss = checkpoint(lambda value: (value * cache.kv[0]["k"]).sin().sum(),
                          x, use_reentrant=False)
        cache.reserve_for_backward(loss)
        with self.assertRaisesRegex(RuntimeError, "before.*backward"):
            cache.reset_kv(2, self.shape, torch.float32, "cpu")
        with self.assertRaisesRegex(RuntimeError, "before.*backward"):
            cache.reset_crossattn(2)
        loss.backward()
        torch.testing.assert_close(x.grad, 2 * torch.cos(2 * x.detach()))

        cache.release_after_backward()
        # Release itself never mutates tensors, including during exception cleanup.
        torch.testing.assert_close(cache.kv[0]["k"], torch.full(self.shape, 2.0))
        cache.reset_kv(2, self.shape, torch.float32, "cpu")
        self.assertEqual(original_pointer, cache.kv[0]["k"].data_ptr())
        self.assertEqual(torch.count_nonzero(cache.kv[0]["k"]).item(), 0)

    def test_reset_detaches_cache_write_graph_without_reallocating(self):
        cache = self.make_cache()
        weight = torch.tensor(2.0, requires_grad=True)
        pointer = cache.kv[0]["k"].data_ptr()
        cache.kv[0]["k"][:] = weight * 2
        loss = (cache.kv[0]["k"] * 3).sum()
        cache.reserve_for_backward(loss)
        loss.backward()
        self.assertEqual(weight.grad.item(), 6 * cache.kv[0]["k"].numel())
        cache.release_after_backward()
        self.assertIsNotNone(cache.kv[0]["k"].grad_fn)
        cache.reset_kv(2, self.shape, torch.float32, "cpu")
        self.assertEqual(pointer, cache.kv[0]["k"].data_ptr())
        self.assertIsNone(cache.kv[0]["k"].grad_fn)
        self.assertFalse(cache.kv[0]["k"].requires_grad)

    def test_replacement_releases_entire_old_cache_before_allocating(self):
        cache = self.make_cache()
        old_refs = [weakref.ref(value) for layer in cache.kv for value in layer.values()]
        allocate = torch.zeros

        def checked_allocate(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in old_refs))
            return allocate(*args, **kwargs)

        with patch.object(torch, "zeros", side_effect=checked_allocate):
            cache.reset_kv(2, (1, 9, 2, 4), torch.float64, "cpu")
        self.assertEqual(cache.kv[0]["k"].shape, (1, 9, 2, 4))
        self.assertEqual(cache.kv[0]["k"].dtype, torch.float64)

    def test_cross_attention_invalidates_values_without_placeholder_allocations(self):
        cache = self.make_cache()
        slots = [id(layer) for layer in cache.crossattn]
        cache.crossattn[0].update(k=torch.ones(self.shape), v=torch.ones(self.shape), is_init=True)
        old_key = weakref.ref(cache.crossattn[0]["k"])
        with patch.object(torch, "zeros", side_effect=AssertionError("unexpected allocation")):
            cache.reset_crossattn(2)
        self.assertIsNone(old_key())
        self.assertEqual(slots, [id(layer) for layer in cache.crossattn])
        self.assertEqual(cache.crossattn, [{"k": None, "v": None, "is_init": False}] * 2)

    def test_offpolicy_capacity_keeps_pair_context_and_bounds_long_pairs(self):
        for steps in (4, 5):
            with self.subTest(steps=steps):
                self.assertEqual(offpolicy_cache_frames(21, 26, steps, 3), 21)
                self.assertEqual(offpolicy_cache_frames(120, 26, steps, 3), (26 + steps) * 3)

    def test_pipeline_uses_actual_generator_dimensions_and_lifetime_guard(self):
        # Load this module directly to avoid pipeline/__init__ importing the GPU
        # model/teacher stack. The production cache and pipeline code are intact.
        path = Path(__file__).parents[1] / "wf_training/pipeline/rolling_forcing_training.py"
        spec = importlib.util.spec_from_file_location("test_rolling_pipeline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        generator = SimpleNamespace(model=SimpleNamespace(blocks=[None] * 2, num_heads=2, dim=8))
        pipe = module.RollingForcingTrainingPipeline([1000, 500], None, generator, num_max_frames=21)
        self.assertEqual(pipe.kv_cache_size, 21 * 1560)
        pipe.kv_cache_size = 6  # Tiny tensors for lifecycle testing.
        pipe._initialize_kv_cache(1, torch.float32, "cpu")
        pipe._initialize_crossattn_cache(1, torch.float32, "cpu")
        self.assertEqual(pipe.kv_cache_clean[0]["k"].shape, self.shape)
        pointer = pipe.kv_cache_clean[0]["k"].data_ptr()
        loss = torch.ones(1, requires_grad=True).square().sum()
        pipe.reserve_caches_for_backward(loss)
        with self.assertRaises(RuntimeError):
            pipe._initialize_kv_cache(1, torch.float32, "cpu")
        loss.backward()
        pipe.release_caches()
        pipe._initialize_kv_cache(1, torch.float32, "cpu")
        self.assertEqual(pointer, pipe.kv_cache_clean[0]["k"].data_ptr())


if __name__ == "__main__":
    unittest.main()
