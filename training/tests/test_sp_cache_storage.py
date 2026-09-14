"""SP cache capacity is independent of the global frame/block indices."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from wf_training.pipeline.rolling_forcing_training import RollingForcingTrainingPipeline


class SpatialKVStorageTest(unittest.TestCase):
    def make_pipeline(self, sp_size):
        generator = SimpleNamespace(model=SimpleNamespace(
            blocks=[None, None], num_heads=40, dim=5120))
        with patch("wf_training.pipeline.rolling_forcing_training.get_sp_world_size",
                   return_value=sp_size):
            return RollingForcingTrainingPipeline(
                [1000, 750, 500, 250], None, generator,
                num_frame_per_block=3, num_max_frames=27)

    def test_head_partition_reduces_kv_without_shortening_history(self):
        original = self.make_pipeline(1)
        spatial = self.make_pipeline(4)
        self.assertEqual(spatial.kv_cache_size, original.kv_cache_size)
        self.assertEqual(spatial.frame_seq_length, 1560)
        self.assertEqual(spatial.num_frame_per_block, 3)
        self.assertEqual(spatial.num_attention_heads, 10)
        sizes = []
        for pipe in (original, spatial):
            # Tiny storage exercises the real allocator without allocating
            # the production 32 GiB history in a CPU unit test.
            pipe.kv_cache_size = 8
            pipe._initialize_kv_cache(1, torch.bfloat16, "cpu")
            pipe._initialize_crossattn_cache(1, torch.bfloat16, "cpu")
            sizes.append(sum(layer[k].numel() * layer[k].element_size()
                             for layer in pipe.kv_cache_clean for k in ("k", "v")))
            self.assertTrue(all(not layer["is_init"] for layer in pipe.crossattn_cache))
            pointers = [layer["k"].data_ptr() for layer in pipe.kv_cache_clean]
            pipe._initialize_kv_cache(1, torch.bfloat16, "cpu")
            self.assertEqual(pointers, [layer["k"].data_ptr() for layer in pipe.kv_cache_clean])
        self.assertEqual(sizes[0], 4 * sizes[1])

    def test_incompatible_head_count_fails_before_cache_allocation(self):
        with self.assertRaisesRegex(ValueError, "heads must be divisible"):
            self.make_pipeline(3)


if __name__ == "__main__":
    unittest.main()
