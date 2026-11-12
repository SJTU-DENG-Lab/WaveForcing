"""Prompt restarts must invalidate the optimized model's actual text caches."""
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch import nn

from wave_rt.interactive.native import switch_prompt
from wave_rt.runtime.conditioning import ConditioningCache
from wave_rt.runtime.compute import FixedConditioning
from wave_rt.runtime.model import WaveParallelWan


class PromptConditioningTest(unittest.TestCase):
    def backend(self, optimized=True):
        model = nn.Module()
        model.hidden_size = 2
        model.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for block in model.blocks:
            block.scale_shift_table = torch.zeros(1, 6, 2)
        model.scale_shift_table = torch.zeros(1, 2, 2)
        model.condition_embedder = NS(
            text_embedder=lambda context: context + 5,
            time_embedder=lambda times, _: times[:, None].expand(-1, 2).float(),
            time_modulation=lambda temb: temb.repeat(1, 6),
        )
        # Exercise production reset/conditioning methods without CUDA or weights.
        adapter = WaveParallelWan.__new__(WaveParallelWan)
        nn.Module.__init__(adapter)
        adapter.model = model
        adapter.joint = NS(reset=Mock())
        adapter.conditioning = ConditioningCache()
        adapter.conditioning.configure(torch.tensor([1]))
        adapter.fixed = FixedConditioning()
        adapter.text_kv = [object(), object()]
        adapter._times = {}
        adapter._step_hint = 1
        adapter._next_normalized = None
        stage = NS(deduplicated_output_fields=('text',))
        native_cache = NS(reset=Mock())
        backend = NS(
            cfg=NS(seed=7, num_frames=24, height=480, width=832),
            pipeline=NS(sampling_params_cls=NS, stages=[lambda b, _: b, stage]),
            shared_conditioning=False, device='cpu', server_args=None,
            batch=NS(), ctx=NS(target_dtype=torch.float32),
            rf_stage=NS(
                _prepare_causal_dmd_prompt_embeds=lambda b, *_: b.text,
                _prepare_causal_dmd_pos_cond_kwargs=lambda *_: {},
                crossattn_cache=[None, native_cache],
            ),
            wp_adapter=adapter if optimized else None,
        )
        # sampling_params_cls is read from the pipeline's class.
        class Pipeline(NS):
            sampling_params_cls = NS
        backend.pipeline = Pipeline(stages=backend.pipeline.stages)
        return backend, adapter, native_cache

    def switch(self, backend, value):
        module_name = 'sglang.multimodal_gen.runtime.pipelines_core.schedule_batch'
        module = types.ModuleType(module_name)
        module.Req = lambda sampling_params: NS(text=torch.full((1, 2, 2), float(sampling_params.prompt)))
        with patch.dict(sys.modules, {module_name: module}), patch(
            'wave_rt.runtime.text_encoder.encode_local_text', side_effect=lambda stage, batch, *_: batch
        ):
            switch_prompt(backend, str(value))

    def test_new_prompt_and_reset_rebuild_projected_text_and_fixed_entries(self):
        backend, adapter, _ = self.backend()
        times = torch.ones(1, 3, dtype=torch.long)
        original = torch.full((1, 2, 2), 1.0)
        warm = adapter._conditioning(times, original)[2]
        self.assertTrue(torch.equal(warm, original + 5))
        # Without invalidation, a different context silently returns warm text.
        self.assertTrue(torch.equal(adapter._conditioning(times, original * 2)[2], warm))
        for value in (2, 1, 3):
            adapter.text_kv = [object(), object()]
            self.switch(backend, value)
            self.assertEqual(adapter.text_kv, [None, None])
            self.assertIsNotNone(adapter.conditioning.timesteps)
            fresh = adapter._conditioning(times, backend.ctx.prompt_embeds)[2]
            self.assertTrue(torch.equal(fresh, torch.full_like(original, value + 5)))
            self.assertEqual(backend.batch.prompt, str(value))

    def test_native_backend_without_adapter_still_resets_native_cache(self):
        backend, _, cache = self.backend(optimized=False)
        self.switch(backend, 2)
        cache.reset.assert_called_once()
        self.assertTrue(torch.equal(backend.ctx.prompt_embeds, torch.full((1, 2, 2), 2.0)))


if __name__ == '__main__':
    unittest.main()
