"""CPU numerical/lifecycle regressions; CUDA gates live in scripts/check_wave_*."""
import argparse
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from wave_rt.config import WaveConfig
from wave_rt.denoiser.engine import WaveDenoiser
from wave_rt.denoiser.exchange.slots import PagedSlots
from wave_rt.runtime.attention import Attention
from wave_rt.runtime.model import WaveParallelWan
from wave_rt.runtime.projection import PackedProjection, ProjectionPart
from wave_rt.runtime.schedule import RFSchedule


class TupleLinear(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class Embedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text_embedder = nn.Linear(dim, dim)
        self.time = nn.Embedding(1001, dim)
        self.time_modulation = nn.Linear(dim, 6*dim)

    def time_embedder(self, times, _):
        return self.time(times.long())


def modulate(x, shift, scale):
    return (x.unflatten(1, (shift.shape[1], -1))*(1+scale)+shift).flatten(1, 2)


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_num_heads, self.dim_head = 2, dim//2
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_q, self.norm_k = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.to_q, self.to_k, self.to_v, self.to_out = [TupleLinear(dim, dim) for _ in range(4)]
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim)*.1)
        self.ffn = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.attn1 = nn.Module()
        self.attn1.forward = self.dense
        self.attn2 = nn.Module()
        self.attn2.local_num_heads, self.attn2.head_dim = 2, dim//2
        for name in ('to_q','to_k','to_v','to_out'):
            setattr(self.attn2, name, TupleLinear(dim, dim))
        self.attn2.norm_q, self.attn2.norm_k = nn.RMSNorm(dim), nn.RMSNorm(dim)

    @staticmethod
    def dense(q, k, v, *args, write_only=False, **kwargs):
        if write_only:
            return None
        return F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2)

    def mlp_residual(self, addition, gate, residual):
        return residual + modulate(addition, torch.zeros_like(gate), gate-1)

    def self_attn_residual_norm(self, x, addition, gate, shift, scale):
        residual = self.mlp_residual(addition, gate, x)
        return self.norm1(residual), residual

    def cross_attn_residual_norm(self, x, addition, gate, shift, scale):
        residual = x + addition
        return modulate(self.norm1(residual), shift, scale), residual


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.num_attention_heads, self.patch_size = 8, 2, (1,2,2)
        self.patch_embedding = nn.Conv3d(2,8,(1,2,2),stride=(1,2,2))
        self.blocks = nn.ModuleList([Block(8),Block(8)])
        self.condition_embedder = Embedder(8)
        self.scale_shift_table = nn.Parameter(torch.randn(1,2,8)*.1)
        self.proj_out = nn.Linear(8,8)

    def norm_out(self, x, shift, scale):
        return modulate(F.layer_norm(x,(8,)),shift,scale)


def pack_for_cpu(model):
    """Use real PackedProjection.part without requiring the SGLang loader."""
    for block in model.blocks:
        owner = PackedProjection.__new__(PackedProjection)
        nn.Module.__init__(owner)
        owner.widths = [8]*3
        owner.linear = TupleLinear(8,24)
        with torch.no_grad():
            owner.linear.weight.copy_(torch.cat([getattr(block,n).weight for n in ('to_q','to_k','to_v')]))
            owner.linear.bias.copy_(torch.cat([getattr(block,n).bias for n in ('to_q','to_k','to_v')]))
        owner.linear.quant_method = SimpleNamespace(apply=lambda view,x,bias: F.linear(x,view.weight,bias))
        owner.reset()
        block._wave_qkv = owner
        for i,name in enumerate(('to_q','to_k','to_v')):
            setattr(block,name,ProjectionPart(owner,i))


@torch.no_grad()
def dense_reference(model, latent, text, times):
    """Uncached/unpacked full forward with explicit residual arithmetic."""
    b,_,frames,height,width = latent.shape
    x = model.patch_embedding(latent).flatten(2).transpose(1,2)
    emb = model.condition_embedder
    temb = emb.time_embedder(times.flatten(),None)
    proj = emb.time_modulation(temb).reshape(b,frames,6,8)
    context = emb.text_embedder(text)
    for block in model.blocks:
        shift,scale,gate,cshift,cscale,cgate = (proj+block.scale_shift_table).chunk(6,2)
        normalized = modulate(block.norm1(x),shift,scale)
        q,k,v = [getattr(block,n)(normalized)[0] for n in ('to_q','to_k','to_v')]
        out = Block.dense(block.norm_q(q).unflatten(2,(2,4)),block.norm_k(k).unflatten(2,(2,4)),v.unflatten(2,(2,4)))
        x = x + modulate(block.to_out(out.flatten(2))[0],torch.zeros_like(gate),gate-1)
        cross = block.attn2
        q = cross.norm_q(cross.to_q(block.norm1(x))[0]).unflatten(2,(2,4))
        k = cross.norm_k(cross.to_k(context)[0]).unflatten(2,(2,4))
        v = cross.to_v(context)[0].unflatten(2,(2,4))
        x = x + cross.to_out(Block.dense(q,k,v).flatten(2))[0]
        out = block.ffn(modulate(block.norm1(x),cshift,cscale))
        x = x + modulate(out,torch.zeros_like(cgate),cgate-1)
    shift,scale = (model.scale_shift_table.unsqueeze(1)+temb.reshape(b,frames,1,8)).chunk(2,2)
    x = model.proj_out(model.norm_out(x,shift,scale)).reshape(b,frames,height//2,width//2,1,2,2,2)
    return x.permute(0,7,1,4,2,5,3,6).flatten(6,7).flatten(4,5).flatten(2,3)


class WaveOptimizationTests(unittest.TestCase):
    def test_relay_retains_store_dispatch_and_publish_drain(self):
        engine = object.__new__(WaveDenoiser)
        engine.cfg = WaveConfig(kv_context='causal',exchange_mode='relay')
        engine._relay = True
        engine._paged = engine._onesided = engine._staggered = False
        with patch.object(engine,'_make_store_exch_relay',return_value='relay') as store, patch.object(engine,'_drain_publish') as drain:
            self.assertEqual(engine._select_store_exchange([],{},2,[0,1]),'relay')
            store.assert_called_once_with([],{},2,[0,1])
            engine._drain_tick_exchange()
            drain.assert_called_once()

    def test_packed_cached_forward_matches_dense_across_requests(self):
        torch.manual_seed(73)
        native = TinyWan().eval()
        with patch('wave_rt.runtime.model.pack_wan_qkv',pack_for_cpu):
            optimized = WaveParallelWan(copy.deepcopy(native),backend='sdpa').eval()
        optimized.rope = lambda *args: (None,None)  # isolate local compute from CUDA RoPE
        for request,width in enumerate((4,6)):
            optimized.reset()
            optimized.conditioning.configure(torch.tensor([0.,250.,500.]))
            text = torch.randn(1,5,8)
            latent = torch.randn(1,2,3,4,width)
            for step in (500,250,500):
                times = optimized.timestep_tensor(1,3,step,'cpu')
                expected = dense_reference(native,latent,text,times)
                actual = optimized(latent,text,times)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            with patch.object(optimized,'_run_block',wraps=optimized._run_block) as blocks, patch.object(optimized.model.proj_out,'forward',wraps=optimized.model.proj_out.forward) as head:
                times = optimized.timestep_tensor(1,3,0,'cpu')
                self.assertIsNone(optimized(latent,text,times,updating_cache=True))
                self.assertEqual(blocks.call_count,len(native.blocks)-1)
                head.assert_not_called()
            self.assertIsNone(optimized._active)
            self.assertIsNone(optimized._next_normalized)

    def test_schedule_preserves_integer_flow_and_fractional_noise_lookup(self):
        scheduler = SimpleNamespace(timesteps=torch.tensor([1000.,750.2,500.4,250.6,0.]),sigmas=torch.tensor([1.,.75,.5,.25,0.]))
        steps = torch.tensor([1000.,750.1,500.3,250.5])
        cache = RFSchedule(scheduler,steps,'cpu')
        x,n = [torch.randn(2,3,4,4).bfloat16() for _ in range(2)]
        for step in steps.tolist():
            index = (scheduler.timesteps-torch.tensor(step)).abs().argmin()
            sigma = scheduler.sigmas[index]
            expected = ((1-sigma.reshape(1,1,1,1))*x+sigma.reshape(1,1,1,1)*n).to(n.dtype)
            torch.testing.assert_close(cache.add_noise_at(x,n,step),expected,atol=0,rtol=0)
        times = torch.tensor([750,250])
        sigma = scheduler.sigmas.double()[(scheduler.timesteps.double()[None]-times[:,None]).abs().argmin(1)].reshape(-1,1,1,1)
        torch.testing.assert_close(cache.x0(n,x,times),(x.double()-sigma*n.double()).to(n.dtype),atol=0,rtol=0)

    def test_fa_dispatch_is_explicit_and_errors_propagate(self):
        for name,module in (('fa2','flash_attn'),('fa3','flash_attn_interface')):
            with patch('wave_rt.runtime.attention.importlib.import_module') as load:
                attention = Attention(name)
                load.assert_called_once_with(module)
                attention(1,2,3)
                load.return_value.flash_attn_func.assert_called_once_with(1,2,3,causal=False,return_attn_probs=False)
                load.return_value.flash_attn_func.side_effect = RuntimeError('kernel failed')
                with self.assertRaisesRegex(RuntimeError,'kernel failed'): attention(1,2,3)

    def test_credit_reuse_waits_for_previous_forward(self):
        slots = PagedSlots(40)
        for consumer in range(5):
            for tick in range(consumer,consumer+15):
                addresses = [slots.address(tick,consumer,l) for l in range(40)]
                self.assertEqual(len({a[0] for a in addresses}),40)
                for layer,(slot,sequence,wait) in enumerate(addresses):
                    self.assertEqual(slot,layer)
                    self.assertEqual(wait,0 if tick==consumer else slots.address(tick-1,consumer,layer)[1])

    def test_working_pages_do_not_overwrite_retained_store_history(self):
        engine = object.__new__(WaveDenoiser)
        engine.rank = engine.store_rank = 4
        engine.device,engine.dtype = 'cpu',torch.float32
        engine.tokens_per_chunk,engine.KV_SHAPE = 6,(2,6,2,4)
        history=[]
        anchor = engine._working_page(0,0,torch.zeros(6,2,4),torch.ones(6,2,4))
        for chunk in range(1,26):
            value = torch.full((6,2,4),float(chunk))
            page = engine._working_page(0,chunk,value,-value)
            for previous,number in history[-7:]:
                self.assertTrue(torch.equal(previous[0],torch.full_like(value,float(number))))
            history.append((page,chunk))
        self.assertEqual(len(engine._working_pages),9)
        self.assertTrue(torch.equal(anchor[0],torch.zeros_like(anchor[0])))

    def test_cli_stays_wave_only(self):
        parser=argparse.ArgumentParser()
        WaveConfig.add_cli_args(parser)
        cfg=WaveConfig.from_args(parser.parse_args(['--attention-backend','fa3','--kv-context','causal','--exchange-mode','paged']))
        self.assertTrue(cfg.uses_optimized_wan and cfg.uses_paged_wan)
        self.assertNotIn('parallel_mode',vars(cfg))
        self.assertEqual(WaveConfig(attention_backend='fa_custom').attention_backend,'fa2')
        with self.assertRaises(ValueError): WaveConfig(exchange_mode='paged')


if __name__ == '__main__':
    unittest.main()
