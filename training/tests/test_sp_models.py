"""Spatial-SP Wan correctness checks with real tiny diffusion backbones.

Fast layout guards run in normal unittest discovery. The distributed runner
compares complete SP1/SP4 models (including real RoPE, modulation, Q/K RMSNorm,
cache updates, checkpoint recomputation and Adam updates):

  python -m torch.distributed.run --standalone --nproc_per_node=4 training/tests/test_sp_models.py --distributed
  python -m torch.distributed.run --standalone --nproc_per_node=8 training/tests/test_sp_models.py --distributed --backend nccl --fsdp

Only attention kernels use SDPA for a numerical reference. Collective traffic
uses the selected real Gloo/NCCL backend. Global parameter-gradient averaging
matches the FSDP reduction convention. The additional --fsdp mode instead wraps
real FULL_SHARD FSDP around the complete model and its transformer blocks, and
compares materialized full gradients/parameters after backward/Adam updates.
Both SP1 and SP4 use identical FSDP wrapping in that mode. FSDP reconstructs
input dicts, so its text-cache flags do not persist across calls and text K/V
are recomputed. Bare-model cache reuse is diagnosed separately, not treated
as an assumption about which FSDP parameter gradients must be zero.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from functools import partial
import copy
import json
import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask
from torch.nn.attention import sdpa_kernel, SDPBackend

from wf_training.utils import sequence_parallel as sp
from wf_training.wan.modules import model as wan
from wf_training.wan.modules import causal_model as causal


def _sdpa(q, k, v, *, k_lens=None, attn_mask=None, **kwargs):
    if k_lens is not None:
        keep = torch.arange(k.shape[1], device=k.device)[None] < k_lens.to(k.device)[:, None]
        padding = keep[:, None, None, :]
        attn_mask = padding if attn_mask is None else attn_mask.masked_fill(~padding, float('-inf'))
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
    ).transpose(1, 2)


def _dense_flex(query, key, value, block_mask):
    mask = create_mask(
        block_mask.mask_mod, query.shape[0], query.shape[1],
        query.shape[2], key.shape[2], device=query.device,
    )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


@contextmanager
def _reference_sp1():
    # Do not modify process groups: every process independently evaluates its
    # own unsharded reference, before all ranks enter the SP collectives.
    with ExitStack() as stack:
        for module in (wan, causal):
            stack.enter_context(patch.object(module, 'get_sp_world_size', return_value=1))
            stack.enter_context(patch.object(module, 'sequence_to_head_qkv',
                                             side_effect=lambda q, k, v, f: (q, k, v)))
            stack.enter_context(patch.object(module, 'head_to_sequence', side_effect=lambda x, f: x))
        yield


class SpatialModelGuardTests(unittest.TestCase):
    def test_spatial_grid_preserves_global_frame_count(self):
        with patch.object(wan, 'get_sp_world_size', return_value=4):
            self.assertEqual(wan._spatial_num_frames(torch.tensor([[3, 2, 4]]), 24), 3)
            self.assertEqual(wan._spatial_num_frames(torch.tensor([[3, 2, 4]]), 48, copies=2), 6)

    def test_spatial_grid_rejects_ambiguous_padding_and_ragged_batches(self):
        with patch.object(wan, 'get_sp_world_size', return_value=4):
            for grids, length in (([[3, 2, 4]], 32), ([[3, 2, 3]], 18),
                                  ([[3, 2, 4], [2, 3, 4]], 24)):
                with self.assertRaises(ValueError):
                    wan._spatial_num_frames(torch.tensor(grids), length)


def _tiny_model(causal_model, device, checkpointed, dtype=torch.float64):
    torch.manual_seed(914)
    cls = causal.CausalWanModel if causal_model else wan.WanModel
    model = cls(
        patch_size=(1, 2, 2), text_len=5, in_dim=2, dim=32, ffn_dim=64,
        freq_dim=16, text_dim=12, out_dim=2, num_heads=4, num_layers=2,
        qk_norm=True, cross_attn_norm=True,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.1)
    model.gradient_checkpointing = checkpointed
    if causal_model:
        model.num_frame_per_block = 3
        for block in model.blocks:
            block.self_attn.frame_length = 8
            block.self_attn.block_length = 24
            block.self_attn.max_attention_size = 21 * 8
    return model


def _cache(model, sp_size, device, dtype=torch.float64):
    kv = [dict(k=torch.zeros(1, 12 * 8, 4 // sp_size, 8, dtype=dtype, device=device),
               v=torch.zeros(1, 12 * 8, 4 // sp_size, 8, dtype=dtype, device=device),
               global_end_index=torch.zeros(1, dtype=torch.long, device=device),
               local_end_index=torch.zeros(1, dtype=torch.long, device=device))
          for _ in model.blocks]
    cross = [dict(is_init=False) for _ in model.blocks]
    return kv, cross


def _run_case(model, *, kind, device, sp_size, dtype=torch.float64, diagnostics=None):
    # A different sample for each SP group also exercises WORLD gradient
    # averaging over two independent data replicas in the 8-rank runner.
    torch.manual_seed(700 + sp.get_data_parallel_rank())
    context = [torch.randn(4, 12, dtype=dtype, device=device)]
    frames = 3 if kind == 'sf' else 6
    inputs = [torch.randn(1, 2, frames, 4, 8, dtype=dtype, device=device,
                          requires_grad=True) for _ in range(2)]
    time = torch.linspace(100, 700, frames, dtype=dtype, device=device).unsqueeze(0)
    outputs = []
    kv = None
    if kind in ('rf', 'crf', 'sf'):
        kv, cross = _cache(model, sp_size, device, dtype)
        with torch.no_grad():
            model(inputs[0][:, :, :3], t=time[:, :3], context=context, seq_len=128,
                  kv_cache=kv, crossattn_cache=cross, current_start=0, cache_start=0)
        if diagnostics is not None:
            diagnostics['cross_cache_after_warm'] = [bool(cache['is_init']) for cache in cross]
        for index, value in enumerate(inputs):
            start = (index + 1) * 24
            output = model(value, t=time, context=context, seq_len=128,
                           kv_cache=kv, crossattn_cache=cross,
                           current_start=start, cache_start=start)
            outputs.append(output)
            # Match rollout's clean overwrite before the eventual backward.
            with torch.no_grad():
                model(output[:, :, :3].detach() * 0.1, t=torch.zeros_like(time[:, :3]),
                      context=context, seq_len=128, kv_cache=kv,
                      crossattn_cache=cross, current_start=start, cache_start=start,
                      updating_cache=True)
        if diagnostics is not None:
            diagnostics['cross_cache_after_rollout'] = [bool(cache['is_init']) for cache in cross]
    else:
        kwargs = {}
        if kind == 'teacher_forcing':
            kwargs['clean_x'] = inputs[1]
            kwargs['aug_t'] = time * 0.1
        outputs = [model(inputs[0], t=time[:, 0] if kind == 'bidirectional' else time,
                         context=context, seq_len=128, **kwargs)]
    output = torch.stack(outputs)
    target = torch.linspace(-0.5, 0.5, output.numel(), device=device, dtype=output.dtype).reshape_as(output)
    loss = (output - target).square().mean()
    loss.backward()
    return output.detach(), loss.detach(), kv, inputs


def _assert_close(actual, expected, *, name, atol=2e-6, rtol=2e-5):
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f'{name}: {exc}') from exc
    return (actual - expected).abs().max().item() if actual.numel() else 0.0


def run_distributed(backend, use_fsdp=False, dtype_name=None):
    if backend == 'nccl':
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        # Reference diagnostics must compare the same full-precision arithmetic
        # across the different GEMM and attention shapes introduced by SP.
        # Production training precision and kernels are intentionally unchanged.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device('cpu')
    if use_fsdp and backend != 'nccl':
        raise ValueError('--fsdp requires NCCL and CUDA')
    torch.set_num_threads(1)
    dist.init_process_group(backend)
    sp.initialize_sequence_parallel(4)
    rank, world = dist.get_rank(), dist.get_world_size()
    if use_fsdp and world != 8:
        raise ValueError('--fsdp validates the requested 8-rank FSDP / SP4 topology')
    dtype = getattr(torch, dtype_name) if dtype_name else (torch.float32 if use_fsdp else torch.float64)
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    results = []
    with ExitStack() as stack:
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        if rank == 0:
            print(json.dumps({'reference_arithmetic': dict(
                dtype=str(dtype), sdpa_backend='math',
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                atol=2e-6, rtol=2e-5, adam_atol=3e-6,
            )}), flush=True)
        stack.enter_context(patch.object(wan, 'flash_attention', side_effect=_sdpa))
        stack.enter_context(patch.object(causal, 'attention', side_effect=_sdpa))
        stack.enter_context(patch.object(causal, '_get_flex_attention', return_value=_dense_flex))
        stack.enter_context(patch.object(causal, '_RF_USE_FLEX', False))
        for kind in ('bidirectional', 'causal', 'teacher_forcing', 'rf', 'crf', 'sf'):
            for checkpointed in (False, True):
                with patch.object(causal, '_RF_BLOCK_CAUSAL', kind == 'crf'):
                    reference = _tiny_model(kind != 'bidirectional', device, checkpointed, dtype)
                    parallel = copy.deepcopy(reference)
                    unused_in_bare_sp1 = set()
                    bare_diagnostics, reference_diagnostics, parallel_diagnostics = {}, {}, {}
                    if use_fsdp:
                        if kind in ('rf', 'crf', 'sf') and not checkpointed:
                            # Bare calls reuse the detached text cache. FSDP's
                            # _to_kwargs reconstructs its dict, so is_init writes
                            # do not persist and text K/V are recomputed instead.
                            # Record that distinction rather than interpreting
                            # bare None gradients as expected FSDP zeros.
                            bare_probe = copy.deepcopy(reference)
                            with _reference_sp1():
                                probe_result = _run_case(
                                    bare_probe, kind=kind, device=device, sp_size=1, dtype=dtype,
                                    diagnostics=bare_diagnostics)
                            unused_in_bare_sp1 = {
                                name for name, param in bare_probe.named_parameters()
                                if param.grad is None
                            }
                            del probe_result, bare_probe

                        def wrap_fsdp(module):
                            return FSDP(
                                module, device_id=device, use_orig_params=True,
                                sharding_strategy=ShardingStrategy.FULL_SHARD,
                                auto_wrap_policy=partial(
                                    transformer_auto_wrap_policy,
                                    transformer_layer_cls={wan.WanAttentionBlock, causal.CausalWanAttentionBlock},
                                ),
                                limit_all_gathers=True,
                            )

                        # Keep optimizer and unused-gradient semantics identical
                        # on both sides; only spatial SP differs. The four SP1
                        # replicas per data group average to the same DP batch.
                        reference = wrap_fsdp(reference)
                        parallel = wrap_fsdp(parallel)
                    actual_optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-4)
                    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
                    with _reference_sp1():
                        expected, expected_loss, full_cache, reference_inputs = _run_case(
                            reference, kind=kind, device=device, sp_size=1, dtype=dtype,
                            diagnostics=reference_diagnostics)
                    actual, loss, shard_cache, parallel_inputs = _run_case(
                        parallel, kind=kind, device=device, sp_size=4, dtype=dtype,
                        diagnostics=parallel_diagnostics)
                if kind in ('rf', 'crf', 'sf'):
                    expected_flag = not use_fsdp
                    for label, diagnostic in (('SP1', reference_diagnostics), ('SP4', parallel_diagnostics)):
                        for moment, flags in diagnostic.items():
                            assert flags and all(flag == expected_flag for flag in flags), (kind, label, moment, flags)
                    for flags in bare_diagnostics.values():
                        assert flags and all(flags), (kind, 'bare', flags)
                    if rank == 0:
                        print(json.dumps({'cache_semantics': dict(
                            kind=kind, checkpoint=checkpointed, bare=bare_diagnostics,
                            sp1=reference_diagnostics, sp4=parallel_diagnostics,
                            reference_different_cache_semantics=bool(unused_in_bare_sp1),
                        )}), flush=True)
                output_error = _assert_close(actual, expected, name=f'{kind} output')
                _assert_close(loss, expected_loss, name=f'{kind} loss')
                grad_error = 0.0
                bare_unused_gradient_diagnostics = []
                with ExitStack() as full_grads:
                    if use_fsdp:
                        full_grads.enter_context(FSDP.summon_full_params(
                            reference, with_grads=True, writeback=False))
                        full_grads.enter_context(FSDP.summon_full_params(
                            parallel, with_grads=True, writeback=False))
                    expected_params = {
                        name.replace('_fsdp_wrapped_module.', ''): param
                        for name, param in reference.named_parameters()
                    }
                    seen = set()
                    for name, param in parallel.named_parameters():
                        name = name.replace('_fsdp_wrapped_module.', '')
                        expected_param = expected_params[name]
                        seen.add(name)
                        if name in unused_in_bare_sp1:
                            diagnostic = {'parameter': name}
                            for label, grad in (('sp1_fsdp', expected_param.grad), ('sp4_fsdp', param.grad)):
                                diagnostic[label] = None if grad is None else dict(
                                    finite=bool(torch.isfinite(grad).all()),
                                    nonzero=int(torch.count_nonzero(grad)),
                                    numel=grad.numel(), max_abs=grad.abs().max().item(),
                                )
                                if grad is not None:
                                    assert torch.isfinite(grad).all(), f'{kind} {label} nonfinite {name}'
                            bare_unused_gradient_diagnostics.append(diagnostic)
                            if rank == 0:
                                print(json.dumps({'bare_unused_gradient': diagnostic, 'kind': kind}), flush=True)
                        if expected_param.grad is None:
                            assert param.grad is None, name
                            continue
                        assert param.grad is not None, name
                        # Replicated loss + gather SUM + WORLD FSDP mean. In
                        # --fsdp the reduction has already happened in FSDP.
                        if not use_fsdp:
                            dist.all_reduce(param.grad)
                            param.grad.div_(world)
                            dist.all_reduce(expected_param.grad)
                            expected_param.grad.div_(world)
                        grad_error = max(grad_error, _assert_close(param.grad, expected_param.grad, name=f'{kind} {name} grad'))
                    assert seen == set(expected_params)
                for index, (value, ref_value) in enumerate(zip(parallel_inputs, reference_inputs)):
                    if ref_value.grad is None:
                        assert value.grad is None
                    else:
                        # Input gradients are local spatial contributions;
                        # unlike parameters they are not reduced by FSDP.
                        dist.all_reduce(value.grad, group=sp.get_sp_group())
                        value.grad.div_(4)
                        _assert_close(value.grad, ref_value.grad, name=f'{kind} input {index} grad')
                if full_cache is not None:
                    for layer, (shard, full) in enumerate(zip(shard_cache, full_cache)):
                        for name in ('k', 'v'):
                            assert not shard[name].requires_grad
                            expected_shard = full[name].chunk(4, dim=2)[sp.get_sp_rank()]
                            _assert_close(shard[name], expected_shard, name=f'{kind} layer {layer} cache {name}')
                        for name in ('global_end_index', 'local_end_index'):
                            torch.testing.assert_close(shard[name], full[name])
                actual_optimizer.step()
                reference_optimizer.step()
                update_error = 0.0
                with ExitStack() as full_params:
                    if use_fsdp:
                        full_params.enter_context(FSDP.summon_full_params(reference, writeback=False))
                        full_params.enter_context(FSDP.summon_full_params(parallel, writeback=False))
                    expected_params = {
                        name.replace('_fsdp_wrapped_module.', ''): param
                        for name, param in reference.named_parameters()
                    }
                    for name, param in parallel.named_parameters():
                        name = name.replace('_fsdp_wrapped_module.', '')
                        update_error = max(update_error, _assert_close(param, expected_params[name], name=f'{kind} Adam {name}', atol=3e-6))
                item = dict(kind=kind, checkpoint=checkpointed, output_max_abs=output_error,
                            gradient_max_abs=grad_error, adam_max_abs=update_error,
                            unused_bare_parameters=len(unused_in_bare_sp1),
                            reference_different_cache_semantics=bool(unused_in_bare_sp1),
                            bare_unused_gradient_diagnostics=bare_unused_gradient_diagnostics)
                results.append(item)
                if rank == 0:
                    print(json.dumps({'case_passed': item}), flush=True)
                dist.barrier()
    if rank == 0:
        print(json.dumps({'result': 'PASS', 'backend': backend, 'world_size': world, 'sp_size': 4, 'fsdp': use_fsdp, 'dtype': str(dtype),
                          'cases': results}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--backend', choices=('gloo', 'nccl'), default='gloo')
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--dtype', choices=('float32', 'float64'))
    args, remaining = parser.parse_known_args()
    if args.distributed:
        run_distributed(args.backend, args.fsdp, args.dtype)
    else:
        unittest.main(argv=[__file__, *remaining])
