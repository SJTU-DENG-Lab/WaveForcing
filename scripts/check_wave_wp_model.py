"""CUDA oracle for optimized WP local compute against unmodified native Wan.

Uses actual weights, RoPE, sgldiff blocks and FA3; two small-grid requests cross
the clean-page eviction boundary. Paged publication is local/no-op: this checks
local compute, not inter-GPU IPC or full generated-video quality. Run the full
paged trajectory gate as well (wave_rt/README.md).
"""
import argparse
import copy
import os
import socket

import torch

from wave_rt.config import WaveConfig
from wave_rt.denoiser.engine import WaveDenoiser


def local_wave():
    from sglang.multimodal_gen.runtime.models.dits import causal_wanvideo as cw
    wave = object.__new__(WaveDenoiser)
    wave.npb, wave.max_attn_frames, wave.store_rank, wave._os_gt = 3, 21, 4, 2
    wave._lp_cur, wave._rope_cache = None, {}
    wave._rope_f64 = cw.current_platform.is_float64_supported()
    wave._get_rotary_pos_embed = cw.get_rotary_pos_embed
    wave._apply_rotary_emb = cw._apply_rotary_emb
    wave._publish_paged = lambda *args: None
    return wave


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--gen-ckpt', required=True)
    parser.add_argument('--atol', type=float, default=.02)
    parser.add_argument('--rtol', type=float, default=.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA and actual FA3 are required; no CPU fallback')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run on one GPU; this oracle holds two full DiT copies')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    if 'MASTER_PORT' not in os.environ:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            os.environ['MASTER_PORT'] = str(listener.getsockname()[1])
    os.environ.setdefault('WAVE_THREADS', '1')
    from wave_rt.runtime.backend import WaveBackend
    from wave_rt.distributed.compat import ensure_causal_transformer_config
    from wave_rt.runtime.model import WaveParallelWan
    ensure_causal_transformer_config(args.model_path)
    cfg = WaveConfig(vae_stages=0, dit_only=True,
                     model_path=args.model_path, gen_ckpt=args.gen_ckpt,
                     height=32, width=64, num_frames=36, attention_backend='fa3')
    # Local compute oracle only: one native replica, no Wave scheduler.
    cfg.wp_size = 1
    be = WaveBackend(cfg, 0)
    try:
        be.init()
        native = be.transformer
        optimized = WaveParallelWan(copy.deepcopy(native)).eval()
        for request, prompt in enumerate(('A corgi on a beach.', 'A red train in snow.')):
            be.rf_stage.transformer = native
            be.prepare_request(prompt, 817 + request, 36, 32, 64 + 32*request)
            optimized.reset()
            context_noise = float(be.server_args.pipeline_config.context_noise)
            optimized.conditioning.configure(torch.cat((be.dsl, be.dsl.new_tensor([context_noise]))))
            ph, pw = native.patch_size[1:]
            h, w = be.ctx.height//ph, be.ctx.width//pw
            optimized.prepare_geometry(36, h, w, be.device)
            histories = [[[] for _ in native.blocks] for _ in range(2)]
            waves = [local_wave(), local_wave()]
            from wave_rt.runtime.compute import AxisRoPE
            for wave in waves:
                wave.rank, wave.device, wave.dtype = 4, be.device, torch.bfloat16
                wave.tokens_per_chunk = 3*h*w
                wave.KV_SHAPE = (2, 3*h*w, native.num_attention_heads,
                                 native.hidden_size//native.num_attention_heads)
                wave._anchor_axes = AxisRoPE(39, h, w, wave.KV_SHAPE[-1], be.device,
                    compute_device=be.device,
                    dtype=torch.float64 if wave._rope_f64 else torch.float32)
            worst_output = worst_kv = 0.
            for chunk in range(10):
                # Keep identical inputs to isolate local implementation errors.
                x = be.full_noise_bcthw[:, :, (chunk % 12)*3:(chunk % 12+1)*3].contiguous()
                for clean in (True, False):
                    start = (chunk if clean else chunk + 1) * 3
                    step = context_noise if clean else float(be.dsl[chunk % be.dsl.numel()])
                    snapshots, outputs = [], []
                    for index, model in enumerate((native, optimized)):
                        wave, history, store, seen = waves[index], histories[index], {}, []
                        wave._cur_tick = chunk + 4
                        if clean and index == 0:
                            # Independent dense/native oracle: do not share the
                            # new context-assembly kernel or history allocator.
                            def exchange(*, layer_idx, roped_key, unroped_key, value,
                                         post_patch_height, post_patch_width, dim, rope_num_heads, **kw):
                                store[layer_idx] = roped_key, unroped_key, value
                                keys, values = wave._anchor_working(history[layer_idx], start, 3,
                                    post_patch_height, post_patch_width, dim, rope_num_heads)
                                return torch.cat(keys + [roped_key], 1), torch.cat(values + [value], 1)
                        elif clean:
                            exchange = wave._make_store_exch_paged(history, store, chunk, [0, 1, 2, 3])
                        else:
                            def exchange(*, layer_idx, roped_key, value, post_patch_height,
                                         post_patch_width, dim, rope_num_heads, **kw):
                                keys, values = wave._anchor_working(history[layer_idx], start, 3,
                                    post_patch_height, post_patch_width, dim, rope_num_heads)
                                return torch.cat(keys + [roped_key], 1), torch.cat(values + [value], 1)
                        def capture(**kw):
                            seen.append(tuple(kw[name].detach().clone() for name in
                                              ('unroped_key', 'roped_key', 'value')))
                            return exchange(**kw)
                        capture.requires_raw_key = True  # Diagnostic snapshots only.
                        if hasattr(exchange, 'prepare_targets'):
                            capture.prepare_targets = exchange.prepare_targets
                        be.set_exchange_fn(capture)
                        be.rf_stage.transformer = model
                        ctx = be.ctx
                        try:
                            output = be.rf_stage._forward_causal_transformer(
                                be.batch, latent_model_input=x, prompt_embeds=ctx.prompt_embeds,
                                timestep=torch.full((1, 3), step, device=be.device).long(),
                                kv_cache=be.rf_stage.causal_kv_cache,
                                crossattn_cache=be.rf_stage.crossattn_cache,
                                current_start_tokens=start*be.num_token_per_frame, start_frame=start,
                                image_kwargs=ctx.image_kwargs, pos_cond_kwargs=ctx.pos_cond_kwargs,
                                current_timestep=0, attn_metadata=None, target_dtype=ctx.target_dtype,
                                autocast_enabled=ctx.autocast_enabled, updating_cache=clean)
                        finally:
                            be.set_exchange_fn(None)
                        snapshots.append(seen)
                        outputs.append(output)
                        if clean:
                            assert len(store) == len(native.blocks), 'Missing clean layer publication'
                            for layer, (kr, ku, value) in store.items():
                                history[layer].append(dict(kr=kr, ku=ku, v=value))
                                if len(history[layer]) > 8:
                                    history[layer] = history[layer][:1] + history[layer][-7:]
                    for actual_layer, expected_layer in zip(snapshots[1], snapshots[0]):
                        for actual, expected in zip(actual_layer, expected_layer):
                            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                            worst_kv = max(worst_kv, (actual.float()-expected.float()).abs().max().item())
                    if clean:
                        assert outputs[1] is None, 'Pruned store executed prediction head'
                    else:
                        torch.testing.assert_close(outputs[1], outputs[0], atol=args.atol, rtol=args.rtol)
                        worst_output = max(worst_output, (outputs[1].float()-outputs[0].float()).abs().max().item())
            print(f'PASS request={request} max_abs_output={worst_output:.6g} max_abs_kv={worst_kv:.6g}', flush=True)
    finally:
        be.set_exchange_fn(None)
        be.shutdown()


if __name__ == '__main__':
    main()
