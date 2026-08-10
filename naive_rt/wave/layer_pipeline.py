"""
WaveLayerPipeline v4 -- per-layer wavefront, reuses the naive real kv_cache buffer for finalized context.

Compared to v3 (hand-rolled chunk-list SWA), v4:
  - Store writes buffer via the **real generator** (updating_cache=True)
    -> buffer is byte-identical to naive (chunk0 stores un-roped anchor;
    rolling eviction goes through the original naive code).
  - Denoised finalized context is extracted from the buffer using **naive's exact
    extraction** (causal_model.py:264-282): [anchor(re-roped) + working(SWA)],
    then concatenates the current tick's in-flight KV.
Only the "in-flight KV" is new; the rest == naive -> should approach naive
regardless of whether eviction is active.
"""
import time
import torch


class WaveLayerPipeline:
    def __init__(self, base_pipeline, context_noise=0.0, verbose=False, det_renoise=False):
        p = base_pipeline
        self.det_renoise = det_renoise
        self.p = p
        self.gen = p.generator
        self.model = p.generator.model
        self.text_encoder = p.text_encoder
        self.vae = p.vae
        self.scheduler = p.scheduler
        self.dsl = p.denoising_step_list
        self.npb = p.num_frame_per_block
        self.fsl = p.frame_seq_length                                  # frame_length 1560
        self.L = p.num_transformer_blocks
        self.block_len = self.npb * self.fsl                           # tokens/chunk 4680
        self.max_attn = self.model.blocks[0].self_attn.max_attention_size  # 21*1560
        self.context_noise = context_noise
        self.verbose = verbose
        self.tick_times = []

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _embed_chunk(self, latent_block, t_row):
        from naive_rt.rolling_forcing.wan.modules.model import sinusoidal_embedding_1d
        m = self.model
        xl = [m.patch_embedding(u.unsqueeze(0)) for u in latent_block.permute(0, 2, 1, 3, 4)]
        grid = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in xl])
        x = torch.cat([u.flatten(2).transpose(1, 2) for u in xl])
        e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t_row.flatten()).type_as(x))
        e0 = m.time_projection(e).unflatten(1, (6, m.dim)).unflatten(dim=0, sizes=t_row.shape)
        return x, e, e0, grid

    def _qkv(self, block, x, e0, grid, start_frame):
        from naive_rt.rolling_forcing.wan.modules.causal_model import causal_rope_apply
        F = e0.shape[1]; fs = x.shape[1] // F
        e = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
        sa_in = (block.norm1(x).unflatten(1, (F, fs)) * (1 + e[1]) + e[0]).flatten(1, 2)
        sa = block.self_attn
        b, s = sa_in.shape[:2]; n, d = sa.num_heads, sa.head_dim
        q = sa.norm_q(sa.q(sa_in)).view(b, s, n, d)
        k = sa.norm_k(sa.k(sa_in)).view(b, s, n, d)
        v = sa.v(sa_in).view(b, s, n, d)
        rq = causal_rope_apply(q, grid, self.model.freqs, start_frame=start_frame).type_as(v)
        rk = causal_rope_apply(k, grid, self.model.freqs, start_frame=start_frame).type_as(v)
        return rq, rk, k, v, e

    def _finish(self, block, x, e, rq, ctxK, ctxV, context_text, cc_L):
        from naive_rt.rolling_forcing.wan.modules.attention import attention
        F = e[0].shape[1]; fs = x.shape[1] // F
        y = block.self_attn.o(attention(rq, ctxK, ctxV).flatten(2))
        x = x + (y.unflatten(1, (F, fs)) * e[2]).flatten(1, 2)
        x = x + block.cross_attn(block.norm3(x), context_text, None, crossattn_cache=cc_L)
        yff = block.ffn((block.norm2(x).unflatten(1, (F, fs)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (yff.unflatten(1, (F, fs)) * e[5]).flatten(1, 2)
        return x

    def _finalized_ctx(self, kvL, grid1, current_start_frame, query_frames):
        """Extract [anchor(re-roped) + working(SWA)] from the naive real buffer kvL.
        Reproduces causal_model.py:264-282."""
        from naive_rt.rolling_forcing.wan.modules.causal_model import causal_rope_apply
        fl, bl = self.fsl, self.block_len
        local_end = kvL["local_end_index"].item()
        if local_end == 0:
            return [], []
        query_length = query_frames * fl
        working_max = self.max_attn - query_length - bl
        extract_end = local_end
        extract_start = max(bl, local_end - working_max)     # exclude the first anchor block
        wk = kvL["k"][:, extract_start:extract_end]
        wv = kvL["v"][:, extract_start:extract_end]
        working_frame_len = wk.shape[1] // fl
        rope_start = current_start_frame - working_frame_len - self.npb
        anchor_k = causal_rope_apply(kvL["k"][:, :bl], grid1, self.model.freqs, start_frame=rope_start).type_as(wv)
        return [anchor_k, wk], [kvL["v"][:, :bl], wv]

    @torch.no_grad()
    def generate(self, noise, text_prompts, initial_latent=None):
        from schedule import build_schedule
        assert initial_latent is None
        B, num_frames, C, H, W = noise.shape
        npb, fsl, K = self.npb, self.fsl, len(self.dsl)
        num_blocks = num_frames // npb
        device, dtype = noise.device, noise.dtype

        cond = self.text_encoder(text_prompts=text_prompts)
        context_text = self.model.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.model.text_len - u.size(0), u.size(1))]) for u in cond["prompt_embeds"]]))

        output = torch.zeros_like(noise)
        noisy_cache = torch.zeros_like(noise)
        # real naive buffer (store writes, denoise reads)
        self.p.kv_cache_clean = None
        self.p._initialize_kv_cache(B, dtype, device)
        self.p._initialize_crossattn_cache(B, dtype, device)
        kv = self.p.kv_cache_clean
        cc = self.p.crossattn_cache
        dsl = self.dsl.to(device)
        sched = build_schedule(num_blocks, self.dsl.tolist(), npb)

        for tick, units in enumerate(sched):
            t0 = time.perf_counter()
            order = [u.chunk for u in units]
            current_start_frame = order[0] * npb
            query_frames = len(order) * npb

            st = {}
            for u in units:
                b = u.chunk; sf, ef = b * npb, (b + 1) * npb
                blk_in = noise[:, sf:ef] if u.stage == 0 else noisy_cache[:, sf:ef]
                t_row = torch.full([B, npb], float(u.step), device=device, dtype=torch.float32)
                x, e_time, e0, grid = self._embed_chunk(blk_in, t_row)
                st[b] = dict(u=u, x=x, e_time=e_time, e0=e0, grid=grid, xt=blk_in)
            grid1 = st[order[0]]['grid']

            for L in range(self.L):
                block = self.model.blocks[L]
                qkv = {b: self._qkv(block, st[b]['x'], st[b]['e0'], st[b]['grid'], b * npb) for b in order}
                aK, aV = self._finalized_ctx(kv[L], grid1, current_start_frame, query_frames)
                ctxK = torch.cat(aK + [qkv[b][1] for b in order], dim=1)
                ctxV = torch.cat(aV + [qkv[b][3] for b in order], dim=1)
                for b in order:
                    rq, rk, ku, v, e = qkv[b]
                    st[b]['x'] = self._finish(block, st[b]['x'], e, rq, ctxK, ctxV, context_text, cc[L])

            for b in order:
                s = st[b]
                xh = self.model.head(s['x'], s['e_time'].unflatten(dim=0, sizes=(B, npb)).unsqueeze(2))
                flow = torch.stack(self.model.unpatchify(xh, s['grid']))
                x0 = self.gen._convert_flow_pred_to_x0(
                    flow_pred=flow.permute(0, 2, 1, 3, 4).flatten(0, 1), xt=s['xt'].flatten(0, 1),
                    timestep=torch.full([B, npb], float(s['u'].step), device=device).flatten(0, 1),
                ).unflatten(0, (B, npb))
                s['x0'] = x0
                output[:, b * npb:(b + 1) * npb] = x0

            cur_nf = len(order) * npb
            dw = torch.cat([st[b]['x0'] for b in order], dim=1)
            for i, b in enumerate(order):
                if st[b]['u'].is_last_stage:
                    continue
                stage = st[b]['u'].stage
                next_t = dsl[stage + 1]
                if self.det_renoise:
                    # match dist: per-chunk seed (1000*chunk+stage), single-chunk randn
                    x0b = st[b]['x0']
                    g = torch.Generator(device=device).manual_seed(1000 * b + stage)
                    nz = torch.randn(x0b.flatten(0, 1).shape, generator=g, device=device, dtype=x0b.dtype)
                    noisy_cache[:, b * npb:(b + 1) * npb] = self.scheduler.add_noise(
                        x0b.flatten(0, 1), nz, next_t * torch.ones([B * npb], device=device, dtype=torch.long)
                    ).unflatten(0, (B, npb))
                else:
                    noisy_cache[:, b * npb:(b + 1) * npb] = self.scheduler.add_noise(
                        dw.flatten(0, 1), torch.randn_like(dw.flatten(0, 1)),
                        next_t * torch.ones([B * cur_nf], device=device, dtype=torch.long)
                    ).unflatten(0, dw.shape[:2])[:, i * npb:(i + 1) * npb]

            # store: real generator writes buffer (byte-identical to naive)
            for b in order:
                if not st[b]['u'].is_last_stage:
                    continue
                ctx_ts = torch.full([B, npb], float(self.context_noise), device=device, dtype=torch.float32)
                self.gen(noisy_image_or_video=st[b]['x0'], conditional_dict=cond, timestep=ctx_ts,
                         kv_cache=kv, crossattn_cache=cc, current_start=b * npb * fsl, updating_cache=True)

            torch.cuda.synchronize()
            self.tick_times.append((time.perf_counter() - t0) * 1000)
            self._log(f"[wavelayer-v4] tick {tick}: {[f'C{u.chunk}S{u.stage}' for u in units]} {self.tick_times[-1]:.1f}ms")

        return output
