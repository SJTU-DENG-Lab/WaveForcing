"""CUDA numerical gates for Wave kernels and latent staging."""
import argparse
import queue
import torch

def check_wp(device):
    from wave_rt.runtime.compute import AxisRoPE
    from wave_rt.runtime.kernels import prepare_qkv, norm_modulate, assemble_context
    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import _apply_rotary_emb
    torch.manual_seed(314)
    for dtype in (torch.bfloat16, torch.float32):
        atol, rtol = ((.032, .008) if dtype == torch.bfloat16 else (2e-5, 2e-5))
        for b, s, h, d in ((2, 6, 12, 128), (1, 15, 40, 128), (1, 15, 1, 64)):
            axes = AxisRoPE(35, 1, s//3, d, device)
            storage = torch.randn(b, s, 3, h, d, device=device, dtype=dtype)
            q, k, v = storage.unbind(2)
            qn, kn = (RMSNorm(h*d, eps=1e-6).to(device=device, dtype=dtype) for _ in range(2))
            qn.weight.data.uniform_(.8, 1.2); kn.weight.data.uniform_(.8, 1.2)
            nq = qn(q.flatten(2)).reshape(q.shape)
            nk = kn(k.flatten(2)).reshape(k.shape)
            for start in (0, 5, 32):
                cos, sin = axes.grid(3, start)
                expected_q = _apply_rotary_emb(nq, cos, sin, is_neox_style=False)
                expected_k = _apply_rotary_emb(nk, cos, sin, is_neox_style=False)
                for kv_only in (False, True):
                    for raw in (False, True):
                        # Guard bytes before/after both final-address destinations.
                        payload = torch.full((2, b*s+16, h, d), -71., device=device, dtype=dtype)
                        targets = tuple(p[8:8+b*s].view(b, s, h, d) for p in payload)
                        oq, raw_k, ov, ok = prepare_qkv(q, k, v, qn, kn, axes, start,
                            targets=targets, raw=raw, kv_only=kv_only)
                        assert ok.data_ptr() == targets[0].data_ptr() and ov.data_ptr() == targets[1].data_ptr()
                        torch.testing.assert_close(ok, expected_k, atol=atol, rtol=rtol)
                        torch.testing.assert_close(ov, v, atol=0, rtol=0)
                        torch.testing.assert_close(oq, expected_k if kv_only else expected_q, atol=atol, rtol=rtol)
                        if raw:
                            torch.testing.assert_close(raw_k, nk, atol=atol, rtol=rtol)
                        else:
                            assert raw_k is None
                        assert bool((payload[:, :8] == -71).all() & (payload[:, -8:] == -71).all())
            x = torch.randn(b, s, h*d, device=device, dtype=dtype)
            shift, scale = [torch.randn(b, 3, 1, h*d, device=device) for _ in range(2)]
            expected = torch.nn.functional.layer_norm(x.float(), (h*d,), eps=1e-6)
            expected = (expected.unflatten(1, (3, -1))*(1+scale)+shift).flatten(1, 2).to(dtype)
            torch.testing.assert_close(norm_modulate(x, shift, scale, 1e-6), expected, atol=atol, rtol=rtol)
            if b == 1:
                for count in (1, 3, 8):
                    keys = [torch.randn_like(k).contiguous() for _ in range(count)]
                    values = [torch.randn_like(v).contiguous() for _ in range(count)]
                    for start in (-3, 5, 32):
                        out = torch.full((2, (count+1)*s, h, d), -71., device=device, dtype=dtype)
                        cos, sin = axes.grid(3, start)
                        expected_k = torch.cat([_apply_rotary_emb(keys[0], cos, sin, is_neox_style=False)]+keys[1:], 1)
                        assemble_context(keys, values, out[0], out[1], axes, start)
                        torch.testing.assert_close(out[0, :count*s].unsqueeze(0), expected_k, atol=atol, rtol=rtol)
                        torch.testing.assert_close(out[1, :count*s].unsqueeze(0), torch.cat(values, 1), atol=0, rtol=0)
                        assert bool((out[:, count*s:] == -71).all())
                    # Promotion uses the same two-output kernel without rotation.
                    assemble_context(keys[:1], values[:1], out[0], out[1], axes, 0, anchor=False)
                    torch.testing.assert_close(out[0, :s].unsqueeze(0), keys[0], atol=0, rtol=0)
    torch.cuda.synchronize()
    print('CUDA WP final-address QKV, norm/modulation, axis-RoPE and assembly gates passed')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA and Triton are required; no CPU fallback')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    check_wp(device)
    from wave_rt.pipelines.latent_output import LatentQueueWriter, latent_from_payload
    channel = queue.Queue()
    writer = LatentQueueWriter(channel, device)
    expected = []
    try:
        for chunk in range(17):
            value = torch.randn(1,16,3,9,11,device=device,dtype=torch.bfloat16)
            expected.append(value.permute(0,2,1,3,4).clone())
            writer.submit(chunk,value)
    finally:
        writer.close()
    for chunk,value in enumerate(expected):
        received,payload = channel.get_nowait()
        assert received == chunk
        torch.testing.assert_close(latent_from_payload(payload,device),value,atol=0,rtol=0)
    assert channel.empty()
    print('PASS Wave fused kernels and exact asynchronous BF16 staging')

if __name__ == '__main__':
    main()
