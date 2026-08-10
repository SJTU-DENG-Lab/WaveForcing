"""Gate C-final-1 PoC: validate the two hardware assumptions one-sided adds
beyond Gate B, cross-process (faithful to the torch.distributed multi-proc run).

  (2) cudaMemcpyPeerAsync GB/s   (expect ~343, Gate B3)
  (3) release/acquire correctness across processes via cuStreamWriteValue32 /
      cuStreamWaitValue32 on an IPC-mapped slab (payload visible-after-flag)
  (4) copy engine does not steal SMs: producer runs SM-heavy compute while
      bursting peer copies; compare kernel wall-time isolated vs concurrent.

Layout: proc0 = producer (GPU0), proc1 = consumer (GPU1).
Consumer allocs remote_kv + gen_flag, IPC-exports; producer writes into them.
"""
import os, sys, time
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wave_rt import p2p_mem as p2p

# Config M unit: KV_SHAPE = (2, tokens_per_chunk, H, D) bf16
KV_SHAPE = (2, 4680, 16, 128)
NBYTES = 2 * 4680 * 16 * 128 * 2  # bf16
ITERS = 50


def consumer(q_to_prod, q_to_cons, dev=1):
    torch.cuda.set_device(dev)
    # allocate the slab + flag ONCE (init-time), IPC export
    remote_kv = torch.zeros(KV_SHAPE, dtype=torch.bfloat16, device=dev)
    gen_flag = torch.zeros(1, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    kv_h = p2p.ipc_get_handle_bytes(remote_kv.data_ptr())
    fl_h = p2p.ipc_get_handle_bytes(gen_flag.data_ptr())
    # base offsets so producer can rebase to the exact slot pointer
    q_to_prod.put({
        "kv_handle": kv_h, "kv_base": remote_kv.data_ptr(),
        "fl_handle": fl_h, "fl_base": gen_flag.data_ptr(),
        "nbytes": NBYTES,
    })

    s_cmp = torch.cuda.Stream(device=dev)
    results = []
    n_elem = remote_kv.numel()
    for k in range(1, ITERS + 1):
        # tell producer to go for generation k
        q_to_cons.get()  # "GO k" signal (payload unused, just a barrier)
        # device-side acquire: wait gen_flag >= k, then read on the SAME stream
        with torch.cuda.stream(s_cmp):
            p2p.stream_wait_u32(s_cmp.cuda_stream, gen_flag.data_ptr(), k)
            checksum = remote_kv.to(torch.float64).sum()
        s_cmp.synchronize()
        # producer wrote every element = k -> expected sum = k * n_elem
        expected = float(k) * n_elem
        got = float(checksum.item())
        ok = abs(got - expected) < 1e-3 * max(1.0, expected)
        results.append((k, ok, got, expected))
        q_to_prod.put(("done", k, ok))
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    q_to_prod.put(("SUMMARY", n_ok, ITERS))
    print(f"[consumer] release/acquire correctness: {n_ok}/{ITERS} generations "
          f"had payload-visible-after-flag", flush=True)
    if n_ok != ITERS:
        for k, ok, got, exp in results:
            if not ok:
                print(f"  [consumer] FAIL gen {k}: got {got} expected {exp}", flush=True)


def producer(q_to_prod, q_to_cons, dev=0, peer=1):
    torch.cuda.set_device(dev)
    torch.zeros(4, device=dev)  # force primary ctx creation / cuInit
    info = q_to_prod.get()  # IPC handles from consumer
    # Retain both primary contexts and enable NVLink peer access self->peer so
    # cuMemcpyPeerAsync uses the copy engine (not host staging).
    ctx_self = p2p.primary_ctx(dev)
    ctx_peer = p2p.primary_ctx(peer)
    p2p.enable_peer_access(ctx_self, ctx_peer)
    # KV slab: open under the PEER (GPU1) context -> dst ptr valid there for the
    # peer copy. gen_flag: open under SELF (GPU0) context (peer-mapped) so a GPU0
    # stream memory op (cuStreamWriteValue32) can target it.
    kv_base_peer = p2p.ipc_open_handle_in_ctx(info["kv_handle"], ctx_peer)
    fl_base_self = p2p.ipc_open_handle(info["fl_handle"])
    # Strategy A alt: also open KV under SELF ctx (peer-mapped into GPU0) for the
    # cuMemcpyDtoDAsync path comparison.
    kv_base_self = p2p.ipc_open_handle(info["kv_handle"])
    nbytes = info["nbytes"]

    def peer_copy(stream):
        p2p.memcpy_peer_async(kv_base_peer, ctx_peer, src.data_ptr(), ctx_self,
                              nbytes, stream)

    def dtod_copy(stream):
        p2p.memcpy_dtod_async(kv_base_self, src.data_ptr(), nbytes, stream)

    s_copy = torch.cuda.Stream(device=dev)
    src = torch.empty(KV_SHAPE, dtype=torch.bfloat16, device=dev)

    # ---- (2) bandwidth: many peer copies back to back ----
    torch.cuda.synchronize()
    warm = 5
    for _ in range(warm):
        peer_copy(s_copy.cuda_stream)
    s_copy.synchronize()
    t0 = time.perf_counter()
    BW_N = 100
    for _ in range(BW_N):
        peer_copy(s_copy.cuda_stream)
    s_copy.synchronize()
    dt = time.perf_counter() - t0
    gbps = (BW_N * nbytes) / dt / 1e9
    per_copy_ms = dt / BW_N * 1e3
    print(f"[producer] (2) peer copy: {gbps:.1f} GB/s, {per_copy_ms:.3f} ms/unit "
          f"({nbytes/2**20:.2f} MiB)", flush=True)

    # (2b) IPC-DtoD (dst opened under self ctx, peer-mapped): bandwidth
    for _ in range(warm):
        dtod_copy(s_copy.cuda_stream)
    s_copy.synchronize()
    t0 = time.perf_counter()
    for _ in range(BW_N):
        dtod_copy(s_copy.cuda_stream)
    s_copy.synchronize()
    dt = time.perf_counter() - t0
    print(f"[producer] (2b) IPC DtoD copy: {BW_N*nbytes/dt/1e9:.1f} GB/s, "
          f"{dt/BW_N*1e3:.3f} ms/unit", flush=True)

    # ---- (3) release/acquire: write pattern=k, copy, set flag=k ----
    for k in range(1, ITERS + 1):
        src.fill_(float(k))
        q_to_cons.put(("GO", k))
        # optional delay to force the consumer's wait to actually stall,
        # proving the flag (not luck) gates visibility
        with torch.cuda.stream(s_copy):
            torch.cuda._sleep(2_000_000)  # ~ short spin on the copy stream's device
        peer_copy(s_copy.cuda_stream)
        # release AFTER the copy on the same stream
        p2p.stream_write_u32(s_copy.cuda_stream, fl_base_self, k)
        tag = q_to_prod.get()  # wait for consumer "done"
    summary = q_to_prod.get()
    print(f"[producer] (3) consumer summary: {summary}", flush=True)

    # ---- (4) SM contention: time the COMPUTE kernel with CUDA events, while
    #      peer copies flood a separate stream. Events isolate the matmul kernel
    #      duration from the copy-stream length. ----
    N = 4096
    a = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
    s_cmp = torch.cuda.Stream(device=dev)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    def time_matmul(rounds=200):
        torch.cuda.synchronize()
        with torch.cuda.stream(s_cmp):
            ev0.record(s_cmp)
            for _ in range(rounds):
                c = a @ b
            ev1.record(s_cmp)
        s_cmp.synchronize()
        return ev0.elapsed_time(ev1)

    time_matmul(20)  # warm
    iso = min(time_matmul(200) for _ in range(3))

    # concurrent: keep s_copy saturated with peer copies spanning the matmul
    torch.cuda.synchronize()
    with torch.cuda.stream(s_copy):
        for _ in range(600):
            peer_copy(s_copy.cuda_stream)
    conc = time_matmul(200)  # matmul kernel time while copies run on s_copy
    s_copy.synchronize()
    slow = (conc - iso) / iso * 100
    print(f"[producer] (4-flood peerAsync) matmul iso={iso:.2f}ms under-copy={conc:.2f}ms "
          f"-> {slow:+.1f}% (target <4%)", flush=True)

    # DtoD flood SM impact (event-based, clean)
    torch.cuda.synchronize()
    with torch.cuda.stream(s_copy):
        for _ in range(600):
            dtod_copy(s_copy.cuda_stream)
    conc_d = time_matmul(200)
    s_copy.synchronize()
    slow_d = (conc_d - iso) / iso * 100
    print(f"[producer] (4-flood DtoD) matmul iso={iso:.2f}ms under-copy={conc_d:.2f}ms "
          f"-> {slow_d:+.1f}%", flush=True)

    # realistic ratio: 4 peer copies (nearest-first fanout) overlapping one
    # attention-sized compute chunk
    torch.cuda.synchronize()
    with torch.cuda.stream(s_copy):
        for _ in range(4):
            peer_copy(s_copy.cuda_stream)
    real = time_matmul(30)
    s_copy.synchronize()
    iso30 = min(time_matmul(30) for _ in range(3))
    slow_r = (real - iso30) / iso30 * 100
    print(f"[producer] (4-real 4copies) matmul iso={iso30:.2f}ms "
          f"under-copy={real:.2f}ms -> {slow_r:+.1f}%", flush=True)
    verdict = "PASS" if slow < 4.0 else ("MARGINAL" if slow < 8 else "FAIL")
    print(f"[producer] (4) SM-contention verdict (flood): {verdict}", flush=True)

    p2p.ipc_close_handle(kv_base_peer)
    p2p.ipc_close_handle(fl_base_self); p2p.ipc_close_handle(kv_base_self)


def main():
    mp.set_start_method("spawn", force=True)
    q_to_prod = mp.Queue()
    q_to_cons = mp.Queue()
    pc = mp.Process(target=consumer, args=(q_to_prod, q_to_cons, 1))
    pp = mp.Process(target=producer, args=(q_to_prod, q_to_cons, 0, 1))
    pc.start(); pp.start()
    pp.join(); pc.join()


if __name__ == "__main__":
    main()
