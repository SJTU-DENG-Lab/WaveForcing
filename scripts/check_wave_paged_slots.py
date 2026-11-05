"""CUDA IPC full-forward publication/reuse stress gate, without model weights.

torchrun --standalone --nproc-per-node=3 -m scripts.check_wave_paged_slots
"""
import argparse
import os

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--chunks', type=int, default=13)
    args = parser.parse_args()
    if min(args.layers, args.chunks) < 1:
        parser.error('Positive layers and chunks required')
    if not torch.cuda.is_available():
        raise RuntimeError('This gate requires CUDA IPC on at least two GPUs')
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if not 2 <= world <= 8:
        parser.error('Use between two and eight ranks')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    from wave_rt.denoiser.engine import WaveDenoiser
    wave = object.__new__(WaveDenoiser)
    wave.rank, wave.world, wave.rf_step, wave.store_rank = rank, world, world-1, world-1
    wave.device, wave.dtype = torch.device('cuda', rank), torch.float32
    wave.npb, wave.Ln, wave.num_blocks = 3, args.layers, args.chunks
    wave.num_ticks, wave.tokens_per_chunk = args.chunks+world-1, 24
    wave.KV_SHAPE, wave.max_attn_frames, wave._paged = (2, 24, 2, 16), 3*(world+4), True
    from types import SimpleNamespace
    wave.be = SimpleNamespace(barrier=dist.barrier)
    wave.cfg = SimpleNamespace(kv_transport_dtype="bf16")
    wave._ipc_handles, wave._closed = [], False
    wave._os_dst, wave._pg_dst, wave._st_ev = {}, {}, {}
    wave._rl_down = wave._rl_kv = wave._rl_flags = wave._st_prefetch = None
    try:
        wave._init_paged()
        # Repeat both forced-full-forward and concurrent producer/consumer runs
        # on the same exports to check generation reset and acquire ordering.
        sources_owned = [torch.empty((2, 24, 2, 16), device=wave.device, dtype=wave.dtype)
                         for _ in range(args.layers)]
        for request in range(4):
            full_forward_boundary = request < 2
            wave._os_scopy.synchronize()
            dist.barrier()
            if wave._pg_consumed is not None:
                wave._pg_consumed.zero_()
                wave._os_flags.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            snapshots, expected = [], []
            migration = torch.zeros(1, device=wave.device)
            migrated = [torch.empty_like(migration) for _ in range(world)]
            for tick in range(wave.num_ticks):
                active = [r for r in range(world-1) if 0 <= tick-r < args.chunks]
                store = 0 <= tick-wave.store_rank < args.chunks
                if rank in active or (rank == wave.store_rank and store):
                    for layer in range(args.layers):
                        number = request*100000+tick*1000+layer*10+rank
                        payload = sources_owned[layer]
                        k, v = payload.unbind(0)
                        k.fill_(number); v.fill_(number + 0.5)
                        torch.cuda._sleep((1+(rank+layer+request)%5)*10000)
                        destinations = [(r, rank) for r in active if r < rank]
                        wave._publish_paged(k, v, destinations, layer, tick%wave._os_gt, tick)
                if full_forward_boundary:
                    # Every publication must FINISH before any consumer may
                    # read/release. An undersized ring cannot pass this phase.
                    wave._os_scopy.synchronize()
                    dist.barrier()
                if rank in active:
                    for layer in range(args.layers):
                        slot, _, _ = wave._pg_slots.address(tick, rank, layer)
                        sources = [r for r in active if r > rank] + ([wave.store_rank] if store else [])
                        wave._wait_paged_flags(sources, layer, tick%wave._os_gt, tick)
                        for src in sources:
                            # Read every byte, not just a sentinel. The clones
                            # emulate attention's final reads on the same stream.
                            snapshots += [wave._pg_k[slot, src*24:(src+1)*24].clone(),
                                          wave._pg_v[slot, src*24:(src+1)*24].clone()]
                            value = request*100000+tick*1000+layer*10+src
                            expected += [value, value+0.5]
                        wave._release_paged(tick, layer)
                wave._drain_publish()
                # Match production's tick boundary, including stream ordering.
                dist.all_gather(migrated, migration)
            torch.cuda.synchronize()
            for actual, value in zip(snapshots, expected):
                torch.testing.assert_close(actual, torch.full_like(actual, value), atol=0, rtol=0)
            dist.barrier()
            if rank == 0:
                print(f'PASS request={request}: {args.layers} layers, {args.chunks} chunks, '
                      f'{wave._pg_slots.depth} payload slots; '
                      f'{"full-forward boundary" if full_forward_boundary else "concurrent acquire/read"}', flush=True)
        wave.close()
    finally:
        # On a rank failure, torchrun owns terminating its peers; don't enter a
        # collective cleanup barrier from the exceptional path.
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
