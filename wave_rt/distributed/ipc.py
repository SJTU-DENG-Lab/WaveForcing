"""One-sided P2P memory primitives for Gate C-final-1 (onesided exchange mode).

Uses the CUDA **driver API** (cuda.bindings.driver) throughout. Rationale: the
installed driver is CUDA 12.9 (570.x); the cuda-bindings *runtime* API links a
CUDA-13 cudart and rejects the older driver with cudaErrorInsufficientDriver.
The driver API negotiates its version at cuInit and works on 12.9.

Primitives:
  - cross-process IPC export/import of a device buffer (cuIpc*MemHandle)
  - device-side generation flag release/acquire (cuStreamWriteValue32/WaitValue32)
  - copy-engine peer copy (cuMemcpyDtoDAsync between IPC-mapped peer pointers,
    peer access enabled -> runs on the copy engine, 0 SM, Gate B3)

Device pointers are plain ints (torch `tensor.data_ptr()` or the CUdeviceptr from
cuIpcOpenMemHandle); streams are plain ints (torch `Stream.cuda_stream`).

This mirrors Gate B4's validated st.release.sys generation + ld.acquire spin,
lifted to the stream / cuda-python level so a producer process can publish into a
consumer process's slab with no matching recv.
"""

from cuda.bindings import driver as _drv

_WAIT_GEQ = int(_drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ.value)
_IPC_LAZY = int(_drv.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS.value)


def _ck(ret):
    if isinstance(ret, tuple):
        err, rest = ret[0], ret[1:]
    else:
        err, rest = ret, ()
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver error: {err} ({getattr(err, 'name', err)})")
    if len(rest) == 1:
        return rest[0]
    return rest


# ---------------------------------------------------------------------------
# generation flag: release (producer) / acquire (consumer), device-side
# ---------------------------------------------------------------------------
def stream_write_u32(stream_handle: int, addr: int, value: int) -> None:
    """Enqueue on `stream` a 32-bit write of `value` to device `addr` (release).
    Ordered after any prior peer copy on the same stream -> consumer never sees
    the flag before the payload bytes (Gate-B st.release.sys)."""
    _ck(_drv.cuStreamWriteValue32(stream_handle, addr, value, 0))


def stream_wait_u32(stream_handle: int, addr: int, value: int) -> None:
    """Enqueue on `stream` a device-side wait until *addr >= value (acquire).
    CPU returns immediately; GPU stalls the stream only if flag not yet set.
    Direct replacement for NCCL work.wait()."""
    _ck(_drv.cuStreamWaitValue32(stream_handle, addr, value, _WAIT_GEQ))


# ---------------------------------------------------------------------------
# copy-engine peer copy (0 SM; 343 GB/s in Gate B3)
# ---------------------------------------------------------------------------
def memcpy_dtod_async(
    dst_ptr: int, src_ptr: int, nbytes: int, stream_handle: int
) -> None:
    """Device-to-device async copy via the generic DtoD path. WARNING: on peer
    memory the driver may implement this as an SM copy kernel (stole ~14% of a
    compute-bound matmul in the PoC). Prefer memcpy_peer_async for the real
    copy-engine path."""
    _ck(_drv.cuMemcpyDtoDAsync(dst_ptr, src_ptr, nbytes, stream_handle))


def primary_ctx(dev: int) -> int:
    """Retain and return the device's primary CUcontext (the one torch uses)."""
    return _ck(_drv.cuDevicePrimaryCtxRetain(dev))


def enable_peer_access(from_ctx, to_ctx) -> None:
    """Enable `from_ctx` -> access memory in `to_ctx` (needed for cuMemcpyPeerAsync
    to use the NVLink copy engine instead of staging through host).
    Idempotent: PEER_ACCESS_ALREADY_ENABLED is fine."""
    _ck(_drv.cuCtxPushCurrent(from_ctx))
    try:
        ret = _drv.cuCtxEnablePeerAccess(to_ctx, 0)
        err = ret[0] if isinstance(ret, tuple) else ret
        # 704 = CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
        if int(err) not in (0, 704):
            raise RuntimeError(f"cuCtxEnablePeerAccess: {err}")
    finally:
        _ck(_drv.cuCtxPopCurrent())


def memcpy_peer_async(
    dst_ptr: int, dst_ctx, src_ptr: int, src_ctx, nbytes: int, stream_handle: int
) -> None:
    """cuMemcpyPeerAsync — the explicit peer copy. Runs on the copy engine
    (0 SM, Gate B3). dst_ctx/src_ctx are CUcontext handles from primary_ctx();
    dst_ptr must be valid in dst_ctx (open the IPC handle under that context)."""
    _ck(
        _drv.cuMemcpyPeerAsync(
            dst_ptr, dst_ctx, src_ptr, src_ctx, nbytes, stream_handle
        )
    )


# ---------------------------------------------------------------------------
# IPC export / import
# ---------------------------------------------------------------------------
def ipc_get_handle_bytes(dev_ptr: int) -> bytes:
    """Export the base allocation containing dev_ptr as raw IPC handle bytes."""
    h = _ck(_drv.cuIpcGetMemHandle(dev_ptr))
    return bytes(h.reserved)


def ipc_open_handle(handle_bytes: bytes) -> int:
    """Open a peer's IPC handle -> device pointer valid in the CURRENT context
    (peer access lazily enabled). Returns the base pointer of the peer alloc."""
    h = _drv.CUipcMemHandle()
    h.reserved = handle_bytes
    ptr = _ck(_drv.cuIpcOpenMemHandle(h, _IPC_LAZY))
    return int(ptr)


def ipc_open_handle_in_ctx(handle_bytes: bytes, ctx) -> int:
    """Open a peer IPC handle with `ctx` (the OWNER device's primary context)
    made current, so the returned pointer is valid in that context -> usable as
    the dst of cuMemcpyPeerAsync (true copy-engine peer copy)."""
    _ck(_drv.cuCtxPushCurrent(ctx))
    try:
        h = _drv.CUipcMemHandle()
        h.reserved = handle_bytes
        ptr = _ck(_drv.cuIpcOpenMemHandle(h, _IPC_LAZY))
    finally:
        _ck(_drv.cuCtxPopCurrent())
    return int(ptr)


def ipc_close_handle(dev_ptr: int) -> None:
    _ck(_drv.cuIpcCloseMemHandle(dev_ptr))


def block_offset(dev_ptr: int) -> int:
    """Byte offset of dev_ptr within its containing CUDA allocation.
    (torch may sub-allocate a tensor at a nonzero offset inside a pooled block;
    cuIpcOpenMemHandle returns the block base, so the peer must add this offset.)"""
    base, _size = _ck(_drv.cuMemGetAddressRange(dev_ptr))
    return int(dev_ptr) - int(base)


def ipc_export(tensor) -> tuple[bytes, int]:
    """Export a torch CUDA tensor for one-sided writes: returns (handle_bytes,
    offset) where offset is the tensor's byte offset within its IPC block.
    Peer does: slot_ptr = ipc_import(handle_bytes) + offset + per_slot_offset."""
    p = tensor.data_ptr()
    return ipc_get_handle_bytes(p), block_offset(p)
