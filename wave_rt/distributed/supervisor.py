"""Bounded waits for child-process completion and request results."""
import queue
import time


def check_workers(processes):
    failed = [(p.pid, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
    if failed:
        raise RuntimeError(f'WaveRT workers failed: {failed}')


def receive_results(channel, processes, kinds, timeout=3600):
    deadline = time.monotonic() + timeout
    results = {}
    while not set(kinds).issubset(results):
        check_workers(processes)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Missing WaveRT results: {set(kinds) - results.keys()}')
        try:
            item = channel.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if all(p.exitcode is not None for p in processes):
                raise RuntimeError(f'Workers exited without results: {set(kinds) - results.keys()}')
            continue
        if item[0] in results:
            raise RuntimeError(f'Duplicate WaveRT result: {item[0]}')
        results[item[0]] = item[1:]
    return results


def stop_workers(processes):
    # Only processes created by this launcher; no global name-based termination.
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def join_workers(processes, timeout=3600):
    deadline = time.monotonic() + timeout
    try:
        while any(p.is_alive() for p in processes):
            check_workers(processes)
            if time.monotonic() >= deadline:
                raise TimeoutError('WaveRT worker completion timed out')
            for process in processes:
                process.join(timeout=0.05)
        check_workers(processes)
    except BaseException:
        stop_workers(processes)
        raise
