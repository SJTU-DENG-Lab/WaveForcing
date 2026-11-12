"""Device-level memory sampling, independent of GPU inference/control requests."""
import csv
import subprocess
import threading
import time


def parse_gpu_rows(text, roles):
    rows = []
    for rank, fields in enumerate(csv.reader(text.strip().splitlines())):
        if len(fields) != 5:
            raise ValueError("Unexpected nvidia-smi memory response")
        index, name, used, total, utilization = (x.strip() for x in fields)
        def number(value):
            try:
                return float(value)
            except ValueError:
                return None
        rows.append({"index": int(index), "name": name, "role": roles[rank] if rank < len(roles) else "GPU",
                     "used_mib": number(used), "total_mib": number(total), "utilization_percent": number(utilization)})
    if not rows:
        raise ValueError("No GPU memory readings available")
    return rows


class GpuMonitor:
    def __init__(self, devices, roles, interval=1.0, query=None):
        self.devices, self.roles, self.interval = devices, roles, interval
        self.query = query or self._query
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.data = {"sampled_at": None, "gpus": [], "error": None}
        self.thread = threading.Thread(target=self._run, daemon=True, name="gpu-memory")
        self.thread.start()

    def _query(self):
        return subprocess.check_output([
            "nvidia-smi", "--id=" + self.devices,
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=2)

    def _run(self):
        while not self.stop.is_set():
            try:
                rows = parse_gpu_rows(self.query(), self.roles)
                with self.lock:
                    self.data = {"sampled_at": time.time(), "gpus": rows, "error": None}
            except Exception as exc:
                with self.lock:
                    self.data = {**self.data, "error": type(exc).__name__}
            self.stop.wait(self.interval)

    def snapshot(self):
        with self.lock:
            data = dict(self.data)
        age = None if data["sampled_at"] is None else max(0, time.time() - data["sampled_at"])
        status = "unavailable" if not data["gpus"] else "stale" if data["error"] or age > 3 else "ok"
        return {**data, "status": status, "age_seconds": age, "interval_seconds": self.interval,
                "scope": "device", "used_mib": sum(r["used_mib"] for r in data["gpus"]) if data["gpus"] and all(r["used_mib"] is not None for r in data["gpus"]) else None,
                "total_mib": sum(r["total_mib"] for r in data["gpus"]) if data["gpus"] and all(r["total_mib"] is not None for r in data["gpus"]) else None}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
