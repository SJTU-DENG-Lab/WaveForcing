"""HTTP adapter for :class:`WaveServingRuntime`."""

from __future__ import annotations

from typing import Any

from wave_rt.config import WaveConfig
from wave_rt.serving.protocol import ProtocolError, QueueFullError
from wave_rt.serving.runtime import WaveServingRuntime


def require_http_dependencies() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "WaveRT serving needs the optional serving dependencies; "
            "install the project with `uv sync --extra serve`"
        ) from exc


def run_http_server(runtime: WaveServingRuntime, cfg: WaveConfig) -> None:
    """Expose blocking and asynchronous generation over HTTP."""
    require_http_dependencies()
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="WaveRT", version="0.1.0")
    server_ref: dict[str, Any] = {}

    @app.post("/generate")
    def generate(body: dict[str, Any]):
        payload = dict(body)
        wait = bool(payload.pop("wait", True))
        timeout_value = payload.pop("wait_timeout_s", None)
        try:
            timeout = None if timeout_value is None else float(timeout_value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid wait_timeout_s: {exc}"
            ) from exc
        if timeout is not None and timeout < 0:
            raise HTTPException(
                status_code=400, detail="wait_timeout_s must be non-negative"
            )
        try:
            result = runtime.submit(payload, wait=wait, timeout=timeout)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ProtocolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status_code = 200 if result["status"] in {"succeeded", "failed"} else 202
        return JSONResponse(status_code=status_code, content=result)

    @app.get("/requests/{request_id}")
    def request_status(request_id: str):
        result = runtime.request_status(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail="request not found")
        return result

    @app.get("/status")
    def status():
        return runtime.status()

    @app.get("/health")
    def health():
        result = runtime.status()
        healthy = result["status"] in {"starting", "ready", "busy"}
        return JSONResponse(status_code=200 if healthy else 503, content=result)

    @app.post("/shutdown")
    def shutdown():
        server_ref["server"].should_exit = True
        return {"ok": True, "draining": True}

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=cfg.serve_host,
            port=cfg.serve_port,
            log_level="warning",
        )
    )
    server_ref["server"] = server
    print(
        f"[wave_rt/serve] http://{cfg.serve_host}:{cfg.serve_port} "
        "(POST /generate | GET /requests/{id} | GET /status | POST /shutdown)",
        flush=True,
    )
    server.run()
