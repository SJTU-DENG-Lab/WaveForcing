"""Single-viewer HTTP demo; generation follows delivery (live stream) or /played."""

from collections import deque
from contextlib import asynccontextmanager
import json
from pathlib import Path
import queue
import re
import secrets
import struct
import threading
import time

from .state import Pages, Session
from .video import MIME


class Trace:
    """Recent server/browser timeline; an optional JSONL copy is written off the request path."""

    def __init__(self, path=None, keep=4096):
        self.rows = {"server": deque(maxlen=keep), "client": deque(maxlen=keep)}
        self.run = secrets.token_hex(4)
        self.error = None
        self.file = open(path, "a") if path else None
        self.pending = queue.SimpleQueue()
        if self.file is not None:
            threading.Thread(target=self._write, daemon=True, name="wavert-trace").start()

    def add(self, source, event, **data):
        row = {"wall": time.time(), "source": source, "event": event, **data}
        self.rows[source].append(row)
        if self.file is not None:
            self.pending.put(row)

    def _write(self):
        while True:
            row = self.pending.get()
            try:
                self.file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                if self.pending.empty():
                    self.file.flush()
            except (OSError, ValueError) as exc:
                self.error = type(exc).__name__


def log(message):
    print(f"[interactive {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Controller:
    def __init__(self, runtime, prompt, limits, output_budget=256 * 2**20, *, autostart=True,
                 trace=None):
        self.runtime = runtime
        self.limits = limits
        self.trace = trace or Trace()
        self.session = Session(prompt, limits)
        self.frames = Pages(output_budget)
        self.lock = threading.RLock()
        self.ready = threading.Condition(self.lock)
        self.error = None
        self.closed = False
        self.viewer = None
        self.last_seen = 0.0
        self.session_id = secrets.token_hex(16)
        self.generation = 0
        self.has_session = False
        self.initial_prompt = ""
        self.resetting = False
        self.last_start_request = None
        self.seed = None
        self.epoch = None
        self.clock = {}  # chunk -> wall times while it moves through the pipeline
        self.prompted = None
        self.prompt_history = []
        runtime.start(self._event)
        if autostart:
            with self.lock:
                self._begin(prompt)

    def view(self):
        return {**self.session.view(), "session_id": self.session_id,
                "generation": self.generation, "has_session": self.has_session,
                "resetting": self.resetting, "initial_prompt": self.initial_prompt,
                "current_prompt": self.session.prompt if self.has_session else "",
                "prompt_history": list(self.prompt_history),
                "seed": self.seed, "error": self.error, "output_bytes": self.frames.used}

    def check_session(self, session_id):
        if session_id != self.session_id:
            raise ValueError("This request belongs to an older generation")

    def _begin(self, prompt):
        # Caller holds the lock. Workers finish what was committed, then
        # restart from chunk zero; older events are told apart by epoch.
        self.initial_prompt = prompt
        self.seed = secrets.randbits(31)
        self.session = Session(prompt, self.limits)
        self.frames = Pages(self.frames.budget)
        self.clock = {}
        self.prompted = None
        self.has_session = True
        self.resetting = True
        self.prompt_history = [{"text": prompt, "wall": time.time(), "kind": "start", "start": 0}]
        self.epoch = self.runtime.reset(self.seed, prompt)
        self.trace.add("server", "start", generation=self.generation, prompt_chars=len(prompt))
        self._dispatch()

    def start(self, prompt, request_id, previous_session):
        # Caller holds the lock. Retried start requests must not launch twice.
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请输入初始提示词")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("Invalid start request")
        if request_id == self.last_start_request:
            return self.view()
        self.check_session(previous_session)
        if self.error or self.closed:
            raise ValueError("生成服务出错，需要重新启动服务")
        if self.resetting:
            raise ValueError("正在准备新的生成，请稍候")
        self.last_start_request = request_id
        self.session_id = secrets.token_hex(16)
        self.generation += 1
        self._begin(prompt.strip())
        self.ready.notify_all()
        return self.view()

    def prompt(self, prompt):
        # Caller holds the lock.
        if not self.has_session or self.resetting:
            raise ValueError("请先完成初始生成准备")
        self.session.interact(prompt)
        self.prompted = time.time()
        self._dispatch()

    def played(self, branch, chunk):
        # Caller holds the lock.
        if branch != 0:
            raise ValueError("Stale playback acknowledgement")
        before = self.session.played
        self.session.acknowledge(chunk)
        if self.session.played == before:
            return
        self.frames.prune(self.session.played + 1, ())
        ahead = self.session.end - self.session.played - 1
        self.trace.add("server", "played", chunk=self.session.played, ahead=ahead)
        if ahead <= 1:
            # The browser acknowledges a chunk once the next one is on screen,
            # so nothing is encoded beyond what it is showing right now.
            log(f"low buffer: chunk {self.session.played} played with {ahead} encoded ahead")
        self._dispatch()
        self.ready.notify_all()

    def _dispatch(self):
        # Caller holds the lock, so every rank sees one command order.
        if not self.has_session or self.error or self.closed:
            return
        before = self.session.issued
        for command in self.session.schedule():
            if "end" in command:
                self.runtime.advance(command["end"])
            else:
                self.runtime.switch(command["prompt"], command["start"])
                self.prompt_history.append({
                    "text": command["prompt"], "wall": time.time(),
                    "kind": "switch", "start": command["start"],
                })
                self.trace.add("server", "switch", start=command["start"],
                               prompt_chars=len(command["prompt"]))
        now = time.time()
        for chunk in range(before, self.session.issued):
            self.clock[chunk] = {"issued": now}

    def _event(self, event):
        # Runs on the runtime's event thread.
        with self.lock:
            try:
                kind = event["kind"]
                if kind == "error":
                    raise RuntimeError(event["error"])
                if event.get("epoch") != self.epoch:
                    return  # Left over from a session that Start already replaced.
                if kind == "reset":
                    self.resetting = False
                    self.ready.notify_all()
                elif kind == "tick":
                    self.trace.add("server", "tick", tick=event["tick"], ms=event["ms"],
                                   waited_ms=event["waited_ms"])
                elif kind == "latent":
                    self.clock.setdefault(event["chunk"], {})["latent"] = event["at"]
                elif kind == "chunk":
                    self._chunk(event)
                return
            except Exception as exc:
                self.error = str(exc)
                self.trace.add("server", "error", error=self.error)
                log(f"error: {self.error}")
                self.ready.notify_all()
        self.runtime.close()

    def _chunk(self, event):
        chunk, now = event["chunk"], time.time()
        self.frames.put(0, chunk, event["data"], event["size"])
        self.session.produced(chunk)
        clock = self.clock.pop(chunk, {})
        row = {"chunk": chunk, "bytes": event["size"],
               "ahead": self.session.end - self.session.played - 1}
        for name, begin, finish in [
            ("diffusion_ms", clock.get("issued"), clock.get("latent")),
            ("vae_ms", clock.get("latent"), event["decoded_at"]),
            ("encode_ms", event["decoded_at"], event["encoded_at"]),
            ("encode_queue_ms", event["decoded_at"], event.get("encode_started_at", event["decoded_at"])),
            ("encode_service_ms", event.get("encode_started_at", event["decoded_at"]), event["encoded_at"]),
            ("deliver_ms", event["encoded_at"], now),
            ("total_ms", clock.get("issued"), now),
        ]:
            if begin is not None:
                row[name] = (finish - begin) * 1000
        if self.prompted is not None and chunk == self.session.boundary:
            # First chunk of the requested prompt; it plays once the chunks
            # committed before the request have been shown.
            row["since_prompt_ms"] = (now - self.prompted) * 1000
            log(f"prompt change ready at chunk {chunk} after {row['since_prompt_ms']:.0f} ms; "
                f"{chunk - self.session.played - 1} committed chunks play first")
            self.prompted = None
        self.trace.add("server", "chunk", **row)
        self.ready.notify_all()

    def attach(self, token=None):
        with self.lock:
            if token and token == self.viewer:
                self.last_seen = time.monotonic()
                return {"client": self.viewer}
            if self.viewer and time.monotonic() - self.last_seen < 30:
                raise ValueError("This demo supports one playback client; close the other tab and wait 30 seconds")
            self.viewer = secrets.token_urlsafe(24)
            self.last_seen = time.monotonic()
            return {"client": self.viewer}

    def validate(self, token):
        if not token or token != self.viewer:
            raise ValueError("Playback client is not attached")
        self.last_seen = time.monotonic()

    def take_live(self, chunk, session_id, token, timeout=30.0):
        """Block until chunk `chunk` is encoded, then hand it to the live stream.

        Advancing `pace` here (not `/played`) keeps generation moving when the
        playback ACK path stalls on the notebook proxy / jump host. A hard cap
        a few chunks past the normal buffer stops a fast reader from extending
        the timeline without bound while `/played` is silent.
        """
        deadline = time.monotonic() + timeout
        with self.lock:
            self.validate(token)
            self.check_session(session_id)
            while True:
                if self.closed or self.error:
                    return None
                if session_id != self.session_id or self.resetting:
                    return None
                limit = self.session.played + self.limits.max_ahead
                if chunk > limit:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self.ready.wait(timeout=min(0.5, remaining))
                    continue
                try:
                    data = self.frames.get(0, chunk)
                except KeyError:
                    data = None
                if data is not None:
                    self.session.advance_pace(chunk)
                    ahead = self.session.end - self.session.played - 1
                    self.trace.add("server", "live", chunk=chunk, bytes=len(data),
                                   pace=self.session.pace, ahead=ahead)
                    self._dispatch()
                    return data
                # Ask the pipeline for this chunk even if /played is behind.
                if chunk >= self.session.issued:
                    if chunk > 0:
                        self.session.advance_pace(chunk - 1)
                    else:
                        self.session.advance_pace(-1)
                    self._dispatch()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.ready.wait(timeout=min(0.5, remaining))

    def close(self):
        with self.lock:
            self.closed = True
            self.ready.notify_all()
        self.runtime.close()


class ProxyPrefix:
    """The notebook absproxy forwards its prefix; direct requests stay valid."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            match = re.match(r"^/absproxy/[0-9]+(?=/|$)", path)
            if match:
                scope = dict(scope)
                scope["path"] = path[match.end():] or "/"
                scope["raw_path"] = scope["path"].encode()
        await self.app(scope, receive, send)


def make_app(controller, fps=16, monitor=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    @asynccontextmanager
    async def lifespan(_app):
        yield
        controller.close()
        if monitor is not None:
            monitor.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(ProxyPrefix)

    @app.exception_handler(ValueError)
    async def bad_request(_request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(Path(__file__).with_name("player.html").read_text(),
                            headers={"Cache-Control": "no-store"})

    @app.post("/attach")
    def attach(request: Request):
        return controller.attach(request.headers.get("X-Client-Id"))

    @app.get("/health")
    def health():
        with controller.lock:
            return {"service": "wavert-interactive", "status": "error" if controller.error else "ok",
                    "transport": "h264-fmp4-live", "controls": True, "has_session": controller.has_session,
                    "generated_chunks": controller.session.end, "error": controller.error}

    @app.post("/detach")
    def detach(request: Request):
        with controller.lock:
            controller.validate(request.headers.get("X-Client-Id"))
            controller.viewer = None
            return {"status": "detached"}

    def check(request, *, session=True):
        controller.validate(request.headers.get("X-Client-Id"))
        if session:
            controller.check_session(request.headers.get("X-Session-Id"))

    def playback():
        return {"fps": fps, "start_buffer": controller.limits.buffer, "mime": MIME}

    @app.get("/state")
    def state(request: Request):
        with controller.lock:
            check(request, session=False)
            return {**controller.view(), **playback()}

    @app.post("/start")
    def start(body: dict, request: Request):
        with controller.lock:
            check(request, session=False)
            return {**controller.start(body.get("prompt"), body.get("request_id"),
                                       request.headers.get("X-Session-Id")), **playback()}

    @app.get("/metrics")
    def metrics(request: Request):
        with controller.lock:
            check(request, session=False)
        return monitor.snapshot() if monitor is not None else {
            "status": "unavailable", "gpus": [], "error": "No GPU monitor configured",
            "sampled_at": None, "age_seconds": None}

    @app.post("/prompt")
    def prompt(body: dict, request: Request):
        with controller.lock:
            check(request)
            controller.prompt(body.get("prompt"))
            return {"status": "pending"}

    @app.post("/played")
    def played(body: dict, request: Request):
        with controller.lock:
            check(request)
            controller.played(body.get("branch"), body.get("chunk"))
            return controller.view()

    @app.get("/live")
    def live(request: Request, after: int = -1):
        # One long response: for each chunk, uint32 chunk_id + uint32 length + bytes.
        # Generation follows how far this stream has read, so a stuck /played cannot
        # drain the browser buffer by parking the pipeline. Sync generator so the
        # notebook proxy and TestClient both see bytes without an async thread hop.
        with controller.lock:
            check(request)
            if after < -1:
                raise ValueError("Invalid live stream cursor")
            session_id = request.headers.get("X-Session-Id")
            token = request.headers.get("X-Client-Id")
            controller.check_session(session_id)

        def frames():
            chunk = after + 1
            idle = 0.0
            while True:
                data = controller.take_live(chunk, session_id, token, timeout=0.5)
                if data is not None:
                    yield struct.pack(">II", chunk, len(data)) + data
                    chunk += 1
                    idle = 0.0
                    continue
                with controller.lock:
                    done = (controller.closed or controller.error
                            or session_id != controller.session_id or controller.resetting)
                if done:
                    return
                idle += 0.5
                if idle >= 30.0:
                    return

        return StreamingResponse(
            frames(),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/chunk/{branch}/{chunk}")
    def chunk(branch: int, chunk: int, request: Request):
        with controller.lock:
            check(request)
            try:
                data = controller.frames.get(branch, chunk)
            except KeyError:
                raise HTTPException(404, "Chunk not available") from None
            return Response(data, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.post("/trace")
    def upload_trace(body: dict, request: Request):
        # Browser diagnostics; batches are bounded by the player.
        with controller.lock:
            check(request, session=False)
        events = body.get("events")
        if not isinstance(events, list) or len(events) > 256:
            raise ValueError("Invalid trace batch")
        for row in events:
            if not isinstance(row, dict):
                raise ValueError("Invalid trace batch")
            controller.trace.add("client", str(row.get("event")), page=body.get("page"), row=row)
            data = row.get("data")
            if row.get("event") == "waiting" and isinstance(data, dict) and data.get("counted"):
                log("viewer reported a buffer underrun")
        return {"accepted": len(events), "dropped": 0, "writer_error": controller.trace.error,
                "relay_run": controller.trace.run, "server_wall_ms": time.time() * 1000}

    @app.get("/trace")
    def read_trace(source: str = "server", limit: int = 200):
        rows = list(controller.trace.rows.get(source, ()))
        return {"run": controller.trace.run, "writer_error": controller.trace.error,
                "rows": rows[max(0, len(rows) - limit):] if limit > 0 else []}

    return app
