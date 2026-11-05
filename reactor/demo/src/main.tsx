import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import "./style.css";

type World = {
  prompt: string;
  seed: number | null;
  active: boolean;
  paused: boolean;
  world_id: number;
  completed_chunks: number;
  output_frames: number;
  last_chunk_seconds: number | null;
};

type WorkspaceProps = {
  endpoint: string;
  activeEndpoint: string;
  onEndpointChange: (value: string) => void;
  onEndpointApply: () => void;
};

function Workspace({
  endpoint,
  activeEndpoint,
  onEndpointChange,
  onEndpointApply,
}: WorkspaceProps) {
  const { status, connect, disconnect, sendCommand, lastError } = useReactor(
    (s) => ({
      status: s.status,
      connect: s.connect,
      disconnect: s.disconnect,
      sendCommand: s.sendCommand,
      lastError: s.lastError,
    }),
  );
  const [world, setWorld] = useState<World | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [handledError, setHandledError] = useState(lastError);
  const dirty = useRef(false);
  const ready = status === "ready";
  useReactorMessage((message) => {
    if (message.type !== "state_update") return;
    const accepted = message.data as World;
    setWorld(accepted);
    if (!dirty.current) setPrompt(accepted.prompt);
  });
  useEffect(() => {
    if (lastError && lastError !== handledError) {
      setError(lastError.message);
      setHandledError(lastError);
    }
  }, [lastError, handledError]);
  useEffect(() => {
    if (status === "disconnected") setWorld(null);
  }, [status]);
  useEffect(() => {
    if (status !== "ready") return;
    let current = true;
    // Read after SDK readiness as well: a joining snapshot can precede it.
    void sendCommand("get_state", {}).then((reply) => {
      if (!current || !reply) return;
      const accepted = reply.data as World;
      setWorld(accepted);
      if (!dirty.current) setPrompt(accepted.prompt);
    });
    return () => {
      current = false;
    };
  }, [status, sendCommand]);
  async function perform(action: () => Promise<unknown>) {
    setError("");
    setHandledError(lastError);
    setBusy(true);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function command(name: string, args: Record<string, unknown> = {}) {
    const reply = await sendCommand(name, args);
    if (reply === undefined) return;
    const accepted = reply.data as World;
    setWorld(accepted);
    if (name === "start" || name === "set_prompt") {
      dirty.current = false;
      setPrompt(accepted.prompt);
    }
    return accepted;
  }
  return (
    <main>
      <header className="topbar">
        <a
          className="brand"
          href="https://reactor.inc"
          target="_blank"
          rel="noreferrer"
        >
          <img src="/reactor-symbol.svg" width="34" height="24" alt="Reactor" />
          <span className="brand-title">
            <strong>WaveForcing</strong>
            <span>on Reactor</span>
          </span>
        </a>
        <form
          className="endpoint"
          onSubmit={(e) => {
            e.preventDefault();
            if (endpoint === activeEndpoint) return;
            void perform(async () => {
              if (status !== "disconnected") await disconnect();
              onEndpointApply();
            });
          }}
        >
          <label htmlFor="runtime-url">Runtime URL</label>
          <input
            id="runtime-url"
            type="url"
            required
            value={endpoint}
            onChange={(e) => onEndpointChange(e.target.value)}
          />
          <button
            type="submit"
            disabled={busy || status === "connecting" || status === "waiting"}
          >
            Use endpoint
          </button>
        </form>
        <div className="connection-controls">
          <span className="connection" data-status={status} role="status">
            <span className="connection-dot" aria-hidden="true" />
            {
              {
                disconnected: "Offline",
                connecting: "Connecting",
                waiting: "Waiting",
                ready: "Connected",
              }[status]
            }
          </span>
          <button
            disabled={busy || status === "connecting" || status === "waiting"}
            onClick={() =>
              void perform(async () => {
                if (ready) {
                  await disconnect();
                  setWorld(null);
                } else await connect();
              })
            }
          >
            {ready ? "Disconnect" : "Connect"}
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside>
          <p className="eyebrow">CONTINUOUS VIDEO GENERATION</p>
          <h1>
            Let the scene
            <br />
            keep unfolding.
          </h1>
          <p>
            Describe a scene. Shape what happens next.
            <br />
            One continuous video, until you stop.
          </p>
          <label>
            Prompt
            <textarea
              aria-label="Prompt"
              rows={8}
              maxLength={4096}
              value={prompt}
              placeholder="A cinematic shot of a corgi running along a sunny beach, waves in the background…"
              onChange={(e) => {
                dirty.current = true;
                setPrompt(e.target.value);
              }}
            />
          </label>
          <button
            className="primary"
            disabled={!ready || busy || !prompt.trim()}
            onClick={() =>
              void perform(() => command("start", { prompt, seed }))
            }
          >
            {world?.active ? "Start new video" : "Start video"}
            <span>↗</span>
          </button>
          <button
            disabled={!ready || busy || !world?.active || !prompt.trim()}
            onClick={() =>
              void perform(() => command("set_prompt", { prompt }))
            }
          >
            Update prompt
          </button>
          <p className="hint">
            Update prompt guides upcoming chunks and keeps the current scene.
            Frames already generated keep their earlier prompt.
          </p>
          <details>
            <summary>Generation settings</summary>
            <label>
              Seed
              <input
                aria-label="Seed"
                type="number"
                min={0}
                max={2147483647}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <p className="hint">
              The seed is applied when you start a new video.
            </p>
          </details>
          <div className="actions">
            <button
              disabled={!ready || busy || !world?.active}
              onClick={() =>
                void perform(() =>
                  command("set_paused", { paused: !world?.paused }),
                )
              }
            >
              {world?.paused ? "Resume" : "Pause"}
            </button>
            <button
              disabled={!ready || busy || !world?.active}
              onClick={() => void perform(() => command("reset", {}))}
            >
              Reset
            </button>
            <button
              disabled={!ready || busy || !world?.active}
              onClick={() => void perform(() => command("stop"))}
            >
              Stop
            </button>
          </div>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <div className="model-note">
            <span>WAVEFORCING · 1.3B · 4 STEPS</span>
            <p>
              Text-to-video with live prompt changes. No reference image is
              required.
            </p>
          </div>
        </aside>
        <section className="stage">
          <div className="stage-heading">
            <div>
              <p className="eyebrow">LIVE OUTPUT</p>
              <h2>Your scene, in motion.</h2>
            </div>
            <span className="badge">∞ Continuous</span>
          </div>
          <div className="player">
            <ReactorView
              key={world?.world_id ?? 0}
              track="main_video"
              muted
              videoObjectFit="contain"
              className="video"
            />
            {(!ready || !world?.active || !world.completed_chunks) && (
              <div className="placeholder">
                <span>▷</span>
                <h3>
                  {!ready
                    ? "Your scene starts here"
                    : world?.active
                      ? "Creating the first frames…"
                      : "Ready when you are"}
                </h3>
                <p>
                  {!ready
                    ? "Connect, enter a prompt, then press Start video."
                    : world?.active
                      ? "The video appears when the first chunk is ready."
                      : "Enter a prompt and press Start video."}
                </p>
              </div>
            )}
          </div>
          <div className="playback">
            <span>
              {world?.active
                ? world.paused
                  ? "Generation paused · buffered video can still play"
                  : "● Generating continuously"
                : ready
                  ? "Connected · waiting for your scene"
                  : "Waiting to connect"}
            </span>
            <span>832 × 480 · Silent video</span>
          </div>
          <p className="progress" role="status">
            {world?.active
              ? `Video ${world.world_id} · ${world.completed_chunks} chunks · ${world.output_frames} frames${world.last_chunk_seconds === null ? "" : ` · ${world.last_chunk_seconds.toFixed(2)} s / chunk`}`
              : "No video generated yet."}
          </p>
          <div className="stage-note">
            <strong>Change the prompt. Keep the story.</strong>
            <p>
              Start new video and Reset begin again. Update prompt continues the
              current video, including while paused. Prompt response may be
              gradual.
            </p>
          </div>
          <footer>
            <a
              href="https://github.com/SJTU-DENG-Lab/WaveForcing"
              target="_blank"
              rel="noreferrer"
            >
              WaveForcing ↗
            </a>
            <span>Built with Reactor</span>
          </footer>
        </section>
      </div>
    </main>
  );
}

function App() {
  const [endpoint, setEndpoint] = useState(
    () =>
      new URLSearchParams(location.search).get("endpoint") ||
      "http://localhost:8080",
  );
  const [active, setActive] = useState(endpoint);
  return (
    <ReactorProvider
      key={active}
      modelName="waveforcing"
      modelTracks={[
        { name: "main_video", kind: "video", direction: "recvonly" },
      ]}
      local
      apiUrl={active}
      connectOptions={{ autoConnect: false }}
    >
      <Workspace
        endpoint={endpoint}
        activeEndpoint={active}
        onEndpointChange={setEndpoint}
        onEndpointApply={() => setActive(endpoint)}
      />
    </ReactorProvider>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
