"""Exercise the real application/model contracts without CUDA or checkpoints."""

import asyncio
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract

from waveforcing import WaveForcing
from waveforcing_model import (
    Anchor,
    NoAnchor,
    WaveForcingInput,
    WaveForcingModel,
    WaveForcingResult,
)
from waveforcing_types import WaveForcingState


def test_live_text_refresh_invalidates_every_text_cache_without_touching_history():
    from waveforcing_backend import Run, WaveForcingBackend

    backend = WaveForcingBackend.__new__(WaveForcingBackend)
    visual_cache, decoder, rng, noise, frontier = (object() for _ in range(5))
    encoder = Mock(side_effect=lambda **kw: {"text": kw["text_prompts"][0]})
    backend.pipeline = SimpleNamespace(
        text_encoder=encoder,
        crossattn_cache=[{"is_init": True} for _ in range(30)],
        kv_cache_clean=visual_cache,
        vae=decoder,
    )
    backend.run = Run(
        prompt="A garden",
        conditioning={"text": "A garden"},
        initial_rng=rng,
        page=noise,
        page_offset=24,
        pending=[frontier],
        tick=11,
    )
    backend._update_prompt("A garden")
    encoder.assert_not_called()
    for prompt in ("A cat arrives", "The cat walks away"):
        for cache in backend.pipeline.crossattn_cache:
            cache["is_init"] = True
        backend._update_prompt(prompt)
        assert backend.run.conditioning == {"text": prompt}
        assert all(not c["is_init"] for c in backend.pipeline.crossattn_cache)
        assert backend.pipeline.kv_cache_clean is visual_cache
        assert backend.pipeline.vae is decoder and backend.run.initial_rng is rng
        assert backend.run.page is noise and backend.run.pending == [frontier]
        assert backend.run.tick == 11 and backend.run.page_offset == 24
    assert encoder.call_count == 2


def test_prepared_weights_are_reusable_without_network_or_writes(tmp_path, monkeypatch):
    import json

    import huggingface_hub

    from waveforcing_assets import BASE_FILES, PROVENANCE, prepare_weights

    root = tmp_path / "wan_models/Wan2.1-T2V-1.3B"
    for name in (
        *BASE_FILES,
        "diffusion_pytorch_model.safetensors",
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"prepared asset")
    (tmp_path / "waveforcing-assets.json").write_text(json.dumps(PROVENANCE))
    forbidden = Mock(side_effect=AssertionError("Prepared assets must be read-only"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    prepare_weights(tmp_path)
    forbidden.assert_not_called()


def app():
    model = WaveForcing()
    model.state = WaveForcingState()
    model.engine = Mock()
    model.output = Mock()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    return model, messages


def result(world=1, index=1):
    return WaveForcingResult(
        world_id=world, chunk_index=index, frames=np.zeros((2, 480, 832, 3), np.uint8)
    )


def test_refused_step_never_reaches_compute():
    model, _ = app()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.start("A garden", 42))
    asyncio.run(model.set_paused(True))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    model.engine.generate.assert_not_called()


def test_explicit_snapshot_is_read_only_and_does_not_broadcast():
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    asyncio.run(model.set_paused(True))
    messages.clear()
    before = model._status()
    assert model.get_state() == before
    assert model._status() == before and not messages
    model.engine.reset.assert_not_called()


def test_runtime_loop_routes_compute_failure_without_acknowledging_output():
    import threading

    from reactor_runtime.core.model import ClientConnected, SessionStarted
    from reactor_runtime.core.values import ConnId

    model, _ = app()
    failure = RuntimeError("failed native compute")
    model.engine.generate.side_effect = failure
    finished = threading.Event()
    failures, outcomes, media = [], [], []
    process_output = model.process_output

    async def record(outcome):
        outcomes.append(outcome)
        return await process_output(outcome)

    def failed(error):
        failures.append(error)
        finished.set()

    model.process_output = record
    model.bind_output(
        broadcast=lambda *args: None,
        addressed=lambda *args: None,
        media=lambda *args: media.append(args),
    )
    model.bind_failure(failed)
    model.start_thread()
    try:
        model.post_reactor_event(SessionStarted("waveforcing-cpu-test"))
        model.post_reactor_event(ClientConnected(ConnId(1), total=1))
        command = ModelContract.of(WaveForcing).validate(
            "start", {"prompt": "A garden", "seed": 42}
        )
        model.submit_command(command, ConnId(1), None)
        assert finished.wait(timeout=5), "Runtime did not report the failed step"
        assert failures == [failure] and not media
        assert len(outcomes) == 1 and outcomes[0].error is failure
        assert model.state._completed == 0 and model.state._applied_world_id is None
        model.engine.generate.assert_called_once()
    finally:
        model.stop()


def test_forward_reads_only_input():
    model, _ = app()
    value = WaveForcingInput(world_id=7, prompt="old")
    model.state = None
    model.generate(value)
    model.engine.generate.assert_called_once_with(value)


def test_anchor_repeats_until_acknowledged_then_is_omitted():
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    first = asyncio.run(model.process_input())
    assert first.anchor == Anchor(seed=42)
    assert asyncio.run(model.process_input()).anchor is first.anchor
    out = asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=0.3)))
    assert out.main_video.shape == (2, 480, 832, 3)
    assert asyncio.run(model.process_input()).anchor is None
    assert messages[-1].completed_chunks == 1 and messages[-1].last_chunk_seconds == 0.3


@pytest.mark.parametrize("transition", ["start", "reset", "stop"])
def test_stale_output_cannot_change_replaced_or_stopped_world(transition):
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    if transition == "start":
        asyncio.run(model.start("A beach", 7))
    elif transition == "reset":
        asyncio.run(model.reset(None))
    else:
        asyncio.run(model.stop_video())
    before = model._status()
    messages.clear()
    assert (
        asyncio.run(model.process_output(StepOutcome(result=result(), elapsed=1)))
        is None
    )
    assert model._status() == before and not messages


def test_two_prompt_updates_preserve_history_and_pause():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    asyncio.run(model.process_output(StepOutcome(result=result(index=3), elapsed=0.3)))
    model.output.flush.reset_mock()
    for text in ("A cat arrives", "The cat walks away"):
        asyncio.run(model.set_paused(True))
        state = asyncio.run(model.set_prompt(text))
        assert state.world_id == 1 and state.completed_chunks == 3 and state.paused
        with pytest.raises(ApplicationError):
            asyncio.run(model.process_input())
        asyncio.run(model.set_paused(False))
        request = asyncio.run(model.process_input())
        assert request.prompt == text and request.anchor is None
    model.output.flush.assert_not_called()
    model.engine.reset.assert_not_called()


@pytest.mark.parametrize(
    "command,args",
    [
        ("start", (" ", 0)),
        ("set_prompt", (" ",)),
        ("set_paused", (True,)),
        ("reset", ()),
    ],
)
def test_refusal_changes_nothing_and_broadcasts_nothing(command, args):
    model, messages = app()
    before = model._status()
    with pytest.raises(CommandError):
        asyncio.run(getattr(model, command)(*args))
    assert model._status() == before and not messages
    model.engine.generate.assert_not_called()


def test_reset_keeps_current_seed_unless_explicitly_changed():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    for seed, expected in ((None, 42), (77, 77), (None, 77)):
        asyncio.run(model.reset(seed))
        assert asyncio.run(model.process_input()).anchor.seed == expected
        assert model.state._completed == 0


def test_stop_flushes_resets_and_preserves_staged_prompt():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    state = asyncio.run(model.stop_video())
    assert not state.active and state.prompt == "A garden" and state.output_frames == 0
    model.engine.reset.assert_called_once_with()
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    assert asyncio.run(model.start("", 0)).active


def test_join_is_private_and_does_not_reset_shared_world():
    model, messages = app()
    asyncio.run(model.start("A garden", 42))
    messages.clear()
    sent = []

    async def send(message):
        sent.append(message)

    client = Mock()
    client.send = send
    asyncio.run(model.on_connected(client))
    assert sent == [model._status()] and not messages
    model.engine.reset.assert_not_called()
    assert "on_disconnected" not in WaveForcing.__dict__


def test_failure_never_acknowledges_or_increments():
    model, _ = app()
    asyncio.run(model.start("A garden", 42))
    failure = RuntimeError("native failed")
    with pytest.raises(RuntimeError, match="native failed"):
        asyncio.run(model.process_output(StepOutcome(error=failure, elapsed=1)))
    assert model.state._completed == 0 and model.state._applied_world_id is None


def test_end_only_resets_healthy_runner_and_new_session_is_empty():
    model, _ = app()
    model.engine.healthy = False
    model.on_session_ended()
    model.engine.reset.assert_not_called()
    model.engine.healthy = True
    model.on_session_ended()
    model.engine.reset.assert_called_once_with()
    model.state = WaveForcingState()
    assert model._status().seed is None and not model._status().active


def test_model_bookkeeping_counts_only_successful_compute_and_keeps_seed():
    model = WaveForcingModel()
    backend = Mock()
    backend.step.return_value = result().frames
    model._backend = backend
    with pytest.raises(NoAnchor):
        model.generate(WaveForcingInput(world_id=7, prompt="garden"))
    value = model.generate(
        WaveForcingInput(world_id=7, prompt="garden", anchor=Anchor(seed=42))
    )
    assert value.chunk_index == 1 and value.world_id == 7
    value = model.generate(WaveForcingInput(world_id=7, prompt="cat"))
    assert value.chunk_index == 2
    backend.start.assert_called_once_with("garden", seed=42)
    backend.step.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError):
        model.generate(WaveForcingInput(world_id=7, prompt="rain"))
    assert model._chunk_index == 2
    backend.step.side_effect = None
    assert (
        model.generate(
            WaveForcingInput(world_id=8, prompt="beach", anchor=Anchor(seed=7))
        ).chunk_index
        == 1
    )
    model.reset()
    assert model._world_id is None and model._chunk_index == 0
    backend.reset.assert_called_once_with()


def test_input_and_result_are_frozen():
    value = WaveForcingInput(world_id=1, prompt="garden")
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.prompt = "new"


def test_schema_exposes_only_real_commands_without_rollout_cap():
    from reactor_runtime import ReactorApp

    contract = ModelContract.of(WaveForcing)
    assert set(contract.commands) == {
        "get_state",
        "start",
        "set_prompt",
        "set_paused",
        "reset",
        "stop",
    }
    assert not WaveForcing.__dict__.get("fps")
    assert WaveForcing.stop is ReactorApp.stop


def test_model_imports_without_runtime_or_torch():
    import subprocess
    import sys

    directory = Path(__file__).resolve().parents[1]
    check = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','reactor_runtime','naive_rt'):raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import waveforcing_model
"""
    subprocess.run(
        [sys.executable, "-c", check], cwd=directory, check=True, capture_output=True
    )


@pytest.mark.parametrize("position", [0, 1020, 1023, 1024, 4096])
def test_absolute_rope_matches_full_analytic_table_beyond_native_limit(
    monkeypatch, position
):
    import sys
    import types

    torch = pytest.importorskip("torch")
    from waveforcing_rope import install

    def table(length, width):
        phase = torch.outer(
            torch.arange(length),
            1.0
            / torch.pow(10000, torch.arange(0, 2 * width, 2).double() / (2 * width)),
        )
        return torch.polar(torch.ones_like(phase), phase)

    # A full absolute table is the reference; the adapter may retain only the
    # fixed native table and compute out-of-range temporal rows on demand.
    def reference(x, grid, frequencies, start_frame=0):
        temporal, vertical, horizontal = frequencies.split([22, 21, 21], dim=1)
        angles = torch.cat(
            [
                temporal[start_frame : start_frame + 3, None, None].expand(3, 2, 2, 22),
                vertical[None, :2, None].expand(3, 2, 2, 21),
                horizontal[None, None, :2].expand(3, 2, 2, 21),
            ],
            dim=-1,
        ).reshape(12, 1, 64)
        return (
            torch.view_as_real(
                torch.view_as_complex(x.double().reshape(2, 12, 2, 64, 2)) * angles
            )
            .flatten(3)
            .to(x.dtype)
        )

    native = types.ModuleType("causal_model")
    native.causal_rope_apply = reference
    package = types.ModuleType("naive_rt.rolling_forcing.wan.modules")
    package.causal_model = native
    monkeypatch.setitem(sys.modules, "naive_rt.rolling_forcing.wan.modules", package)
    install()
    full = torch.cat(
        [
            table(position + 3, 22),
            table(max(position + 3, 2), 21)[: position + 3],
            table(max(position + 3, 2), 21)[: position + 3],
        ],
        dim=1,
    )
    limited = torch.cat([table(1024, 22), table(1024, 21), table(1024, 21)], dim=1)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((2, 12, 2, 128), generator=generator).to(torch.bfloat16)
    grid = torch.tensor([[3, 2, 2], [3, 2, 2]])
    expected = reference(x, grid, full, start_frame=position)
    actual = native.causal_rope_apply(x, grid, limited, start_frame=position)
    assert torch.equal(actual, expected)
    assert limited.shape[0] == 1024
