"""Rank-local text encoder loading and residency for WaveRT's direct stages."""
from contextlib import contextmanager
from copy import copy


@contextmanager
def local_text_encoder_loading(server_args):
    # Upstream text_encoder_cpu_offload can create a WORLD DeviceMesh even when
    # use_fsdp_inference=False. Only rank 0 loads T5 in shared-conditioning mode;
    # peers never enter that collective. Keep plain CPU staging without FSDP.
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import TextEncoderLoader

    original = TextEncoderLoader.load_model
    target_args = server_args

    def load_local(loader, model_path, model_config, server_args, dtype='fp16', cpu_offload_flag=None):
        args = server_args
        if args is target_args:
            if cpu_offload_flag is None:
                cpu_offload_flag = args.text_encoder_cpu_offload
            args = copy(args)
            args.text_encoder_cpu_offload = False
        return original(loader, model_path, model_config, args, dtype,
                        cpu_offload_flag=cpu_offload_flag)

    TextEncoderLoader.load_model = load_local
    try:
        yield
    finally:
        TextEncoderLoader.load_model = original


def encode_local_text(stage, batch, server_args, device):
    # WaveRT bypasses PipelineExecutor and its component residency manager.
    # Move plain modules explicitly, retaining serving residency when requested.
    try:
        for encoder in stage.text_encoders:
            encoder.to(device)
        return stage(batch, server_args)
    finally:
        if server_args.text_encoder_cpu_offload:
            for encoder in stage.text_encoders:
                encoder.to('cpu')
