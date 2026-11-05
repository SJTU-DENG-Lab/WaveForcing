"""Build the RF pipeline without loading VAE weights or creating a decode stage."""


def build_dit_pipeline(server_args):
    from sglang.multimodal_gen.runtime.pipelines.wan_rolling_forcing_pipeline import (
        WanRollingForcingPipeline,
    )

    class DiTOnlyPipeline(WanRollingForcingPipeline):
        _required_config_modules = [
            name for name in WanRollingForcingPipeline._required_config_modules if name != 'vae'
        ]

        def add_standard_decoding_stage(self, *args, **kwargs):
            return self

    # Wan2.1 T2V latent geometry comes from pipeline_config, not a VAE forward.
    return DiTOnlyPipeline(server_args.model_path, server_args)


def build_role_pipeline(server_args, *, load_vae, load_text):
    """Construct only the components consumed by this diffusion rank."""
    from sglang.multimodal_gen.runtime.pipelines.wan_rolling_forcing_pipeline import WanRollingForcingPipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage

    excluded = (set() if load_vae else {'vae'}) | (set() if load_text else {'text_encoder', 'tokenizer'})

    class RolePipeline(WanRollingForcingPipeline):
        _required_config_modules = [name for name in WanRollingForcingPipeline._required_config_modules
                                    if name not in excluded]

        def add_standard_decoding_stage(self, *args, **kwargs):
            return super().add_standard_decoding_stage(*args, **kwargs) if load_vae else self

        def add_standard_text_encoding_stage(self, *args, **kwargs):
            if load_text:
                return super().add_standard_text_encoding_stage(*args, **kwargs)
            # Retain stage indices. This placeholder is never executed: request
            # conditioning arrives from the actual encoding rank instead.
            return self.add_stage(TextEncodingStage([], []))

    return RolePipeline(server_args.model_path, server_args)
