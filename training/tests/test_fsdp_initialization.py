"""CPU coverage of rank-zero loading and parameter-only meta construction."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from accelerate import init_empty_weights

from wf_training.utils import distributed, model_init


class TinyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3, dtype=dtype))
        self.child = nn.Linear(3, 2, dtype=dtype)
        self.register_buffer("scale", torch.tensor([3.0]))
        self.freqs = torch.polar(torch.ones(4), torch.arange(4, dtype=torch.float32))


class TinyWan(TinyModel):
    def __init__(self):
        super().__init__()
        self.num_frame_per_block = 1
        self.independent_first_frame = False
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


class TinyDiffusionWrapper(nn.Module):
    def __init__(self, model=None, model_name="tiny", **kwargs):
        super().__init__()
        self.model = model if model is not None else TinyWan()
        self.model_name = model_name

    def enable_gradient_checkpointing(self):
        self.model.enable_gradient_checkpointing()


class FSDPProxy(nn.Module):
    """Mirror FSDP attribute reads; attribute writes stay on the wrapper."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@contextmanager
def fake_rank(rank, peer_error=None):
    def gather(errors, local_error):
        errors[:] = [None, None]
        errors[rank] = local_error
        if peer_error is not None:
            errors[1 - rank] = peer_error

    with mock.patch.object(model_init.dist, "get_rank", return_value=rank), \
            mock.patch.object(model_init.dist, "get_world_size", return_value=2), \
            mock.patch.object(model_init.dist, "all_gather_object", side_effect=gather) as exchange:
        yield exchange


class RankZeroInitializationTest(unittest.TestCase):
    def test_nonzero_rank_builds_meta_without_reading_weights(self):
        loader = mock.Mock(side_effect=AssertionError("nonzero rank read checkpoint"))
        factory = mock.Mock(side_effect=lambda meta: TinyModel())
        with fake_rank(1) as exchange, \
                mock.patch.object(model_init, "fsdp_wrap", side_effect=lambda module, **kw: module) as wrap:
            module = model_init.build_rank0_model(
                factory, state_loader=loader, trainable=True,
                sharding_strategy="full", mixed_precision=True)
        loader.assert_not_called()
        factory.assert_called_once_with(True)
        self.assertTrue(all(parameter.is_meta for parameter in module.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in module.parameters()))
        self.assertEqual(module.freqs.device.type, "cpu")
        self.assertEqual(module.scale.device.type, "cpu")
        torch.testing.assert_close(module.scale, torch.tensor([3.0]))
        self.assertTrue(torch.isfinite(module.freqs).all())
        self.assertIsNone(exchange.call_args.args[1])
        self.assertTrue(wrap.call_args.kwargs["sync_module_states"])

    def test_rank_zero_loads_bf16_state_before_wrap_with_fp32_master(self):
        state = {key: torch.full_like(value, 7, dtype=torch.bfloat16)
                 for key, value in TinyModel().state_dict().items()}
        loader = mock.Mock(return_value=state)

        def check_wrap(module, **kwargs):
            for value in module.state_dict().values():
                torch.testing.assert_close(value, torch.full_like(value, 7))
            self.assertTrue(kwargs["sync_module_states"])
            return module

        with fake_rank(0), mock.patch.object(model_init, "fsdp_wrap", side_effect=check_wrap):
            module = model_init.build_rank0_model(
                lambda meta: TinyModel(), state_loader=loader, trainable=True)
        loader.assert_called_once_with()
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in module.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in module.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in module.parameters()))

    def test_frozen_models_keep_fp32_parameters_without_gradients(self):
        with fake_rank(0), \
                mock.patch.object(model_init, "fsdp_wrap", side_effect=lambda module, **kw: module):
            module = model_init.build_rank0_model(lambda meta: TinyModel())
        self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in module.parameters()))

    def test_rank_zero_load_error_is_exchanged_before_fsdp(self):
        loader = mock.Mock(side_effect=OSError("missing weights"))
        with fake_rank(0) as exchange, mock.patch.object(model_init, "fsdp_wrap") as wrap:
            with self.assertRaisesRegex(RuntimeError, "rank 0: OSError: missing weights"):
                model_init.build_rank0_model(lambda meta: TinyModel(), state_loader=loader)
        self.assertEqual(exchange.call_args.args[1], "rank 0: OSError: missing weights")
        wrap.assert_not_called()

    def test_peer_load_error_stops_nonzero_rank_before_fsdp(self):
        loader = mock.Mock(side_effect=AssertionError("nonzero rank read checkpoint"))
        with fake_rank(1, peer_error="rank 0: OSError: missing weights") as exchange, \
                mock.patch.object(model_init, "fsdp_wrap") as wrap:
            with self.assertRaisesRegex(RuntimeError, "rank 0: OSError: missing weights"):
                model_init.build_rank0_model(lambda meta: TinyModel(), state_loader=loader)
        self.assertIsNone(exchange.call_args.args[1])
        loader.assert_not_called()
        wrap.assert_not_called()

    def test_shape_mismatch_and_low_precision_master_fail_collectively(self):
        cases = (
            (lambda meta: TinyModel(), lambda: {"weight": torch.ones(1)}, "checkpoint does not match"),
            (lambda meta: TinyModel(torch.bfloat16), None, "Expected FP32 master"),
        )
        for factory, loader, message in cases:
            with self.subTest(message=message), fake_rank(0) as exchange, \
                    mock.patch.object(model_init, "fsdp_wrap") as wrap:
                with self.assertRaisesRegex(RuntimeError, message):
                    model_init.build_rank0_model(factory, state_loader=loader)
                self.assertIn(message, exchange.call_args.args[1])
                wrap.assert_not_called()


class MaterializationTest(unittest.TestCase):
    def test_materializes_only_local_meta_tensors_and_preserves_real_values(self):
        module = TinyModel()
        original_weight = module.weight
        original_buffer = module.scale
        original_freqs = module.freqs
        module.empty_weight = nn.Parameter(torch.empty(2, device="meta"), requires_grad=False)
        module.register_buffer("empty_buffer", torch.empty(3, device="meta"))
        module.child = nn.Linear(3, 2, device="meta")

        distributed.materialize_meta_parameters(module, torch.device("cpu"))
        self.assertIs(module.weight, original_weight)
        self.assertIs(module.scale, original_buffer)
        self.assertIs(module.freqs, original_freqs)
        torch.testing.assert_close(module.weight, torch.ones(2, 3))
        torch.testing.assert_close(module.scale, torch.tensor([3.0]))
        self.assertEqual(module.empty_weight.device.type, "cpu")
        self.assertEqual(module.empty_buffer.device.type, "cpu")
        self.assertFalse(module.empty_weight.requires_grad)
        self.assertTrue(module.child.weight.is_meta)
        distributed.materialize_meta_parameters(module.child, torch.device("cpu"))
        self.assertEqual(module.child.weight.device.type, "cpu")
        self.assertTrue(module.child.weight.requires_grad)

    def test_fsdp_wrap_configures_sync_and_compute_precision_without_casting_master(self):
        module = TinyModel()
        with mock.patch.object(distributed, "FSDP", return_value=mock.sentinel.fsdp) as fsdp, \
                mock.patch.object(torch.cuda, "current_device", return_value=0):
            result = distributed.fsdp_wrap(
                module, sharding_strategy="full", mixed_precision=True, sync_module_states=True)
        self.assertIs(result, mock.sentinel.fsdp)
        kwargs = fsdp.call_args.kwargs
        self.assertTrue(kwargs["sync_module_states"])
        self.assertTrue(kwargs["use_orig_params"])
        self.assertEqual(kwargs["sharding_strategy"], distributed.ShardingStrategy.FULL_SHARD)
        self.assertIs(kwargs["param_init_fn"].func, distributed.materialize_meta_parameters)
        policy = kwargs["mixed_precision"]
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in module.parameters()))


class WanMetaConstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import real model definitions without probing a CUDA device. No model
        # forward or attention kernel is used by these construction tests.
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            from wf_training.utils import wan_wrapper
        cls.wrapper = wan_wrapper

    def test_dmd_sets_block_config_on_underlying_model_through_nested_fsdp(self):
        from wf_training.model import dmd

        underlying = TinyWan()
        inner = FSDPProxy(underlying)
        outer = FSDPProxy(inner)
        generator = FSDPProxy(TinyDiffusionWrapper(model=outer))
        critic = TinyDiffusionWrapper()

        def initialize_base(owner, args, device):
            nn.Module.__init__(owner)
            owner.generator = generator
            owner.fake_score = critic
            owner.scheduler = SimpleNamespace(alphas_cumprod=None)

        args = SimpleNamespace(num_frame_per_block=3, independent_first_frame=True,
                               gradient_checkpointing=True, num_train_timestep=1000,
                               guidance_scale=3.0, distribution_loss="dmd", lpips_weight=0.0)
        with mock.patch.object(dmd, "FSDP", FSDPProxy), \
                mock.patch.object(dmd.RollingForcingModel, "__init__", initialize_base):
            model = dmd.DMD(args, device="cpu")

        self.assertEqual(underlying.num_frame_per_block, 3)
        self.assertTrue(underlying.independent_first_frame)
        self.assertTrue(underlying.gradient_checkpointing)
        self.assertTrue(critic.model.gradient_checkpointing)
        self.assertEqual(model.num_frame_per_block, 3)
        for wrapper in (inner, outer):
            self.assertNotIn("num_frame_per_block", vars(wrapper))
            self.assertNotIn("independent_first_frame", vars(wrapper))

    def test_rank_zero_factory_sets_each_role_before_fsdp_wrap(self):
        owner = SimpleNamespace(generator_name="student", real_model_name="teacher",
                                fake_model_name="critic")
        config = SimpleNamespace(
            generator_ckpt="", generator_fsdp_wrap_strategy="size",
            real_score_fsdp_wrap_strategy="size", fake_score_fsdp_wrap_strategy="size",
            text_encoder_fsdp_wrap_strategy="size", sharding_strategy="full",
            mixed_precision=True, num_frame_per_block=3, independent_first_frame=True,
            gradient_checkpointing=True, profile_memory=False,
        )
        snapshots = []

        def capture_before_wrap(module, **kwargs):
            if isinstance(module, TinyDiffusionWrapper):
                snapshots.append((
                    module.model_name, module.model.num_frame_per_block,
                    module.model.independent_first_frame, module.model.gradient_checkpointing,
                    all(parameter.requires_grad for parameter in module.parameters()),
                ))
            return FSDPProxy(module)

        with fake_rank(0), \
                mock.patch.object(self.wrapper, "WanDiffusionWrapper", TinyDiffusionWrapper), \
                mock.patch.object(self.wrapper, "WanTextEncoder",
                                  side_effect=lambda **kwargs: nn.Linear(3, 2)), \
                mock.patch.object(self.wrapper, "WanVAEWrapper", return_value=nn.Linear(3, 2)), \
                mock.patch.object(model_init, "fsdp_wrap", side_effect=capture_before_wrap):
            model_init.initialize_rank0_models(owner, config, device="cpu")

        self.assertEqual(snapshots, [
            ("student", 3, True, True, True),
            ("teacher", 1, False, False, False),
            ("critic", 1, False, True, True),
        ])
        self.assertTrue(owner.fsdp_initialized)
        self.assertTrue(all(not parameter.requires_grad for parameter in owner.text_encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in owner.vae.parameters()))

    def test_causal_and_bidirectional_wan_config_only_meta_construction(self):
        config = dict(model_type="t2v", text_len=8,
                      in_dim=16, dim=24, ffn_dim=32, freq_dim=8,
                      out_dim=16, num_heads=2, num_layers=1)
        for causal, model_class in ((True, self.wrapper.CausalWanModel),
                                    (False, self.wrapper.WanModel)):
            with self.subTest(causal=causal), \
                    mock.patch.object(self.wrapper, "model_directory", return_value="unused-model"), \
                    mock.patch.object(model_class, "load_config", return_value=config) as config_loader, \
                    mock.patch.object(model_class, "from_pretrained",
                                      side_effect=AssertionError("weight read")) as weight_loader, \
                    mock.patch.object(torch, "load", side_effect=AssertionError("weight read")), \
                    init_empty_weights(include_buffers=False):
                wrapped = self.wrapper.WanDiffusionWrapper(
                    model_name="tiny", is_causal=causal, load_pretrained=False)
            config_loader.assert_called_once_with("unused-model")
            weight_loader.assert_not_called()
            self.assertTrue(all(parameter.is_meta for parameter in wrapped.parameters()))
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in wrapped.parameters()))
            self.assertEqual(wrapped.model.freqs.device.type, "cpu")
            self.assertTrue(wrapped.model.freqs.is_complex())
            self.assertTrue(torch.isfinite(wrapped.model.freqs).all())
            self.assertEqual(wrapped.uniform_timestep, not causal)

    def test_t5_meta_construction_uses_real_small_t5_without_loading_weights(self):
        original_umt5 = self.wrapper.umt5_xxl

        def small_umt5(**kwargs):
            return original_umt5(vocab_size=32, dim=16, dim_attn=16, dim_ffn=24,
                                 num_heads=2, encoder_layers=1, decoder_layers=1,
                                 num_buckets=8, **kwargs)

        with mock.patch.object(self.wrapper, "umt5_xxl", side_effect=small_umt5) as constructor, \
                mock.patch.object(self.wrapper, "model_directory", return_value="unused-model"), \
                mock.patch.object(self.wrapper, "HuggingfaceTokenizer", return_value=mock.sentinel.tokenizer), \
                mock.patch("wf_training.utils.checkpoint.load_model_checkpoint",
                           side_effect=AssertionError("weight read")) as loader, \
                init_empty_weights(include_buffers=False):
            encoder = self.wrapper.WanTextEncoder(
                model_name="tiny", load_pretrained=False, init_device="meta")
        loader.assert_not_called()
        self.assertEqual(constructor.call_args.kwargs["device"], torch.device("meta"))
        self.assertTrue(all(parameter.is_meta for parameter in encoder.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        self.assertLess(sum(parameter.numel() for parameter in encoder.parameters()), 10000)


if __name__ == "__main__":
    unittest.main()
