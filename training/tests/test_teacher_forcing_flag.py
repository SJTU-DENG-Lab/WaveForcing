"""CPU numerical coverage for explicit teacher-forcing mode and recomputation."""

from contextlib import ExitStack
import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from wf_training.wan.modules import causal_model as causal
from wf_training.wan.modules import model as wan


def _sdpa(q, k, v, *, k_lens=None, **kwargs):
    mask = None
    if k_lens is not None:
        mask = (torch.arange(k.shape[1])[None] < k_lens[:, None])[:, None, None]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)


def _dense_flex(query, key, value, block_mask):
    mask = create_mask(
        block_mask.mask_mod, query.shape[0], query.shape[1],
        query.shape[2], key.shape[2], device=query.device,
    )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


class TeacherForcingFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _check_attention_mode(self, teacher_forcing):
        torch.manual_seed(314)
        attention = causal.CausalWanSelfAttention(dim=16, num_heads=2).double()
        reference = copy.deepcopy(attention)
        x = torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        grid = torch.tensor([[2, 1, 2]])
        freqs = torch.cat([wan.rope_params(16, 4), wan.rope_params(16, 2),
                           wan.rope_params(16, 2)], dim=1)

        # False deliberately looks like two copies to the old length heuristic;
        # True deliberately does not. Only the explicit argument selects mode.
        seq_lens = torch.tensor([8 if teacher_forcing else 4])

        def unpadded_attention(query, key, value, block_mask):
            # Exclude the production kernel's alignment padding from softmax.
            return F.scaled_dot_product_attention(
                query, key[:, :, :8], value[:, :, :8])

        with patch.object(causal, '_get_flex_attention', return_value=unpadded_attention):
            actual = attention(x, seq_lens, grid, freqs, None,
                               teacher_forcing=teacher_forcing)

        q = reference.norm_q(reference.q(reference_x)).reshape(1, 8, 2, 8)
        k = reference.norm_k(reference.k(reference_x)).reshape(1, 8, 2, 8)
        v = reference.v(reference_x).reshape(1, 8, 2, 8)

        def rotated(value, repeat):
            if repeat:
                return torch.cat([wan.rope_apply(half, grid, freqs)
                                  for half in value.chunk(2, dim=1)], dim=1)
            return wan.rope_apply(value, grid, freqs)

        def expected_output(repeat):
            return reference.o(_sdpa(rotated(q, repeat), rotated(k, repeat), v).flatten(2))

        expected = expected_output(teacher_forcing)
        alternative = expected_output(not teacher_forcing)
        self.assertGreater((expected - alternative).abs().max().item(), 1e-4)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-10)
        for (name, parameter), (_, expected_parameter) in zip(
                attention.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter.grad, expected_parameter.grad,
                                           rtol=1e-10, atol=1e-10)

    def test_teacher_forcing_repeats_rope_even_when_lengths_do_not_imply_it(self):
        self._check_attention_mode(True)

    def test_ordinary_attention_does_not_infer_teacher_forcing_from_lengths(self):
        self._check_attention_mode(False)

    def test_model_propagates_mode_through_checkpoint_forward_and_backward(self):
        for teacher_forcing in (False, True):
            reference_result = None
            for checkpointed in (False, True):
                with self.subTest(teacher_forcing=teacher_forcing, checkpointed=checkpointed):
                    torch.manual_seed(427)
                    model = causal.CausalWanModel(
                        patch_size=(1, 2, 2), text_len=3, in_dim=2, dim=16,
                        ffn_dim=32, freq_dim=8, text_dim=8, out_dim=2,
                        num_heads=2, num_layers=1,
                    ).double()
                    model.num_frame_per_block = 1
                    model.gradient_checkpointing = checkpointed
                    with torch.no_grad():
                        model.head.head.weight.normal_(std=0.1)
                    x = torch.randn(1, 2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
                    clean = torch.randn_like(x, requires_grad=True)
                    context = [torch.randn(2, 8, dtype=torch.float64)]
                    time = torch.tensor([[100., 300., 700.]], dtype=torch.float64)
                    observed = {'block': [], 'attention': []}

                    def record(label):
                        def hook(module, args, kwargs):
                            observed[label].append(kwargs.get('teacher_forcing'))
                        return hook

                    with ExitStack() as stack:
                        stack.enter_context(patch.object(wan, 'flash_attention', side_effect=_sdpa))
                        stack.enter_context(patch.object(causal, '_get_flex_attention',
                                                         return_value=_dense_flex))
                        for label, module in [('block', model.blocks[0]),
                                              ('attention', model.blocks[0].self_attn)]:
                            handle = module.register_forward_pre_hook(record(label), with_kwargs=True)
                            stack.callback(handle.remove)
                        kwargs = {'clean_x': clean, 'aug_t': time * 0.1} if teacher_forcing else {}
                        output = model(x, t=time, context=context, seq_len=16, **kwargs)
                        self.assertEqual(tuple(output.shape), tuple(x.shape))
                        for flags in observed.values():
                            self.assertEqual(flags, [teacher_forcing])
                        output.square().mean().backward()

                    for flags in observed.values():
                        self.assertEqual(flags, [teacher_forcing] * (2 if checkpointed else 1))
                    gradients = {name: parameter.grad.detach().clone()
                                 for name, parameter in model.named_parameters()
                                 if parameter.grad is not None}
                    self.assertIn('blocks.0.self_attn.q.weight', gradients)
                    for gradient in [x.grad, *gradients.values()]:
                        self.assertTrue(torch.isfinite(gradient).all().item())
                    if teacher_forcing:
                        self.assertIsNotNone(clean.grad)
                        self.assertGreater(clean.grad.abs().max().item(), 0)
                        gradients['clean_input'] = clean.grad.detach().clone()
                    else:
                        self.assertIsNone(clean.grad)
                    gradients['noisy_input'] = x.grad.detach().clone()
                    if reference_result is None:
                        reference_result = output.detach(), gradients
                    else:
                        torch.testing.assert_close(output, reference_result[0], rtol=1e-10, atol=1e-10)
                        self.assertEqual(gradients.keys(), reference_result[1].keys())
                        for name, gradient in gradients.items():
                            with self.subTest(gradient=name):
                                torch.testing.assert_close(gradient, reference_result[1][name],
                                                           rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
