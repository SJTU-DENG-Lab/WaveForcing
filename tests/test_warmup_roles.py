"""Preflight must respect rank-specific model ownership."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from wave_rt.denoiser.engine import WaveDenoiser


class WarmupRolesTest(unittest.TestCase):
    def make_engine(self, rank=4, vae_stages=3, dit_only=False):
        engine = object.__new__(WaveDenoiser)
        engine.rank, engine.store_rank = rank, 4
        engine.device, engine.dtype = 'cpu', torch.bfloat16
        engine.npb, engine.dsl_list = 3, [1000.]
        engine.cfg = SimpleNamespace(vae_stages=vae_stages, dit_only=dit_only)
        engine.be = SimpleNamespace(
            latent_channels=16, ctx=SimpleNamespace(height=2, width=2),
            set_exchange_fn=Mock(), forward_chunk=Mock(), decode=Mock())
        return engine

    @patch('wave_rt.denoiser.exchange.common.settings.TICK_PROFILE', False)
    @patch('torch.cuda.synchronize')
    def test_only_serial_store_warms_local_decoder(self, synchronize):
        for rank, stages, dit_only, decodes in (
                (4, 3, False, 0), (4, 0, False, 1),
                (0, 0, False, 0), (4, 0, True, 0)):
            with self.subTest(rank=rank, stages=stages, dit_only=dit_only):
                engine = self.make_engine(rank, stages, dit_only)
                if not decodes:
                    engine.be.decode.side_effect = RuntimeError('This rank does not own a VAE decoder')
                engine._warmup()
                self.assertEqual(engine.be.decode.call_count, decodes)
                self.assertEqual(engine.be.forward_chunk.call_count, 3)
                engine.be.set_exchange_fn.assert_called_with(None)

    def test_failed_forward_releases_exchange_hook(self):
        engine = self.make_engine()
        engine.be.forward_chunk.side_effect = RuntimeError('forward failed')
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            engine._warmup()
        engine.be.set_exchange_fn.assert_called_with(None)
        engine.be.decode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
