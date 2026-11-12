from io import BytesIO
import unittest
import av
import numpy as np
from wave_rt.interactive.video import encode_chunk

class VideoTest(unittest.TestCase):
    def test_planar_and_packed_rgb_encode_identically_without_mutating_input(self):
        rng = np.random.default_rng(7)
        # Stay below VBV saturation: the original multithreaded x264 path
        # itself is not byte-deterministic on incompressible random noise.
        packed = rng.integers(0, 8, (9, 480, 832, 3), dtype=np.uint8)
        packed += np.linspace(0, 200, 832, dtype=np.uint8)[None, None, :, None]
        original = packed.copy()
        planar = [frame.transpose(2, 0, 1).copy().transpose(1, 2, 0) for frame in packed]
        self.assertFalse(planar[0].flags.c_contiguous)
        self.assertEqual(encode_chunk(packed, 16), encode_chunk(planar, 16))
        np.testing.assert_array_equal(packed, original)
        np.testing.assert_array_equal(planar, original)

    def test_independent_initial_and_continuation_fragments(self):
        for count in (9, 12):
            frames = [np.full((480, 832, 3), n * 15, dtype=np.uint8) for n in range(count)]
            data = encode_chunk(frames)
            self.assertIn(b"moof", data)
            with av.open(BytesIO(data)) as container:
                stream = container.streams.video[0]
                self.assertEqual(stream.codec_context.extradata[1:4].hex(), "42c01f")
                decoded = list(container.decode(video=0))
            self.assertEqual(len(decoded), count)
            self.assertTrue(decoded[0].key_frame)
            for n, frame in enumerate(decoded):
                self.assertAlmostEqual(float(frame.pts * frame.time_base), n / 24)
                self.assertEqual((frame.width, frame.height), (832, 480))
                self.assertLess(abs(frame.to_ndarray(format="rgb24").mean() - n * 15), 4)

    def test_bitrate_bound_holds_for_incompressible_frames(self):
        rng = np.random.default_rng(0)
        for count in (9, 12):
            frames = [rng.integers(0, 256, (480, 832, 3), dtype=np.uint8) for _ in range(count)]
            self.assertGreater(len(encode_chunk(frames, 16)), 150_000)
            data = encode_chunk(frames, 16, kbps=1000)
            # 5 % for the container; the bound is on the average over the chunk.
            self.assertLess(len(data) * 8, 1000_000 * count / 16 * 1.05)
            with av.open(BytesIO(data)) as container:
                self.assertEqual(len(list(container.decode(video=0))), count)
        with self.assertRaises(ValueError):
            encode_chunk(frames, 16, kbps=100)
