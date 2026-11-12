"""Independently decodable H.264/fMP4 chunks for one continuous MSE timeline."""

from fractions import Fraction
from io import BytesIO

MIME = 'video/mp4; codecs="avc1.42C01F"'


def encode_chunk(frames, fps=24, kbps=None):
    """Start each chunk with an IDR, so future branches share no codec state.

    Model/VAE state stays lossless; only the display transport is compressed.
    Relative timestamps are placed on the session timeline by SourceBuffer.
    `kbps` bounds the average bitrate of a chunk, for links slower than the
    default 2.5 Mbit/s peak; quality drops instead of the chunk growing.
    """
    import av
    import numpy as np

    if fps <= 0 or len(frames) not in (9, 12):
        raise ValueError("Expected 9 initial or 12 continuation frames and positive FPS")
    if kbps is None:
        maxrate = bufsize = 2500
    elif kbps < 200:
        raise ValueError("Video bitrate below 200 kbit/s is not usable at 832x480")
    else:
        # Every chunk is a fresh encode that may also spend most of the
        # buffer, so: maxrate + 0.9 * bufsize / duration <= kbps.
        bufsize = kbps // 4
        maxrate = kbps - int(0.9 * bufsize * fps / len(frames)) - 1
    height, width, channels = frames[0].shape
    if (height, width, channels) != (480, 832, 3):
        raise ValueError("Interactive video is validated at 832x480 RGB")
    output = BytesIO()
    with av.open(output, "w", format="mp4", options={
        "movflags": "frag_keyframe+empty_moov+default_base_moof",
    }) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        stream.options = {
            "preset": "veryfast", "tune": "zerolatency", "profile": "baseline",
            "level": "3.1", "crf": "23",
            "x264-params": "keyint=12:min-keyint=12:scenecut=0:bframes=0:"
                           f"vbv-maxrate={maxrate}:vbv-bufsize={bufsize}:threads=2",
        }
        for index, rgb in enumerate(frames):
            # The VAE supplies an HWC view of planar tensor storage. PyAV's
            # from_ndarray serializes that strided view and copies it again.
            # Pack once with NumPy, then let PyAV retain the buffer directly.
            # Pixel values, RGB->YUV conversion and x264 settings stay identical.
            frame = av.VideoFrame.from_numpy_buffer(np.ascontiguousarray(rgb), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    data = output.getvalue()
    # Fail closed if an encoder upgrade changes the declared browser codec.
    avcc = data.find(b"avcC")
    if avcc < 0 or data[avcc + 5:avcc + 8] != bytes.fromhex("42c01f"):
        raise RuntimeError("Encoded AVC profile does not match the browser MIME")
    return data
