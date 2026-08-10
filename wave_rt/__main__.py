"""WaveRT entrypoint:  python -m wave_rt --rf-step 4 --wp-size 5 --num-frames 24

Before importing torch/sglang we must guarantee that ``libcudart.so.13`` (pulled
in transitively by deep_gemm on the sglang import path) is loadable.  ld.so only
reads LD_LIBRARY_PATH at process startup, so if the cu13 lib dir is missing we
inject it and re-exec this same interpreter once.  deep_gemm is NOT functionally
needed by Rolling Forcing -- this only satisfies the import chain.
"""

import glob
import os
import sys

_REEXEC_FLAG = "_WAVE_RT_LDPATH_FIXED"


def _ensure_cu13_on_ldpath() -> None:
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    # locate <venv>/lib/pythonX.Y/site-packages/nvidia/cu13/lib
    hits = glob.glob(
        os.path.join(sys.prefix, "lib", "python*", "site-packages",
                     "nvidia", "cu13", "lib")
    )
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    need = [h for h in hits if h not in ld.split(":")]
    os.environ[_REEXEC_FLAG] = "1"
    if need:
        os.environ["LD_LIBRARY_PATH"] = ":".join(need + ([ld] if ld else []))
        # re-exec so ld.so picks up the new search path
        os.execv(sys.executable, [sys.executable, "-m", "wave_rt", *sys.argv[1:]])


_ensure_cu13_on_ldpath()

import argparse  # noqa: E402

from wave_rt.config import WaveConfig  # noqa: E402
from wave_rt.launcher import launch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser("wave_rt", description="WaveRT wavefront runtime")
    WaveConfig.add_cli_args(parser)
    ns = parser.parse_args()
    cfg = WaveConfig.from_args(ns)
    launch(cfg)


if __name__ == "__main__":
    main()
