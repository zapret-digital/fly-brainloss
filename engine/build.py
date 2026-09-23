"""Build the engine shared library next to this file.

Uses the system C compiler when there is one (cc / clang / gcc), otherwise the zig toolchain
from the `ziglang` pip package, which works on Windows without Visual Studio.
    python engine/build.py            # optimised for this CPU
    python engine/build.py --portable # no -march=native
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "evlif.c"
OUT = HERE / {"win32": "evlif.dll", "darwin": "libevlif.dylib"}.get(sys.platform, "libevlif.so")


def compiler():
    if sys.platform != "win32":
        for cc in ("cc", "clang", "gcc"):
            if shutil.which(cc):
                return [cc]
    try:
        import ziglang  # noqa: F401
    except ImportError:
        sys.exit("no C compiler found: pip install ziglang")
    return [sys.executable, "-m", "ziglang", "cc"]


def main():
    flags = ["-O3", "-shared"]
    if "--portable" not in sys.argv:
        flags.append("-march=native")
    if sys.platform != "win32":
        flags += ["-fPIC", "-lm"]
    cmd = compiler() + flags + ["-o", str(OUT), str(SRC)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    for junk in (OUT.with_suffix(".pdb"), OUT.with_suffix(".lib")):
        junk.unlink(missing_ok=True)
    print("built", OUT)


if __name__ == "__main__":
    main()
