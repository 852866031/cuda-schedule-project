"""Drop-in compiled LMCache kernels for the engines in .venv-matched. Inert unless
KV_COPS=1. Loads the c_ops .so built in .venv-lmc (same torch 2.9.1+cu128 ABI) under the
module name `lmcache.c_ops` before lmcache's own import attempt, so lmcache/__init__
finds a working backend instead of falling back to non_cuda_equivalents. Nothing else in
LMCache or vLLM changes.
"""
import os
import sys

if os.environ.get("KV_COPS") == "1":
    import importlib.util

    _so = os.environ.get(
        "KV_COPS_SO",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                     ".venv-lmc", "lib", "python3.12", "site-packages", "lmcache",
                     "c_ops.cpython-312-x86_64-linux-gnu.so"))
    _so = os.path.abspath(_so)

    class _COpsFinder:
        """Serve `lmcache.c_ops` from the compiled .so, whoever imports it first."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "lmcache.c_ops":
                return None
            return importlib.util.spec_from_file_location("lmcache.c_ops", _so)

    sys.meta_path.insert(0, _COpsFinder())
    print(f"[cops_shim] lmcache.c_ops -> {_so}", file=sys.stderr, flush=True)
