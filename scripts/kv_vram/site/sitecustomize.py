"""Engine-side patches for the KV-slot study, loaded via PYTHONPATH into the decode
engine only (kv_launch.sh, KV_TRACE=1). Each is inert unless its env switch is set:

  kv_trace.py   KV_TRACE=1   record-only scheduler / KV-load timeline (E1)

Only one `sitecustomize` module is ever imported, so this one runs every file.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
for _f in ("kv_trace.py",):
    _p = os.path.join(_here, _f)
    with open(_p) as _fh:
        exec(compile(_fh.read(), _p, "exec"), {"__name__": "kvv_site_" + _f[:-3], "__file__": _p})
