"""Engine-side patches for the priority study, loaded via PYTHONPATH into every vLLM
process launched by pfp_launch.sh. Each is inert unless its env switch is set:

  async_save.py  LMC_ASYNC_SAVE=1    token #1 before LMCache's GPU->host copy (3B prefill)
  hp_signal.py   COLOC_HP_SIGNAL=1   publish this engine's execute_model busy window to
                                     /dev/shm/coloc_hp_busy for the CUPTI gate (orion_gate/)

Only one `sitecustomize` module is ever imported, so this one runs both files.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
for _f in ("async_save.py", "hp_signal.py"):
    _p = os.path.join(_here, _f)
    with open(_p) as _fh:
        exec(compile(_fh.read(), _p, "exec"), {"__name__": "pfp_site_" + _f[:-3], "__file__": _p})
