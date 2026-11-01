#!/usr/bin/env python3
"""Re-apply the client's ok rule and TPOT definition (D4) to raw records written before it
existed, and recompute the summary from the stored per-request records. Marks the record
with rescored = "D4". Only requests whose error was 'short output' can change.

    .venv/bin/python scripts/kv_vram/rescore.py output/raw/kv_base_b24_q2.json ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402

for p in sys.argv[1:]:
    rec = json.loads(Path(p).read_text())
    mt = rec["max_tokens"]
    rs = []
    for r in rec["records"]:
        r = dict(r)
        r.setdefault("n_chunks", len(r["itls"]) + 1 if r["ttft"] is not None else 0)
        r["prompt_len"] = 6272
        if (r["error"] or "").startswith("short output"):
            r["ok"] = r["n_chunks"] > 1
            if r["ok"]:
                r["error"] = None
        rs.append(r)
    dur = rec["summary"]["duration_s"]
    rec.setdefault("summary_strict", rec["summary"])
    rec["summary"] = client.summarize(rs, dur, mt)
    rec["records"] = [{k: r[k] for k in ("index", "session_id", "t_submit", "ttft", "e2e", "ok",
                                         "error", "n_chunks", "itls")} for r in rs]
    n_failed_old = rec["n_failed"]
    rec["n_failed"] = rec["summary"]["n_failed"]
    md = rec["metrics_delta"]["decode"]
    rec["decode_gen_tokens"] = md.get("vllm:generation_tokens_total")
    rec["tokens_complete"] = rec["decode_gen_tokens"] == rec["requests"] * mt
    rec["ok"] = (not rec["hung"] and rec["n_failed"] == 0 and not any(rec["markers"].values())
                 and (rec.get("decode_hit_frac") or 0) >= 0.95 and rec["tokens_complete"])
    rec["rescored"] = "D4"
    Path(p).write_text(json.dumps(rec, indent=1))
    print(f"{rec['name']}: failed {n_failed_old} -> {rec['n_failed']}, ok={rec['ok']}, "
          f"chunk mismatches {rec['summary']['n_chunk_mismatch']}")
