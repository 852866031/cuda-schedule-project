# CUDA scheduling project

Every branch of this repository, gathered in one place. Each directory under `branches/`
is the full tree of the branch of the same name, merged with its history, so
`git log -- branches/<name>/` and the original branches both remain usable.

**Start here:** [`project_summary/REPORT_SUMMARY.md`](project_summary/REPORT_SUMMARY.md) — the
project by theme (platform, glossary, findings, what was built, corrected beliefs, limitations,
where it points) — and [`project_summary/REPORT_CHRONOLOGICAL.md`](project_summary/REPORT_CHRONOLOGICAL.md)
— the same work in date order, March → October 2026. Both were fact-checked against the branch
documents in three independent review rounds; their Appendix B lists the discrepancies found.

| Branch | Directory | Merged at |
|---|---|---|
| `colocation` | [`branches/colocation/`](branches/colocation/) | `0baec85` |
| `vram-limit` | [`branches/vram-limit/`](branches/vram-limit/) | `9d8c830` |
| `kvcache-vram_size` | [`branches/kvcache-vram_size/`](branches/kvcache-vram_size/) | `d89a09d` |
| `literature-review` | [`branches/literature-review/`](branches/literature-review/) | `b6e16ae` |
| `gpu-interfere` | [`branches/gpu-interfere/`](branches/gpu-interfere/) | `e120026` |
| `util_examples` | [`branches/util_examples/`](branches/util_examples/) | `3fbb6bd` |

Gitignored files on each branch (raw results, engine logs, model weights, virtualenvs)
were not on the branches and are not here.

`branches/literature-review/reproduce/tally` is a submodule; fetch it with
`git submodule update --init`. Local edits to it are in
`branches/literature-review/reproduce/tally_local_changes.patch`.
