#!/usr/bin/env python3
import json
from pathlib import Path


OUT_PATH = Path("prompts.jsonl")
NUM_PROMPTS = 256


BASE_PARAGRAPH = (
    "Urban transportation systems are under pressure from population growth, "
    "traffic congestion, air pollution, and changing work patterns. "
    "City planners are increasingly interested in balancing public transit, "
    "walkability, cycling infrastructure, and new mobility services. "
    "At the same time, transportation policy must consider cost, accessibility, "
    "equity, resilience, and the long-term environmental impact of infrastructure choices. "
)

TASKS = [
    "Summarize the following paragraph in one concise academic paragraph.",
    "Rewrite the following paragraph in a more formal style.",
    "Explain the main policy trade-offs discussed in the following paragraph.",
    "List the key planning concerns mentioned in the following paragraph.",
]


def build_prompt(i: int) -> str:
    task = TASKS[i % len(TASKS)]
    repeated = " ".join([BASE_PARAGRAPH] * 3)
    return (
        f"{task}\n\n"
        f"Text:\n{repeated}\n\n"
        f"Response:"
    )


def main() -> None:
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(NUM_PROMPTS):
            row = {
                "id": i,
                "prompt": build_prompt(i),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {NUM_PROMPTS} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()