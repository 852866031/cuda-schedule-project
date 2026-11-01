#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROMPT_PATH = THIS_DIR / "prompts.jsonl"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 32
DTYPE = torch.float16
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to write runtime metrics as JSON.",
    )
    return p.parse_args()


def load_prompts(path: Path) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])
    return prompts


def batchify(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    prompts = load_prompts(PROMPT_PATH)
    batches = batchify(prompts, BATCH_SIZE)

    with torch.cuda.nvtx.range("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.cuda.nvtx.range("load_model"):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        model.eval()

    total_generated_tokens = 0
    total_requests = 0

    t0 = time.perf_counter()

    for batch_id, batch_prompts in enumerate(batches):
        with torch.cuda.nvtx.range(f"batch_{batch_id}_tokenize"):
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

        generated = input_ids
        current_attention_mask = attention_mask
        past_key_values = None

        with torch.no_grad():
            with torch.cuda.nvtx.range(f"batch_{batch_id}_prefill"):
                outputs = model(
                    input_ids=generated,
                    attention_mask=current_attention_mask,
                    use_cache=True,
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens = [next_token]

            for step in range(MAX_NEW_TOKENS - 1):
                with torch.cuda.nvtx.range(f"batch_{batch_id}_decode_step_{step}"):
                    current_attention_mask = torch.cat(
                        [
                            current_attention_mask,
                            torch.ones(
                                (current_attention_mask.shape[0], 1),
                                device=DEVICE,
                                dtype=current_attention_mask.dtype,
                            ),
                        ],
                        dim=1,
                    )

                    outputs = model(
                        input_ids=next_token,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values

                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated_tokens.append(next_token)

        with torch.cuda.nvtx.range(f"batch_{batch_id}_postprocess"):
            new_tokens = torch.cat(generated_tokens, dim=1)
            _ = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        total_generated_tokens += new_tokens.numel()
        total_requests += len(batch_prompts)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    toks_per_s = total_generated_tokens / total_time if total_time > 0 else 0.0
    reqs_per_s = total_requests / total_time if total_time > 0 else 0.0

    metrics = {
        "model_name": MODEL_NAME,
        "prompt_path": str(PROMPT_PATH),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": str(DTYPE),
        "device": DEVICE,
        "total_requests": total_requests,
        "total_generated_tokens": total_generated_tokens,
        "total_time_sec": total_time,
        "tokens_per_sec": toks_per_s,
        "requests_per_sec": reqs_per_s,
    }

    print(f"Model: {MODEL_NAME}")
    print(f"Requests: {total_requests}")
    print(f"Generated tokens: {total_generated_tokens}")
    print(f"Total time (s): {total_time:.4f}")
    print(f"Tokens/s: {toks_per_s:.2f}")
    print(f"Requests/s: {reqs_per_s:.2f}")

    if args.metrics_output is not None:
        out_path = Path(args.metrics_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics JSON to {out_path}")


if __name__ == "__main__":
    main()