"""Reproduction entry for NanoFlow Llama3-8B on a single RTX 5090 (32GB).

Based on entry/run_llama3.py, with changes for this machine:
- weights/tokenizer come from the ungated NousResearch/Meta-Llama-3-8B-Instruct
  mirror (identical to meta-llama/Meta-Llama-3-8B-Instruct, which is gated);
- KV pool, batch sizes and sequence length shrunk to fit 32GB (original
  targets 80GB-class GPUs);
- per-cycle timing added to report throughput (tokens/s), which the original
  script does not measure.

Usage:
  python run_llama3_5090.py -l          # first run: convert+cache weights
  python run_llama3_5090.py             # subsequent runs: load cached weights
  python run_llama3_5090.py --correctness   # small-batch generation sanity test
"""
import os
import sys
import time
import argparse

sys.path.append("../")
sys.path.append("../utils")
sys.path.append('../pybind/build')

# Sized for 32GB: pages are 2MB each (32 layers x 16 tokens x 8 kv-heads x 128 x 2B x K/V)
KV_PAGES = int(os.environ.get("NANOFLOW_KV_PAGES", "5120"))  # 10.7 GB
os.environ["NANOFLOW_KV_PAGES"] = str(KV_PAGES)

SEQ_LEN = int(os.environ.get("NANOFLOW_SEQ_LEN", "512"))
GLOBAL_BATCH = int(os.environ.get("NANOFLOW_GLOBAL_BATCH", "2048"))
DECODE_BATCH = int(os.environ.get("NANOFLOW_DECODE_BATCH", "128"))
NUM_CYCLES = int(os.environ.get("NANOFLOW_CYCLES", "20"))

from utils.frontend import requestManager  # noqa: F401  (parity with original)
from utils.util_functions import prepare_weight
from transformers import AutoTokenizer
from utils.input_test import prefill_context

from models.llama3_FlashinferKVCache import Pipeline

HF_REPO = "NousResearch/Meta-Llama-3-8B-Instruct"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-l", "--load_hf_weight", action="store_true",
                        help="Convert HF safetensors and cache to disk")
arg_parser.add_argument("--correctness", action="store_true",
                        help="Run the small-batch correctness test instead of perf")
args = arg_parser.parse_args()

from huggingface_hub import snapshot_download
weight_dir = snapshot_download(HF_REPO, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
print(f"weights: {weight_dir}")

tokenizer = AutoTokenizer.from_pretrained(HF_REPO)

if args.load_hf_weight:
    # convert + cache weights, then exit: keeping the conversion pipeline and the
    # serving pipeline in one process needs 2x16GB on-GPU, which OOMs a 32GB card
    pipeline_weight_list = [(i, f"cuda:{i}", Pipeline()) for i in range(1)]
    prepare_weight(pipeline_weight_list, weight_dir)
    print("weight cache written; re-run without -l to serve")
    sys.exit(0)

pipeline = Pipeline()
pipeline.init(weight_dir, cached=True)


def test_performance():
    seq_len = SEQ_LEN
    global_batch_size = GLOBAL_BATCH
    decode_batch_size = DECODE_BATCH
    prefill_batch_size = global_batch_size - decode_batch_size

    prefill_context_ids = tokenizer.encode(prefill_context)  # length 1912
    # tile the context so we can slice arbitrary prefill sizes
    while len(prefill_context_ids) < max(prefill_batch_size, seq_len):
        prefill_context_ids = prefill_context_ids + prefill_context_ids

    prefill_input_ids = prefill_context_ids[:seq_len]
    output_strings = {}
    decode_inputs = []
    t_warm = time.time()
    for i in range(decode_batch_size):
        input = [(i, prefill_input_ids.copy())]
        pipeline.update(input)
        output_strings[i] = prefill_input_ids.copy()
        new_tokens = pipeline.run()
        for _, new_token in new_tokens:
            output_strings[i].extend(new_token)
        decode_inputs.extend(new_tokens)
    print(f"warmup prefill of {decode_batch_size} reqs took {time.time()-t_warm:.1f}s")

    # steady-state configuration: decode_batch_size decode reqs + 1 chunked prefill req
    output_strings[decode_batch_size] = prefill_context_ids[:prefill_batch_size].copy()
    decode_inputs.extend([(decode_batch_size, prefill_context_ids[:prefill_batch_size].copy())])
    pipeline.update(decode_inputs, decode_batch_size, use_cuda_graph=True, use_nano_split=True)

    cycle_times = []
    tokens_per_cycle = None
    for i in range(decode_batch_size, decode_batch_size + NUM_CYCLES):
        print("Cycle: ", i - decode_batch_size)
        t0 = time.time()
        next_prefill_idx = i + 1
        new_tokens = pipeline.run()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        new_tokens = new_tokens[:-1]
        decode_batchsize = len(new_tokens)
        assert decode_batchsize == decode_batch_size
        # retire the finished prefill request and free its KV pages (the demo
        # never frees pages, which exhausts a small pool within a few cycles)
        retired = pipeline.kv_cache.cache.pop(i, None)
        if retired is not None:
            retired.release()
        output_strings[next_prefill_idx] = prefill_context_ids[:prefill_batch_size].copy()
        new_tokens.extend([(next_prefill_idx, prefill_context_ids[:prefill_batch_size].copy())])
        pipeline.update(new_tokens, decode_batchsize, use_cuda_graph=True, use_nano_split=True)
        dt = time.time() - t0
        cycle_times.append(dt)
        if tokens_per_cycle is None:
            tokens_per_cycle = decode_batch_size + prefill_batch_size
        print(f"  cycle time {dt*1000:.1f} ms")

    # throughput accounting (paper metric: total tokens processed / s / GPU)
    steady = cycle_times[2:] if len(cycle_times) > 4 else cycle_times
    avg = sum(steady) / len(steady)
    print("=" * 60)
    print(f"config: global_batch={global_batch_size} decode_batch={decode_batch_size} "
          f"seq_len={seq_len} kv_pages={KV_PAGES} cycles={NUM_CYCLES}")
    print(f"avg steady-state cycle time: {avg*1000:.1f} ms")
    print(f"tokens per cycle (decode + prefill chunk): {tokens_per_cycle}")
    print(f"THROUGHPUT: {tokens_per_cycle/avg:,.0f} tokens/s/GPU")
    print("=" * 60)

    output_text = tokenizer.batch_decode(list(output_strings.values())[:1], skip_special_tokens=True)
    print(output_text[0][:500])


def test_correctness(use_kv_cache=True):
    input_string = "Hi, who are you?"
    input_ids = tokenizer.encode(input_string)
    special_inputs_0 = [(0, input_ids.copy()), (1, input_ids.copy())]
    special_inputs_1 = [(2, input_ids.copy()), (3, input_ids.copy())]
    output_strings = {}
    for idx, tensor in special_inputs_0 + special_inputs_1:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    assert len(new_tokens) == 2

    new_tokens.extend(special_inputs_1)
    pipeline.update(new_tokens, 2)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    assert len(new_tokens) == 4

    pipeline.update(new_tokens, 4)
    for i in range(20):
        print("Cycle: ", i)
        new_tokens = pipeline.run()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        assert len(new_tokens) == 4
        pipeline.update(new_tokens, 4)

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)


if args.correctness:
    test_correctness()
else:
    test_performance()
