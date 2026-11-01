#!/usr/bin/env python3
"""One request through the disaggregated path, with a hard timeout.

Never probe this stack with a bare `curl`. The consumer's wait for KV had no timeout
upstream, so a mis-keyed transfer hangs the client as long as the engine -- which is to
say forever. Everything here carries a deadline.

    scripts/decode/disagg_probe.py                # one short request via the router
    scripts/decode/disagg_probe.py --isl 6144     # the real workload's prompt length
    scripts/decode/disagg_probe.py --leg prefill  # one leg only, to isolate a failure
    scripts/decode/disagg_probe.py --needle        # check the KV carries real content
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"


def post(url, body, timeout, headers=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - start, r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return time.time() - start, e.code, {"error": e.read().decode()[:500]}
    except TimeoutError:
        return time.time() - start, None, {"error": f"TIMEOUT after {timeout}s"}
    except Exception as e:  # connection refused, reset, ...
        return time.time() - start, None, {"error": f"{type(e).__name__}: {e}"}


def make_request_id(prefill_kv, decode_kv):
    """Mirrors disagg_p2p_proxy.make_request_id; the id is the connector's routing table."""
    return (
        f"___prefill_addr_{prefill_kv}___decode_addr_{decode_kv}_{uuid.uuid4().hex}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--leg", choices=["proxy", "prefill", "decode"], default="proxy",
                    help="proxy exercises the whole path; the others hit one instance "
                         "directly, still carrying a connector-shaped request id")
    ap.add_argument("--isl", type=int, default=64, help="prompt length in tokens (approx)")
    ap.add_argument("--osl", type=int, default=16)
    ap.add_argument("--needle", action="store_true",
                    help="hide a code at the very start of the prompt and ask for it back "
                         "at the end. A coherent-looking continuation only proves the "
                         "decode node has *some* KV; recalling the needle proves it has "
                         "the prefill node's, from the far end of the context.")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--prefill-kv", default=None)
    ap.add_argument("--decode-kv", default=None,
                    help="both default to whatever the router reports on /health, which "
                         "is what the engines actually bound -- not localhost")
    args = ap.parse_args()

    if args.leg != "proxy" and not (args.prefill_kv and args.decode_kv):
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as r:
            kv = json.loads(r.read())
        args.prefill_kv = args.prefill_kv or kv["prefill_kv"]
        args.decode_kv = args.decode_kv or kv["decode_kv"]

    # "word" repeated is ~1 token each, close enough to set a prompt length.
    filler = "The quick brown fox jumps over the lazy dog. " * max(1, args.isl // 9)
    needle = "4271"
    if args.needle:
        prompt = (f"Remember this: the access code is {needle}.\n\n" + filler
                  + "\n\nQuestion: what is the access code?\nAnswer: The access code is")
    else:
        prompt = filler

    body = {"model": MODEL, "prompt": prompt, "max_tokens": args.osl,
            "temperature": 0.0, "stream": False}
    headers = {}
    url = args.url
    if args.leg == "prefill":
        url, body["max_tokens"] = "http://127.0.0.1:8100", 1
    elif args.leg == "decode":
        url = "http://127.0.0.1:8200"
    if args.leg != "proxy":
        rid = make_request_id(args.prefill_kv, args.decode_kv)
        headers["X-Request-Id"] = rid
        print(f"request id: {rid}")

    print(f"POST {url}/v1/completions  leg={args.leg} isl~{args.isl} osl={body['max_tokens']} "
          f"timeout={args.timeout}s", flush=True)
    elapsed, status, payload = post(f"{url}/v1/completions", body, args.timeout, headers)

    print(f"\n{elapsed:8.2f}s  status={status}")
    if status == 200:
        choice = payload["choices"][0]
        print(f"  finish_reason={choice.get('finish_reason')}  usage={payload.get('usage')}")
        print(f"  text: {choice['text']!r}")
        if args.needle:
            found = needle in choice["text"]
            print(f"  needle {needle!r}: {'RECALLED' if found else 'LOST -- the decode node '
                  'is not decoding against the prefill node KV'}")
            sys.exit(0 if found else 1)
        sys.exit(0)
    print(f"  {payload}")
    sys.exit(1)


if __name__ == "__main__":
    main()
