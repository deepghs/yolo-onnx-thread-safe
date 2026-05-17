#!/usr/bin/env python
"""stress_test.py — minimal, self-contained reproducer for the ORT CUDA EP
multi-threaded inference race + a measurement harness for the patched fix.

What this script does
---------------------
Given an ONNX file, spin up N worker threads, share a single
`InferenceSession` across all of them with **no lock**, and have every thread
call `Run()` in a loop. Print whether anyone crashed, how long it took, and
what throughput we got.

How to read the output
----------------------
• Run on an unpatched ult-style YOLO ONNX with --workers 32 --rounds 100:
    Expect: a hang (CUDA deadlock) — kill with timeout, OR an immediate crash
    with `CUDA failure 700: illegal memory access` / `CUDNN_STATUS_INTERNAL_ERROR`.
    Both are the bug.

• Run the same load on the corresponding ONNX patched by `patch_yolo_onnx.py`:
    Expect: ok=32 fail=0, wall ≈ a few seconds, throughput ≈ hundreds of imgs/s.

So **the difference between the two output blobs IS the value of this repo**.

Why this script is in this repo
-------------------------------
The patched ONNX (produced by `patch_yolo_onnx.py`) is meant to be a drop-in
replacement that allows concurrent multi-threaded inference WITHOUT external
locks. The only way to *verify* that claim is to actually try concurrent
inference. This is the minimum harness for that.

Dependencies: `onnxruntime` (or `onnxruntime-gpu`), `numpy`. No other repo
files are needed; this script is fully self-contained.

CLI
---
    # default: 32 worker threads, 100 rounds each, CUDA EP
    python stress_test.py path/to/model.onnx

    # heavier:
    python stress_test.py model.onnx --workers 64 --rounds 200

    # CPU EP (race is not present here; useful for correctness sanity)
    python stress_test.py model.onnx --ep cpu --workers 4 --rounds 50

    # custom input shape (default is (1, 3, 640, 640) — YOLO standard)
    python stress_test.py model.onnx --shape 1 3 320 320
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def _make_session(model_path: str, ep: str):
    """Build an InferenceSession that *insists* on the requested EP.
    Fail fast if CUDA was asked for but isn't available — otherwise the
    crash test would falsely 'pass' by silently running on CPU."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # only ERROR + FATAL on stdout — we want signal
    providers = []
    if ep == "cuda":
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                f"CUDAExecutionProvider unavailable; got {avail}. "
                "Did you install onnxruntime-gpu and configure LD_LIBRARY_PATH?")
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    used = sess.get_providers()
    if ep == "cuda" and used[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA requested but using {used} — silent CPU fallback!")
    return sess


def _worker(wid: int, sess, input_name: str, n_rounds: int,
            inputs: list[np.ndarray], results: list, errors: list):
    """One worker thread: loop n_rounds times calling sess.run()."""
    try:
        rng = np.random.default_rng(1337 + wid)
        for _ in range(n_rounds):
            x = inputs[rng.integers(len(inputs))]
            sess.run(None, {input_name: x})
        results[wid] = "ok"
    except BaseException as e:
        errors[wid] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
        results[wid] = "fail"


def main():
    ap = argparse.ArgumentParser(
        description="Minimal concurrent-inference stress test / throughput "
                    "harness for ORT CUDA EP. See header for details.")
    ap.add_argument("model", help="path to .onnx (patched or unpatched)")
    ap.add_argument("--workers", type=int, default=32,
                    help="number of worker threads sharing the session (default 32)")
    ap.add_argument("--rounds", type=int, default=100,
                    help="per-worker number of sess.run() calls (default 100)")
    ap.add_argument("--ep", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--shape", type=int, nargs="+", default=[1, 3, 640, 640],
                    help="model input shape (default: 1 3 640 640, YOLO standard)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup calls before launching workers")
    ap.add_argument("--n-inputs", type=int, default=8,
                    help="how many random input tensors to pre-build and cycle through")
    args = ap.parse_args()

    print(f"[stress_test] model:   {args.model}")
    print(f"[stress_test] ep:      {args.ep}")
    print(f"[stress_test] workers: {args.workers}")
    print(f"[stress_test] rounds:  {args.rounds}  → total inferences = {args.workers * args.rounds}")
    print(f"[stress_test] shape:   {tuple(args.shape)}")

    sess = _make_session(args.model, args.ep)
    input_name = sess.get_inputs()[0].name

    # Pre-build a handful of random inputs we'll cycle through.
    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal(args.shape).astype(np.float32) * 0.1 + 0.5
              for _ in range(args.n_inputs)]

    # Warmup so cuDNN / TRT autotuning is done before we time anything.
    for _ in range(args.warmup):
        sess.run(None, {input_name: inputs[0]})
    print(f"[stress_test] warmup done; launching {args.workers} worker threads with NO lock…")

    results = ["?"] * args.workers
    errors  = [""]  * args.workers

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, i, sess, input_name, args.rounds, inputs,
                          results, errors) for i in range(args.workers)]
        for _ in as_completed(futs):
            pass
    dt = time.perf_counter() - t0

    n_ok   = sum(1 for r in results if r == "ok")
    n_fail = sum(1 for r in results if r == "fail")
    rate = (args.workers * args.rounds) / dt if (n_fail == 0 and dt > 0) else 0.0

    print()
    print("=== RESULTS ===")
    print(f"  workers ok:       {n_ok}/{args.workers}")
    print(f"  workers failed:   {n_fail}")
    print(f"  wall elapsed:     {dt:.2f} s")
    if n_fail == 0:
        print(f"  throughput:       {rate:.1f} imgs/sec")
        print(f"  status:           ✓ PASS — concurrent run() is safe on this ONNX")
    else:
        first_err = next((e for e in errors if e), "")
        print(f"  status:           ✗ FAIL — race triggered")
        print(f"  first error tail:")
        for line in first_err.splitlines()[-6:]:
            print(f"    {line}")
        print()
        print("  This is exactly the bug. Patch this ONNX with:")
        print(f"      python patch_yolo_onnx.py {args.model}")
        print(f"  and re-run to see ok={args.workers}/{args.workers}.")

    # Machine-readable summary on the last line (for CI / scripts):
    print(json.dumps({
        "model": args.model, "ep": args.ep,
        "workers": args.workers, "rounds": args.rounds,
        "ok": n_ok, "fail": n_fail,
        "elapsed_s": round(dt, 2),
        "throughput_imgs_per_s": round(rate, 1) if n_fail == 0 else None,
        "passed": n_fail == 0,
    }, ensure_ascii=False))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
