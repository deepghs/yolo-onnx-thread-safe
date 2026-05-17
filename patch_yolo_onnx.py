#!/usr/bin/env python
"""patch_yolo_onnx.py — minimal, self-contained ONNX surgery to make any
ultralytics-exported YOLO ONNX safe under multi-threaded `CUDAExecutionProvider`.

Why this exists
---------------
`ultralytics` YOLO ONNX models (v3/v5/v8/v9/v10/v11/v12 — any size) crash or
hang when many threads share one `InferenceSession` and call `Run()` concurrently
on `onnxruntime-gpu`'s CUDA EP. The error looks like:

    CUDNN_STATUS_INTERNAL_ERROR at cudnnDestroy(cudnn_handle_)
    CUDA failure 700: illegal memory access

Bisecting the YOLO compute graph layer by layer pins the trigger to a single
operator: `Softmax`. ORT's CUDA EP dispatches `Softmax` to cuDNN, and cuDNN's
internal handle is shared across threads of one session — that handle is the
race window.

What this script does
---------------------
For every `Softmax(axis=a)` node in the graph it substitutes the mathematically
equivalent 5-op subgraph

    Y_max  = ReduceMax(X,    axes=[a], keepdims=1)
    Y_shft = Sub(X, Y_max)            # numerically stable: x - max(x)
    Y_exp  = Exp(Y_shft)
    Y_sum  = ReduceSum(Y_exp, axes=[a], keepdims=1)
    Y      = Div(Y_exp, Y_sum)        # exp / sum = softmax

The output tensor name `Y` is the same as the original `Softmax`'s output, so
the rest of the graph sees no change. ORT never calls cuDNN softmax for this
model anymore — the race window is gone.

Everything else is preserved untouched: producer / opset / model_version /
graph.name / every initializer / every non-Softmax node / every input and
output tensor (name + shape + dtype) / every existing entry in
`metadata_props`. We only ADD a small batch of new keys under the namespace
`yolo_onnx_softmax_safe.*` to mark the patch.

The patched ONNX is a true drop-in replacement: load it in ORT the same way,
no preprocessing / postprocessing changes needed. Measured impact on real
detection accuracy across 16 ult-pretrained models on COCO128 + 4 deepghs
production anime models on real validation sets: **at most −0.04 % mAP50-95**
(deepghs models bit-exact). Detail: TECHNICAL_REPORT.md §6.

Usage
-----
    python patch_yolo_onnx.py path/to/your_yolo.onnx

This writes `your_yolo_safe.onnx` next to the input. Add `--verify` to also
run both ONNX through a CPU EP correctness check on a single random tensor.

For advanced options (--in-place / --backup / --stress / 32w concurrent GPU
stress / batch processing across a directory) see `scripts/21_patch_softmax.py`.

Dependencies
------------
    pip install onnx numpy          # core
    pip install onnxruntime         # only needed for --verify
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

VERSION = "1.0"
META_NS = "yolo_onnx_softmax_safe"
REPO_URL = "https://github.com/deepghs/yolo-onnx-thread-safe"


# ----------------------------------------------------------------------------- helpers
def _opset_version(model: onnx.ModelProto) -> int:
    """Return the ai.onnx opset version of this model (default: 13)."""
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            return o.version
    return 13


def _is_already_patched(model: onnx.ModelProto) -> bool:
    """Idempotency check: has this ONNX already been patched once?"""
    for kv in model.metadata_props:
        if kv.key == f"{META_NS}.patched" and kv.value in ("1", "true", "True"):
            return True
    return False


def _set_meta(model: onnx.ModelProto, key: str, value: str) -> None:
    """Set (or update) one entry in metadata_props under our namespace.
    Preserves all other entries unchanged."""
    full = f"{META_NS}.{key}"
    for kv in model.metadata_props:
        if kv.key == full:
            kv.value = value
            return
    model.metadata_props.add(key=full, value=value)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- core: one Softmax → 5 ops
def _replacement_subgraph(softmax_node: onnx.NodeProto, opset: int):
    """Build the five replacement nodes + the one axes initializer that take
    over from the original Softmax. Returns (nodes, initializers, info_dict)."""
    if softmax_node.op_type != "Softmax":
        raise ValueError(f"expected Softmax, got {softmax_node.op_type}")
    if len(softmax_node.input) != 1 or len(softmax_node.output) != 1:
        raise ValueError(
            f"unexpected Softmax I/O arity: in={list(softmax_node.input)}, "
            f"out={list(softmax_node.output)}")

    # Read the Softmax's input tensor name (X) and output tensor name (Y) —
    # we'll keep Y intact so the rest of the graph sees the same wiring.
    X = softmax_node.input[0]
    Y = softmax_node.output[0]

    # Read the axis attribute. ONNX Softmax default since opset 13 is axis=-1.
    axis = -1
    for attr in softmax_node.attribute:
        if attr.name == "axis":
            axis = attr.i
            break

    # Naming: derive a unique prefix from the original node name so multiple
    # Softmaxes in the same model don't collide.
    base = (softmax_node.name or f"softmax_{id(softmax_node)}").replace("/", "_").strip("_")
    prefix = f"{base}__safe"

    # Intermediate tensor names
    n_max   = f"{prefix}_max"      # = max(X) along axis
    n_shift = f"{prefix}_shift"    # = X - max(X)   (numerical stabilization)
    n_exp   = f"{prefix}_exp"      # = exp(X - max)
    n_sum   = f"{prefix}_sum"      # = sum(exp(...))
    axes_const_name = f"{prefix}_axes"

    # ReduceMax / ReduceSum need their reduction axes specified. The way that
    # is done changed across ONNX opsets:
    #   - opset >= 18: axes is an INPUT (typed int64 initializer)
    #   - opset 13-17: ReduceSum axes is input, ReduceMax axes is ATTRIBUTE
    #   - opset < 13:  both are attributes
    # We support both forms.
    axes_init = numpy_helper.from_array(
        np.array([axis], dtype=np.int64), name=axes_const_name)
    initializers = [axes_init]
    nodes = []

    if opset >= 18:
        # axes-as-input for ReduceMax (opset >= 18)
        nodes.append(helper.make_node(
            "ReduceMax", [X, axes_const_name], [n_max],
            name=f"{prefix}_ReduceMax", keepdims=1))
    else:
        # axes-as-attribute (opset <= 17)
        nodes.append(helper.make_node(
            "ReduceMax", [X], [n_max],
            name=f"{prefix}_ReduceMax", axes=[axis], keepdims=1))

    # Sub, Exp are unchanged across opsets.
    nodes.append(helper.make_node("Sub", [X, n_max], [n_shift],
                                  name=f"{prefix}_Sub"))
    nodes.append(helper.make_node("Exp", [n_shift], [n_exp],
                                  name=f"{prefix}_Exp"))

    if opset >= 13:
        # ReduceSum has axes-as-input from opset 13 onwards
        nodes.append(helper.make_node(
            "ReduceSum", [n_exp, axes_const_name], [n_sum],
            name=f"{prefix}_ReduceSum", keepdims=1))
    else:
        nodes.append(helper.make_node(
            "ReduceSum", [n_exp], [n_sum],
            name=f"{prefix}_ReduceSum", axes=[axis], keepdims=1))

    # Final Div: Y = exp / sum   (this is what softmax is)
    nodes.append(helper.make_node("Div", [n_exp, n_sum], [Y],
                                  name=f"{prefix}_Div"))

    info = dict(name=softmax_node.name, axis=axis, input_tensor=X, output_tensor=Y)
    return nodes, initializers, info


# ----------------------------------------------------------------------------- main pipeline
def patch(in_path: Path, out_path: Path, verbose: bool = True) -> dict:
    """Load ONNX, replace every Softmax with the manual decomposition, save.
    Returns a dict describing what happened (for logging / tests)."""
    log = (lambda *a, **kw: print(*a, **kw, flush=True)) if verbose else (lambda *a, **kw: None)

    # ──── 1. Load and inspect ────
    log(f"[1/5] Loading {in_path}")
    model = onnx.load(str(in_path), load_external_data=False)
    opset = _opset_version(model)
    inputs  = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
               for i in model.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
               for o in model.graph.output]
    log(f"      producer:   {model.producer_name!r} {model.producer_version!r}")
    log(f"      ai.onnx opset: {opset}")
    log(f"      input:      {inputs}")
    log(f"      output:     {outputs}")
    log(f"      existing metadata_props: {len(model.metadata_props)} entries")
    if _is_already_patched(model):
        log("      ⚠ already patched — nothing to do (idempotent no-op).")
        return {"status": "already_patched", "input": str(in_path)}

    # ──── 2. Locate Softmax nodes ────
    log(f"[2/5] Scanning graph for Softmax nodes")
    sm_idxs = [i for i, n in enumerate(model.graph.node) if n.op_type == "Softmax"]
    if not sm_idxs:
        log("      no Softmax nodes — model is already race-free.")
        return {"status": "no_softmax_found", "input": str(in_path)}
    log(f"      found {len(sm_idxs)} Softmax node(s):")
    for idx in sm_idxs:
        n = model.graph.node[idx]
        axis = next((a.i for a in n.attribute if a.name == "axis"), -1)
        log(f"        • {n.name!r}  axis={axis}  in={list(n.input)}  out={list(n.output)}")

    # ──── 3. Replace each Softmax with the 5-op subgraph ────
    #    Process right-to-left so node-index references stay valid as we mutate.
    log(f"[3/5] Replacing each Softmax with ReduceMax + Sub + Exp + ReduceSum + Div")
    log(f"      (mathematically equivalent: softmax(x) = exp(x-max(x)) / sum(exp(x-max(x))))")
    patches = []
    for idx in sorted(sm_idxs, reverse=True):
        old = model.graph.node[idx]
        new_nodes, new_inits, info = _replacement_subgraph(old, opset)
        patches.append(info)
        del model.graph.node[idx]
        for offset, nn in enumerate(new_nodes):
            model.graph.node.insert(idx + offset, nn)
        for nn_init in new_inits:
            model.graph.initializer.append(nn_init)
        log(f"      ✓ replaced {old.name!r} → 5 new ops, 1 axes initializer")
    patches.reverse()  # restore original order in our log

    # ──── 4. Add patch metadata (preserve all existing entries) ────
    log(f"[4/5] Adding patch metadata to metadata_props (preserving original entries)")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    in_sha = _sha256(in_path)
    _set_meta(model, "patched", "1")
    _set_meta(model, "version", VERSION)
    _set_meta(model, "tool", "patch_yolo_onnx.py")
    _set_meta(model, "repo", REPO_URL)
    _set_meta(model, "patched_at_utc", now_iso)
    _set_meta(model, "input_sha256", in_sha)
    _set_meta(model, "n_softmax_replaced", str(len(patches)))
    _set_meta(model, "softmax_axes",
              json.dumps([p["axis"] for p in patches]))
    _set_meta(model, "softmax_node_names",
              json.dumps([p["name"] for p in patches]))
    _set_meta(model, "decomposition",
              "ReduceMax->Sub->Exp->ReduceSum->Div")
    _set_meta(model, "note",
              "Drop-in race-free replacement for ORT CUDA EP. "
              "I/O signature unchanged.")
    for kv in model.metadata_props:
        if kv.key.startswith(META_NS):
            v = kv.value[:60] + "…" if len(kv.value) > 60 else kv.value
            log(f"      + {kv.key} = {v}")

    # ──── 5. Validate + save ────
    log(f"[5/5] Validating + saving to {out_path}")
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    out_sha = _sha256(out_path)
    size_mb = out_path.stat().st_size / 1e6
    log(f"      size:    {size_mb:.2f} MB")
    log(f"      sha256:  {out_sha[:16]}…")
    log()
    log("✓ Done. Drop-in usage:")
    log(f"      import onnxruntime as ort")
    log(f"      sess = ort.InferenceSession({str(out_path)!r},")
    log(f"          providers=[('CUDAExecutionProvider', {{'device_id': 0}}),")
    log(f"                     'CPUExecutionProvider'])")
    log(f"      # share `sess` across threads with NO lock — race is gone.")

    return {
        "status": "ok",
        "input": str(in_path),
        "output": str(out_path),
        "input_sha256": in_sha,
        "output_sha256": out_sha,
        "opset": opset,
        "softmax_replaced": len(patches),
        "patches": patches,
    }


# ----------------------------------------------------------------------------- optional: verify on CPU EP
def verify(orig_path: Path, patched_path: Path, *, n_trials: int = 3) -> dict:
    """Run BOTH ONNX on the same random fp32 input via CPU EP and report
    the worst element-wise |Δ|. This catches any logic error in the patch.
    On CPU EP the race is irrelevant — we just compare correctness."""
    print()
    print(f"[verify] Running both ONNX on CPU EP, comparing outputs on {n_trials} random inputs")
    try:
        import onnxruntime as ort
    except ImportError:
        print("      ⚠ onnxruntime not installed; skipping verification.")
        print("        $ pip install onnxruntime")
        return {"status": "ort_missing"}

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sa = ort.InferenceSession(str(orig_path),    sess_options=so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(patched_path), sess_options=so, providers=["CPUExecutionProvider"])
    in_a, out_a = sa.get_inputs()[0], sa.get_outputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in in_a.shape]
    # Most YOLOs accept (1, 3, 640, 640) fp32. Use that if dynamic.
    if shape == [1] * len(shape):
        shape = [1, 3, 640, 640]
    print(f"        input shape used: {shape}")

    rng = np.random.default_rng(1337)
    worst = 0.0
    for t in range(n_trials):
        x = rng.standard_normal(shape).astype(np.float32) * 0.1 + 0.5
        ya = sa.run([out_a.name], {in_a.name: x})[0]
        yb = sb.run([sb.get_outputs()[0].name], {sb.get_inputs()[0].name: x})[0]
        if ya.shape != yb.shape:
            print(f"        ⚠ trial {t}: SHAPE MISMATCH {ya.shape} vs {yb.shape}")
            return {"status": "shape_mismatch", "orig": ya.shape, "patched": yb.shape}
        d = float(np.max(np.abs(ya - yb)))
        worst = max(worst, d)
        print(f"        trial {t+1}: max|Δ| = {d:.3e}")
    print()
    if worst <= 5e-4:
        print(f"✓ verify OK  worst max|Δ| = {worst:.3e}  (≤ 5e-4 fp32 noise band)")
    elif worst <= 1e-2:
        print(f"⚠ verify OK with caveat: worst max|Δ| = {worst:.3e}")
        print("    Some models with attention softmax (yolo11/v12) or top-K head (v10n)")
        print("    show element-wise diff above 5e-4 due to fp32 reduction-order noise.")
        print("    Real mAP impact ≤ 0.04% — see TECHNICAL_REPORT.md §6.4 for measurements.")
    else:
        print(f"✗ verify FAILED: worst max|Δ| = {worst:.3e} (unexpectedly large)")
        return {"status": "fail", "worst_max_abs": worst}
    return {"status": "ok", "worst_max_abs": worst, "n_trials": n_trials}


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(
        description="Minimal universal patcher: makes any ult-style YOLO ONNX "
                    "thread-safe under ORT CUDA EP. See header for full rationale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n"
               "  python patch_yolo_onnx.py models/yolov8n.onnx --verify\n"
               "  python patch_yolo_onnx.py models/yolo12n.onnx out/yolo12n_safe.onnx\n"
               "\nFor advanced options (--in-place, --stress, batch mode) see\n"
               "scripts/21_patch_softmax.py.\n")
    ap.add_argument("input",  help="path to source ONNX (ult-style or any with Softmax)")
    ap.add_argument("output", nargs="?", default=None,
                    help="output path (default: <input>_safe.onnx)")
    ap.add_argument("--verify", action="store_true",
                    help="after patching, run both ONNX on CPU EP and compare outputs")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"ERR: {in_path} does not exist", file=sys.stderr)
        sys.exit(2)
    out_path = (Path(args.output).resolve() if args.output
                else in_path.with_name(in_path.stem + "_safe.onnx"))

    info = patch(in_path, out_path)
    if info["status"] in ("already_patched", "no_softmax_found"):
        return 0
    if args.verify:
        v = verify(in_path, out_path)
        if v.get("status") not in ("ok", "ort_missing"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
