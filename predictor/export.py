"""
predictor/export.py — ONNX Export + Latency Benchmark

Pipeline:
  PyTorch (.pt) → ONNX FP32 → Dynamic INT8 quantized ONNX → p50/p95/p99 benchmark

Target: p50 < 15ms on single-core CPU (simulates on-device Samsung Galaxy inference).
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────

def export_to_onnx(
    model,
    seq_len:   int,
    save_path: str = "exports/ax_predictor.onnx",
    opset:     int = 17,
) -> str:
    """Export ContextAwareLSTM to ONNX FP32."""
    import torch.onnx

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.cpu()

    dummy_app  = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_ctx  = torch.zeros(1, seq_len, 12, dtype=torch.float32)
    dummy_uid  = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_app, dummy_ctx, dummy_uid),
        save_path,
        opset_version      = opset,
        input_names        = ["app_ids", "ctx_vecs", "user_ids"],
        output_names       = ["logits"],
        dynamic_axes       = {
            "app_ids":  {0: "batch"},
            "ctx_vecs": {0: "batch"},
            "user_ids": {0: "batch"},
            "logits":   {0: "batch"},
        },
        do_constant_folding = True,
        verbose             = False,
    )
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"ONNX (FP32) saved → {save_path}  ({size_mb:.1f} MB)")
    return save_path


def quantize_onnx(
    onnx_path:   str,
    output_path: str = "exports/ax_predictor_int8.onnx",
) -> str:
    """Dynamic INT8 quantization via onnxruntime-tools."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input  = onnx_path,
            model_output = output_path,
            weight_type  = QuantType.QInt8,
        )
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"ONNX (INT8) saved → {output_path}  ({size_mb:.1f} MB)")
        return output_path
    except Exception as e:
        print(f"[WARN] INT8 quantization failed ({e}). Skipping.")
        return onnx_path


# ─────────────────────────────────────────────────────────
# Latency Benchmark
# ─────────────────────────────────────────────────────────

def benchmark_onnx(
    onnx_path: str,
    seq_len:   int,
    n_warmup:  int = 50,
    n_runs:    int = 500,
) -> Dict:
    """
    Measure p50/p95/p99 inference latency (CPU, batch=1).
    Simulates single-core inference on a Samsung Galaxy device.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # single core
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        onnx_path, sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    dummy = {
        "app_ids":  np.zeros((1, seq_len), dtype=np.int64),
        "ctx_vecs": np.zeros((1, seq_len, 12), dtype=np.float32),
        "user_ids": np.zeros((1,), dtype=np.int64),
    }

    for _ in range(n_warmup):
        sess.run(None, dummy)

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, dummy)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    p50     = float(np.percentile(latencies_ms, 50))

    result = {
        "model_path":          onnx_path,
        "n_runs":              n_runs,
        "latency_p50_ms":      round(p50, 2),
        "latency_p95_ms":      round(float(np.percentile(latencies_ms, 95)), 2),
        "latency_p99_ms":      round(float(np.percentile(latencies_ms, 99)), 2),
        "latency_mean_ms":     round(float(np.mean(latencies_ms)), 2),
        "model_size_mb":       round(size_mb, 2),
        "passes_15ms_target":  p50 < 15.0,
    }

    print(f"\n=== ONNX Benchmark: {Path(onnx_path).name} ===")
    print(f"  p50={result['latency_p50_ms']}ms  "
          f"p95={result['latency_p95_ms']}ms  "
          f"size={result['model_size_mb']}MB  "
          f"{'✓ PASS' if result['passes_15ms_target'] else '✗ FAIL'} (< 15ms)")

    return result


def run_full_export_pipeline(
    model,
    seq_len:        int,
    checkpoint_dir: str = "checkpoints",
    export_dir:     str = "exports",
) -> Dict:
    """
    Full pipeline: export FP32 → INT8 → benchmark both.
    Saves exports/benchmark_results.json.
    """
    fp32_path = f"{export_dir}/ax_predictor.onnx"
    int8_path = f"{export_dir}/ax_predictor_int8.onnx"

    export_to_onnx(model, seq_len, fp32_path)

    results = {}

    # FP32 benchmark
    try:
        results["fp32"] = benchmark_onnx(fp32_path, seq_len)
    except Exception as e:
        print(f"[WARN] FP32 benchmark failed: {e}")
        results["fp32"] = {"error": str(e)}

    # INT8 quantize + benchmark
    try:
        int8_out = quantize_onnx(fp32_path, int8_path)
        results["int8"] = benchmark_onnx(int8_out, seq_len)
    except Exception as e:
        print(f"[WARN] INT8 benchmark failed: {e}")
        results["int8"] = {"error": str(e)}

    out_path = f"{export_dir}/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results → {out_path}")

    return results


# ─────────────────────────────────────────────────────────
# PyTorch latency benchmark (no ONNX required)
# ─────────────────────────────────────────────────────────

def benchmark_pytorch(model, seq_len: int, n_runs: int = 200) -> Dict:
    """Fallback benchmark using raw PyTorch (CPU)."""
    model.eval().cpu()
    dummy_app = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_ctx = torch.zeros(1, seq_len, 12)
    dummy_uid = torch.zeros(1, dtype=torch.long)

    latencies = []
    with torch.no_grad():
        for _ in range(20):   # warmup
            model(dummy_app, dummy_ctx, dummy_uid)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_app, dummy_ctx, dummy_uid)
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = float(np.percentile(latencies, 50))
    result = {
        "backend": "pytorch_cpu",
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "passes_15ms_target": p50 < 15.0,
    }
    print(f"PyTorch CPU benchmark: p50={result['latency_p50_ms']}ms  "
          f"{'✓' if result['passes_15ms_target'] else '✗'}")
    return result
