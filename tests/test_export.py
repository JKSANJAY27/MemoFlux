"""
tests/test_export.py — ONNX export tests (gracefully skip if onnxruntime absent)
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from predictor.model  import ContextAwareLSTM
from predictor.export import export_to_onnx, benchmark_pytorch

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed — skipping ONNX tests")


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    return ContextAwareLSTM(vocab_size=20, n_users=5, hidden_dim=32, seq_len=6)


@pytest.fixture
def onnx_path(tiny_model, tmp_path):
    out = str(tmp_path / "test_model.onnx")
    export_to_onnx(tiny_model, seq_len=6, save_path=out)
    return out


# ── Tests ─────────────────────────────────────────────────

def test_onnx_export_produces_file(tiny_model, tmp_path):
    out = str(tmp_path / "model.onnx")
    export_to_onnx(tiny_model, seq_len=6, save_path=out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_onnx_load_via_onnxruntime(onnx_path):
    import onnxruntime as _ort
    sess = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert sess is not None


def test_onnx_output_shape(onnx_path):
    import onnxruntime as _ort
    import numpy as np
    sess = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = {
        "app_ids":  np.zeros((1, 6), dtype=np.int64),
        "ctx_vecs": np.zeros((1, 6, 12), dtype=np.float32),
        "user_ids": np.zeros((1,), dtype=np.int64),
    }
    out = sess.run(None, dummy)
    logits = out[0]
    assert logits.shape == (1, 20), f"Expected (1,20), got {logits.shape}"


def test_onnx_output_matches_pytorch(tiny_model, onnx_path):
    """ONNX and PyTorch logits must agree within 1e-3 tolerance."""
    import onnxruntime as _ort
    import numpy as np

    S = 6
    app_np  = np.zeros((1, S), dtype=np.int64)
    ctx_np  = np.random.randn(1, S, 12).astype(np.float32)
    uid_np  = np.zeros((1,), dtype=np.int64)

    # ONNX
    sess   = _ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {
        "app_ids": app_np, "ctx_vecs": ctx_np, "user_ids": uid_np,
    })[0]

    # PyTorch
    tiny_model.eval()
    with torch.no_grad():
        pt_logits, _ = tiny_model(
            torch.tensor(app_np),
            torch.tensor(ctx_np),
            torch.tensor(uid_np),
        )
    pt_np = pt_logits.numpy()

    diff = abs(onnx_logits - pt_np).max()
    assert diff < 1e-3, f"Max logit difference {diff:.6f} exceeds 1e-3"


def test_pytorch_benchmark_runs(tiny_model):
    """Benchmark should complete without error and return a dict."""
    result = benchmark_pytorch(tiny_model, seq_len=6, n_runs=20)
    assert "latency_p50_ms" in result
    assert result["latency_p50_ms"] > 0
