"""Smoke-tests for sizer/project_vla.py — Phase 3a (single-loop VLA projection).

Validates the invariants the Phase 3a scope was built around:
  - the 5090 (🟢 measured) reproduces the measured e2e / action rate
  - NPU projections (🔵 calibrated) are decode-BW-walled and single-digit Hz
    (NOT the reviewer's optimistic 12-18 Hz — that ballpark missed the
    autoregressive token-count × per-token-BW multiplier)
  - dual-loop architectures defer to Phase 3b
  - dtype/memory gates fire

Run: `python -m pytest tests/test_project_vla.py -q` or `python tests/test_project_vla.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ratchet import RTX_5090_REFERENCE, NPU_MID, NPU_HIGH, IMX95_MEASURED

from sizer.vla_models import VLA_MODELS
from sizer.project_vla import project_vla

NORA = VLA_MODELS["nora_3b"]
OPENVLA = VLA_MODELS["openvla_7b_single"]
BITVLA = VLA_MODELS["bitvla"]


def test_nora_on_5090_reproduces_measurement():
    r = project_vla(NORA, RTX_5090_REFERENCE)
    assert r["runs"] and r["vla_source"] == "measured"
    assert r["regime"] == "single_loop_measured"
    assert 79.0 <= r["ms_per_action"] <= 79.5      # measured e2e p50 79.22
    assert 12.0 <= r["action_hz"] <= 13.0          # ~12.6 Hz


def test_openvla_on_5090_reproduces_measurement():
    r = project_vla(OPENVLA, RTX_5090_REFERENCE)
    assert r["runs"] and r["vla_source"] == "measured"
    assert 126.0 <= r["ms_per_action"] <= 127.0    # measured e2e p50 126.50
    assert 7.5 <= r["action_hz"] <= 8.5            # ~7.9 Hz
    assert r["n_action_tokens"] == 7               # 7-DOF discrete (measured)


def test_nora_on_mid_is_bw_walled_single_digit_hz():
    """The headline finding: single-loop autoregressive decode is brutally
    BW-bound on edge memory. Mid is ~16× less BW than the 5090, so NORA lands
    in single-digit Hz — NOT the 12-18 Hz the review brief guessed."""
    r = project_vla(NORA, NPU_MID)
    assert r["runs"] and r["vla_source"] == "calibrated"
    assert r["dtype"] == "int8"                    # Mid is INT8-only (bf16=0)
    assert r["action_hz"] < 5.0                    # single-digit, BW-walled
    # decode term dominates the per-action time (the BW wall, not compute)
    decode_total = r["decode_ms_per_token"] * r["n_action_tokens"]
    assert decode_total > r["vlm_forward_ms"]


def test_mid_far_slower_than_5090_bw_wall():
    fast = project_vla(NORA, RTX_5090_REFERENCE)["action_hz"]
    slow = project_vla(NORA, NPU_MID)["action_hz"]
    assert slow < fast / 3                          # at least 3× slower (really ~7×)


def test_high_not_dramatically_faster_than_mid():
    """Mid and High share the same 94 GB/s effective BW; the workload is
    decode-BW-bound, so High's extra compute barely helps — the BW-wall story."""
    mid = project_vla(NORA, NPU_MID)["action_hz"]
    high = project_vla(NORA, NPU_HIGH)["action_hz"]
    assert high < 5.0
    assert abs(high - mid) / mid < 0.5             # within 50% — not a step change


def test_bitvla_is_cross_class_and_runs():
    r = project_vla(BITVLA, NPU_MID)
    assert r["runs"] and r["vla_source"] == "cross_class"
    assert r["regime"] == "single_loop_cross_class_roofline"
    assert r["action_hz"] > 0


def test_dual_loop_models_defer_to_phase3b():
    for key in ("nora_1p5", "pi_0p5", "openvla_7b_cached"):
        r = project_vla(VLA_MODELS[key], NPU_MID)
        assert r.get("deferred") is True
        assert "Phase 3b" in r["reason"]


def test_dtype_gate_bf16_on_int8_only_silicon():
    """Forcing bf16 on Mid (peak_tops_bf16=0) must not-run, not divide by zero."""
    r = project_vla(NORA, NPU_MID, dtype="bf16")
    assert r["runs"] is False
    assert "bf16" in r["reason"]


def test_npu_share_slows_bw_bound_decode():
    full = project_vla(NORA, NPU_MID, npu_share=1.0)["action_hz"]
    contended = project_vla(NORA, NPU_MID, npu_share=0.5)["action_hz"]
    assert contended < full                         # less BW → slower


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    # quick projection table for eyeballing
    print("\n--- NORA / OpenVLA / BitVLA across tiers ---")
    for vla in (NORA, OPENVLA, BITVLA):
        for hw in (RTX_5090_REFERENCE, NPU_HIGH, NPU_MID, IMX95_MEASURED):
            r = project_vla(vla, hw)
            if r.get("deferred"):
                continue
            if not r.get("runs"):
                print(f"{vla.key:18s} {hw.name:30s} won't run: {r['reason'][:40]}")
                continue
            print(f"{vla.key:18s} {hw.name:30s} {r['vla_source']:11s} "
                  f"{r['ms_per_action']:8.1f} ms  {r['action_hz']:6.2f} Hz")
    sys.exit(1 if failed else 0)
