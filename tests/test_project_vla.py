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


def test_bitvla_on_5090_reproduces_oft_measurement():
    """BitVLA is OFT parallel-chunk (NOT single-loop AR): one prefill-shaped
    forward → 8 actions in parallel, amortized to 65 Hz."""
    r = project_vla(BITVLA, RTX_5090_REFERENCE)
    assert r["runs"] and r["vla_source"] == "measured"
    assert r["regime"] == "oft_parallel_chunk_measured"
    assert r["architecture"] == "oft_parallel_chunk"
    assert 15.0 <= r["ms_per_action"] <= 15.8       # forward 123 / 8 = 15.4
    assert 64.0 <= r["action_hz"] <= 66.0           # 65 Hz
    assert r["action_chunk_length"] == 8
    assert "decode_ms_per_token" not in r           # no AR decode term in OFT


def test_bitvla_oft_runs_on_int8_only_mid_no_fp_gate():
    """BitVLA is the only int_only entry: no FP-required head, so it RUNS on
    INT8-only Mid (where both dual-loop FP heads hit the hard gate)."""
    r = project_vla(BITVLA, NPU_MID)
    assert r["runs"] and r["vla_source"] == "calibrated"
    assert r["regime"] == "oft_parallel_chunk_calibrated_latency_scaled"
    assert r["dtype"] == "int8"
    assert r["action_hz"] > 0


def test_bitvla_oft_not_bw_walled_unlike_single_loop_ar():
    """The OFT speed story: a prefill-shaped parallel forward avoids the AR
    decode BW-wall, so BitVLA on Mid runs an order of magnitude faster than the
    single-loop AR models that get BW-walled to <1 Hz there."""
    bitvla_mid = project_vla(BITVLA, NPU_MID)["action_hz"]
    openvla_mid = project_vla(OPENVLA, NPU_MID)["action_hz"]   # single-loop AR, BW-walled
    assert bitvla_mid > openvla_mid * 10


def test_uncalibrated_dual_loop_still_defers():
    """Dual-loop models WITHOUT a measured topology anchor still defer — the
    projection needs measured_5090_dual_loop, not just the architecture flag.
    (openvla_7b_cached is the synthetic projection-only entry; NORA-1.5 and π0.5
    are now both calibrated.)"""
    r = project_vla(VLA_MODELS["openvla_7b_cached"], NPU_HIGH)
    assert r.get("deferred") is True
    assert "dual-loop anchor" in r["reason"]


# ── Phase 3b: NORA-1.5 dual-loop (flow-matching action expert) ──────────────

NORA15 = VLA_MODELS["nora_1p5"]


def test_nora15_on_5090_reproduces_dual_loop_measurement():
    r = project_vla(NORA15, RTX_5090_REFERENCE)
    assert r["runs"] and r["vla_source"] == "measured"
    assert r["regime"] == "dual_loop_measured"
    assert 36.0 <= r["ms_per_action"] <= 37.0       # amortized 36.55
    assert 27.0 <= r["action_hz"] <= 28.0           # 27.4 Hz
    assert abs(r["fast_loop_only_hz"] - 32.0) < 0.5  # published ~40 Hz regime
    assert r["num_denoise_steps"] == 10
    assert r["action_chunk_length"] == 5


def test_nora15_fp_required_head_is_hard_gate_on_int8_only_mid():
    """The FP-required flow-matching head eliminates INT8-only silicon entirely.
    NPU Mid has fp8=bf16=0, so NORA-1.5 cannot run — the QuantVLA hard gate."""
    r = project_vla(NORA15, NPU_MID)
    assert r["runs"] is False
    assert "FP" in r["reason"] and "hard gate" in r["reason"]


def test_nora15_runs_on_high_via_fp8_expert():
    r = project_vla(NORA15, NPU_HIGH)
    assert r["runs"] and r["vla_source"] == "calibrated"
    assert r["expert_dtype"] == "fp8"               # High has fp8=400
    assert r["vlm_dtype"] == "int8"                 # VLM stages int8
    assert r["action_hz"] > 0


def test_nora15_launch_bound_step_barely_degrades_on_edge():
    """NORA-1.5's denoise is PURE launch-bound (1.6% BW), so the launch+BW
    decomposition leaves the edge step ≈ the measured 5090 step — NOT the naive
    full-BW-scaled value (≈ ref × 16 ≈ 250 ms/step nonsense)."""
    ref = project_vla(NORA15, RTX_5090_REFERENCE)
    high = project_vla(NORA15, NPU_HIGH)
    assert high["denoise_projection"] == "eager_launch_plus_bw_decomp"
    # launch_const dominates → within ~50% of the measured step, nowhere near 16×
    assert high["denoise_step_ms"] < ref["denoise_step_ms"] * 1.6
    assert high["denoise_step_ms"] >= ref["denoise_step_ms"]   # small BW penalty, never faster


def test_nora15_optimized_floor_beats_eager_ceiling():
    """Headroom is real: the compiled/fused physics floor is faster than the
    launch-bound eager ceiling, and the headline uses the conservative eager."""
    high = project_vla(NORA15, NPU_HIGH)
    assert high["action_hz_optimized_floor"] > high["action_hz"]
    assert high["denoise_step_ms_optimized_floor"] < high["denoise_step_ms"]


def test_nora15_dual_loop_not_a_hard_bw_wall_unlike_single_loop():
    """Contrast with the single-loop story: NORA-3B is BW-walled to <5 Hz on
    edge; NORA-1.5's dual-loop on High clears that comfortably (the fast loop
    has headroom, the slow VLM amortizes over the chunk)."""
    nora3b_high = project_vla(NORA, NPU_HIGH)["action_hz"]
    nora15_high = project_vla(NORA15, NPU_HIGH)["action_hz"]
    assert nora15_high > nora3b_high


# ── Phase 3b: π0.5 dual-loop (the amortization extreme + partial-BW fast loop) ──

PI05 = VLA_MODELS["pi_0p5"]


def test_pi05_on_5090_is_the_amortization_extreme():
    """One VLM forward amortized over a 50-action chunk → ~367 Hz on the 5090,
    an order of magnitude past NORA-1.5's chunk-of-5 → 27 Hz."""
    r = project_vla(PI05, RTX_5090_REFERENCE)
    assert r["runs"] and r["vla_source"] == "measured"
    assert r["action_chunk_length"] == 50
    assert 360.0 <= r["action_hz"] <= 375.0         # 367 Hz amortized
    assert r["action_hz"] > project_vla(NORA15, RTX_5090_REFERENCE)["action_hz"] * 5


def test_pi05_fp_required_head_hard_gate_on_mid():
    r = project_vla(PI05, NPU_MID)
    assert r["runs"] is False
    assert "FP" in r["reason"] and "hard gate" in r["reason"]


def test_pi05_partial_bw_step_degrades_more_than_nora15_on_edge():
    """The data-driven distinction: π0.5's denoise is partial-BW (13.4%), so its
    step degrades a LARGER multiple on a low-BW edge part than NORA-1.5's
    near-pure-launch (1.6%) step. Branching is implicit in the decomposition —
    the per-model effective-BW drives it, no label string-match."""
    n15_ref = project_vla(NORA15, RTX_5090_REFERENCE)["denoise_step_ms"]
    n15_hi = project_vla(NORA15, NPU_HIGH)["denoise_step_ms"]
    p5_ref = project_vla(PI05, RTX_5090_REFERENCE)["denoise_step_ms"]
    p5_hi = project_vla(PI05, NPU_HIGH)["denoise_step_ms"]
    assert (p5_hi / p5_ref) > (n15_hi / n15_ref)    # π0.5 degrades by a larger factor


def test_pi05_optimized_floor_is_headroom_over_eager():
    r = project_vla(PI05, NPU_HIGH)
    assert r["runs"] and r["vla_source"] == "calibrated"
    assert r["action_hz_optimized_floor"] > r["action_hz"]
    assert r["denoise_bottleneck"].startswith("mixed")


def test_dtype_gate_bf16_on_int8_only_silicon():
    """Forcing bf16 on Mid (peak_tops_bf16=0) must not-run, not divide by zero."""
    r = project_vla(NORA, NPU_MID, dtype="bf16")
    assert r["runs"] is False
    assert "bf16" in r["reason"]


def test_npu_share_slows_bw_bound_decode():
    full = project_vla(NORA, NPU_MID, npu_share=1.0)["action_hz"]
    contended = project_vla(NORA, NPU_MID, npu_share=0.5)["action_hz"]
    assert contended < full                         # less BW → slower


# ── Phase 3c: multi-camera + fleet ──────────────────────────────────────────

from sizer.project_vla import project_vla_sensitivity  # noqa: E402

PI05 = VLA_MODELS["pi_0p5"]


def test_pi05_default_is_measured_3cam_and_1cam_scales_vision_down():
    """π0.5's measured anchor is 3-camera, so the default projection IS 3-camera
    (measured headline); n_cameras=1 down-scales vision ~3× (linear assumption)."""
    d3 = project_vla(PI05, RTX_5090_REFERENCE)                 # default → measured 3
    assert d3["n_cameras"] == 3 and d3["vla_source"] == "measured"
    d1 = project_vla(PI05, RTX_5090_REFERENCE, n_cameras=1)
    assert abs(d3["vision_ms"] / d1["vision_ms"] - 3.0) < 0.01  # 3-cam ≈ 3× 1-cam
    assert d3["mode"] == "multi_camera" and d1["mode"] == "single"
    assert d3["vision_ms_per_camera"] == d3["vision_ms"] / 3


def test_openvla_2cam_infeasible_native_max_1():
    r = project_vla(OPENVLA, NPU_HIGH, n_cameras=2)
    assert r["runs"] is False
    assert "supports max 1 camera" in r["reason"]


def test_fleet_memory_gate_and_per_robot_throttle():
    """fleet_size = N independent instances time-sharing one NPU: per-robot rate
    = single / N, memory = N × per-instance, overflow → INFEASIBLE."""
    ok = project_vla(NORA, NPU_HIGH, fleet_size=4)
    assert ok["runs"] is True and ok["mode"] == "fleet"
    assert ok["fleet_memory_gb"] == ok["dram_gb"] * 4
    assert abs(ok["action_hz"] - ok["per_instance_action_hz"] / 4) < 1e-6
    assert ok["aggregate_action_hz"] == ok["per_instance_action_hz"]
    big = project_vla(NORA, NPU_HIGH, fleet_size=100)
    assert big["runs"] is False and "exceeds" in big["reason"]


def test_pi05_fleet_multi_camera_combinatorial():
    r = project_vla(PI05, NPU_HIGH, n_cameras=3, fleet_size=2)
    assert r["runs"] is True and r["mode"] == "fleet_multi_camera"
    assert r["n_cameras"] == 3 and r["fleet_size"] == 2
    assert abs(r["ms_per_action"] - r["per_instance_ms_per_action"] * 2) < 1e-6


def test_offmeasured_camera_count_downgrades_to_cross_class():
    """Scaling to a camera count other than the measured one is linear-assumed →
    badge drops to 🟠 with a reason; the measured count stays calibrated."""
    r2 = project_vla(PI05, NPU_HIGH, n_cameras=2)               # measured at 3
    assert r2["vla_source"] == "cross_class"
    assert "assumed linear" in r2["vla_source_reason"]
    r3 = project_vla(PI05, NPU_HIGH, n_cameras=3)               # the measured count
    assert r3["vla_source"] == "calibrated" and "vla_source_reason" not in r3


def test_sensitivity_matrix_covers_camera_x_fleet_grid():
    rows = project_vla_sensitivity(PI05, NPU_HIGH,
                                   n_cameras_options=(1, 2, 3),
                                   fleet_size_options=(1, 2))
    assert len(rows) == 6
    assert {(r["n_cameras"], r["fleet_size"]) for r in rows} == {
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)}


def test_bitvla_measured_at_2cam_runs_on_int8_mid_fleet():
    """BitVLA: measured at 2 cameras (default), int_only → runs on INT8-only Mid
    even as a fleet. Because its measured default is 2 cams, any fleet is
    inherently fleet_multi_camera (vision_scale stays 1.0 → calibrated, no
    🟠 downgrade)."""
    r = project_vla(BITVLA, NPU_MID, fleet_size=3)
    assert r["n_cameras"] == 2                                  # measured default
    assert r["runs"] is True and r["mode"] == "fleet_multi_camera"
    assert r["vla_source"] == "calibrated"                      # 2 == measured, not downgraded
    assert r["fleet_memory_gb"] == r["dram_gb"] * 3


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
    print("\n--- NORA / OpenVLA / BitVLA / NORA-1.5 / π0.5 across tiers ---")
    for vla in (NORA, OPENVLA, BITVLA, NORA15, PI05):
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
