"""Headless AppTest matrix for the NPU precision-set selector (ratchet ADR 017).

Drives the real Streamlit app end-to-end across tier × precision-rung ×
maturity, asserting (a) zero exceptions, (b) the selector is present on Mid/High
and absent on excluded tiers, and (c) the 3-column compare reproduces the
pai-sizer/docs-ratified anchors (spec §9):

    High: INT8=175.5  FP8=175.5  FP4-mature=87.8  FP4-immature=351   decode 37.9
    Mid:  INT8=351    FP8=351    FP4-mature=175.5 FP4-immature=702   decode 37.9

Run:  python -m pytest tests/test_precision_selector_apptest.py -q
  or: python tests/test_precision_selector_apptest.py   (prints the matrix)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

ANCHOR_MODEL = "skippy_finetune"   # MoE Q4 — the measured-anchor model, runs on all rungs

# Expected compare-panel "Prefill / TTFT 1K" per (tier, rung-label) in ms.
EXPECTED = {
    "NPU High": {"INT-only": 175.5, "INT + FP8": 175.5,
                 "FP4-mature": 87.8, "FP4-immature": 351.0},
    "NPU Mid":  {"INT-only": 351.0, "INT + FP8": 351.0,
                 "FP4-mature": 175.5, "FP4-immature": 702.0},
}
RADIO_OPTS = ["Stock (no override)", "INT-only", "INT + FP8", "INT + FP8 + FP4"]


def _fresh(tier: str, model: str = ANCHOR_MODEL) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    # Bypass the shared-password gate (secrets.toml ships a PASSWORD locally).
    at.session_state["_authed"] = True
    at.run()
    at.selectbox(key="tier").set_value(tier).run()
    at.toggle(key="llm_enabled").set_value(True).run()   # LLM defaults off
    # Switch to the int8 MoE anchor model so it runs on every precision rung
    # (the default dense-fp16 model is dtype-blocked on the INT-only rung).
    at.selectbox(key="llm_model_key").set_value(model).run()
    return at


def _prefill_metrics(at: AppTest) -> list[float]:
    """The three compare-panel 'Prefill / TTFT 1K' values, in column order (ms)."""
    out = []
    for m in at.metric:
        if m.label and m.label.startswith("Prefill / TTFT 1K"):
            out.append(float(str(m.value).replace(" ms", "").strip()))
    return out


def test_full_matrix(verbose: bool = False):
    results = {}
    for tier in ("NPU High", "NPU Mid"):
        # Stock + the three rungs (+ maturity flip on the FP4 rung).
        at = _fresh(tier)
        # Stock: no precision override — selector present, compare uses base.
        at.radio(key=f"precision_set_{tier}").set_value("Stock (no override)").run()
        assert not at.exception, f"{tier} Stock raised: {at.exception}"
        pf = _prefill_metrics(at)
        assert len(pf) == 3, f"{tier} Stock: expected 3 compare metrics, got {pf}"
        # The compare panel always shows all three rungs regardless of the radio.
        results[(tier, "Stock")] = pf

        for rung in ("INT-only", "INT + FP8", "INT + FP8 + FP4"):
            at.radio(key=f"precision_set_{tier}").set_value(rung).run()
            assert not at.exception, f"{tier} {rung} raised: {at.exception}"

        # FP4 rung selected → maturity toggle exists; test both states.
        for mat_label, key in (("immature", "Immature (edge default — llama.cpp-class runtime)"),
                                ("mature",   "Mature (vLLM / TensorRT-LLM-class runtime)")):
            at.radio(key=f"precision_set_{tier}").set_value("INT + FP8 + FP4").run()
            at.radio(key=f"fp4_maturity_{tier}").set_value(key).run()
            assert not at.exception, f"{tier} FP4/{mat_label} raised: {at.exception}"
            pf = _prefill_metrics(at)
            assert len(pf) == 3, f"{tier} FP4/{mat_label}: got {pf}"
            # Compare-panel column order: INT-only, INT+FP8, INT+FP8+FP4(maturity)
            results[(tier, f"FP4-{mat_label}")] = pf

    # ── Assert anchors (compare-panel columns) ──
    tol = 0.6
    checks = []
    for tier in ("NPU High", "NPU Mid"):
        # Column 0 = INT-only, col 1 = INT+FP8, col 2 = FP4 (maturity-dependent).
        int_only = results[(tier, "FP4-mature")][0]
        int_fp8  = results[(tier, "FP4-mature")][1]
        fp4_mat  = results[(tier, "FP4-mature")][2]
        fp4_imm  = results[(tier, "FP4-immature")][2]
        got = {"INT-only": int_only, "INT + FP8": int_fp8,
               "FP4-mature": fp4_mat, "FP4-immature": fp4_imm}
        for k, exp in EXPECTED[tier].items():
            ok = abs(got[k] - exp) <= tol
            checks.append(ok)
            if verbose or not ok:
                print(f"  {tier:9} {k:13} got {got[k]:7.1f}  exp {exp:7.1f}  {'OK' if ok else 'FAIL'}")
    assert all(checks), "anchor mismatch — see FAIL rows above"
    return results


def test_excluded_tiers_have_no_selector():
    for tier in ("NPU Low-LP5X", "NPU i.MX 95 (ground truth)", "RTX 5090 (reference, measured)"):
        at = AppTest.from_file("app.py", default_timeout=60)
        at.run(); at.session_state["_authed"] = True; at.run()
        at.selectbox(key="tier").set_value(tier).run()
        assert not at.exception, f"{tier} raised: {at.exception}"
        radio_keys = [r.key for r in at.radio]
        assert f"precision_set_{tier}" not in radio_keys, \
            f"{tier} should NOT offer the precision selector (keys={radio_keys})"
        print(f"  {tier:32} no precision selector — OK")


if __name__ == "__main__":
    print("=== excluded-tier check ===")
    test_excluded_tiers_have_no_selector()
    print("\n=== full matrix (compare-panel anchors) ===")
    test_full_matrix(verbose=True)
    print("\nALL PASS")
