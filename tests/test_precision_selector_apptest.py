"""Headless AppTest for the NPU precision what-if compare (ratchet ADR 017).

The precision selector is a SELF-CONTAINED what-if block in the LLM section
(relocated out of the sidebar 2026-06-05): a 3-column compare with its own FP4
runtime-maturity toggle. The headline projection runs on the REAL stock tier;
this block posits FP-capable silicon at the same memory class.

Asserts: (a) zero exceptions, (b) NO sidebar precision radio anywhere
(`precision_set_*`), (c) the compare reproduces the pai-sizer/docs-ratified
anchors (spec §9), and (d) the what-if is absent on excluded tiers.

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

ANCHOR_MODEL = "skippy_finetune"   # MoE Q4 — measured-anchor model, runs on all rungs

# Column order in the compare panel: INT-only, INT+FP8, INT+FP8+FP4(maturity).
EXPECTED = {
    "NPU High": {"mature": [175.5, 175.5, 87.8], "immature": [175.5, 175.5, 351.0]},
    "NPU Mid":  {"mature": [351.0, 351.0, 175.5], "immature": [351.0, 351.0, 702.0]},
}
MAT_OPT = {
    "immature": "Immature — edge default (llama.cpp-class runtime)",
    "mature":   "Mature — vLLM / TensorRT-LLM-class runtime",
}


def _fresh(tier: str, model: str = ANCHOR_MODEL) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.session_state["_authed"] = True          # bypass shared-password gate
    at.run()
    at.selectbox(key="tier").set_value(tier).run()
    at.toggle(key="llm_enabled").set_value(True).run()   # LLM defaults off
    at.selectbox(key="llm_model_key").set_value(model).run()
    return at


def _prefill_metrics(at: AppTest) -> list[float]:
    """The three compare-panel 'Prefill / TTFT 1K' values, in column order (ms)."""
    return [float(str(m.value).replace(" ms", "").strip())
            for m in at.metric
            if m.label and m.label.startswith("Prefill / TTFT 1K")]


def _radio_keys(at: AppTest) -> list[str]:
    return [r.key for r in at.radio]


def test_no_sidebar_precision_radio():
    """The old sidebar `precision_set_*` radio must be gone (relocated)."""
    for tier in ("NPU Mid", "NPU High"):
        at = _fresh(tier)
        keys = _radio_keys(at)
        assert f"precision_set_{tier}" not in keys, \
            f"{tier} still has the sidebar precision radio (keys={keys})"
        # The relocated maturity toggle lives in the compare panel instead.
        assert f"fp4_maturity_compare_{tier}" in keys, \
            f"{tier} missing the compare-panel maturity toggle (keys={keys})"
        print(f"  {tier:9} sidebar radio gone, compare toggle present — OK")


def test_full_matrix(verbose: bool = False):
    results = {}
    for tier in ("NPU High", "NPU Mid"):
        at = _fresh(tier)
        for mat in ("mature", "immature"):
            at.radio(key=f"fp4_maturity_compare_{tier}").set_value(MAT_OPT[mat]).run()
            assert not at.exception, f"{tier}/{mat} raised: {at.exception}"
            pf = _prefill_metrics(at)
            assert len(pf) == 3, f"{tier}/{mat}: expected 3 compare metrics, got {pf}"
            results[(tier, mat)] = pf

    tol = 0.6   # UI metric rounds to whole ms (175.5 -> "175"); engine is exact
    checks = []
    for tier in ("NPU High", "NPU Mid"):
        for mat in ("mature", "immature"):
            got, exp = results[(tier, mat)], EXPECTED[tier][mat]
            for col, (g, e) in enumerate(zip(got, exp)):
                ok = abs(g - e) <= tol
                checks.append(ok)
                if verbose or not ok:
                    name = ["INT-only", "INT+FP8", f"FP4-{mat}"][col]
                    print(f"  {tier:9} {name:13} got {g:7.1f}  exp {e:7.1f}  "
                          f"{'OK' if ok else 'FAIL'}")
    assert all(checks), "anchor mismatch — see FAIL rows above"
    return results


def test_excluded_tiers_have_no_whatif():
    for tier in ("NPU Low-LP5X", "NPU i.MX 95 (ground truth)",
                 "RTX 5090 (reference, measured)"):
        at = _fresh(tier)
        assert not at.exception, f"{tier} raised: {at.exception}"
        keys = _radio_keys(at)
        assert f"fp4_maturity_compare_{tier}" not in keys, \
            f"{tier} should NOT offer the precision what-if (keys={keys})"
        assert f"precision_set_{tier}" not in keys
        print(f"  {tier:32} no precision what-if — OK")


if __name__ == "__main__":
    print("=== sidebar radio removed / toggle relocated ===")
    test_no_sidebar_precision_radio()
    print("\n=== excluded-tier check ===")
    test_excluded_tiers_have_no_whatif()
    print("\n=== full matrix (compare-panel anchors) ===")
    test_full_matrix(verbose=True)
    print("\nALL PASS")
