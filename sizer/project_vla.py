"""VLA (Vision-Language-Action) projection — Phase 3a (single-loop).

Projects per-action latency + action rate (Hz) for a `VLAModel` on a target
`Hardware` tier. **Single-loop autoregressive only** (NORA, OpenVLA, BitVLA).
Dual-loop (flow-matching action expert) is Phase 3b — deferred until a
measured dual-loop anchor lands; this function returns a `deferred` result
for any non-single-loop architecture rather than guessing.

Composition (single-loop, autoregressive):

    ms_per_action = vision_ms + llm_prefill_ms + n_action_tokens × decode_ms/token
    action_hz     = 1000 / ms_per_action

This mirrors the existing engine: vision + LLM prefill are compute-bound
forwards (cf. `project_vision`'s compute floor), action-token decode is
BW-bound (cf. `project_llm`'s decode ceiling).

## Three projection regimes (3-state taxonomy, ratified 2026-05-29)

Distinct from `project_llm`'s `same_class` semantics on purpose: the RTX 5090
shares no `tier_family` with any NPU, and Kyle's locked principle is that
"straight-line scaling from the 5090 has a slope assumption that breaks at
class boundaries." So a 5090→NPU VLA projection is honest about being a
cross-class projection — but distinguishes a 5090-*anchored* roofline from a
pure paper-FLOP guess:

  🟢 measured    — hw IS the reference (5090) and the model is calibrated:
                   return the measured component latencies verbatim.
  🔵 calibrated  — model calibrated, hw != reference: scale the measured 5090
                   component latencies to the target by per-component
                   compute/BW ratio (latency-anchor scaling). Footgun-free —
                   the 5090 util cancels in the ratio, so we never re-apply
                   the back-solved "effective FLOP at 5090 util" against a
                   different tier's util.
  🟠 cross_class — model NOT calibrated: first-principles roofline from the
                   catalog's per-component flops_per_call_g / arithmetic_intensity.

Only a real **NPU-class** VLA measurement would earn 🟡 `same_class` (none
exists yet). See `memory/project_vla_arc.md`.
"""
from __future__ import annotations

from ratchet import Hardware, RTX_5090_REFERENCE
from .npu_model import hw_peak_tops_for_dtype
from .vla_models import VLAModel


# The reference bake-offs were captured at bf16 (see vla_summary.json). The
# 🔵 calibrated path scales FROM this dtype on the reference TO the target's
# execution dtype (edge tiers commonly run int8). Crossing precision is
# inherent to edge projection and is part of why these are 🔵, not 🟢.
_REFERENCE_DTYPE = "bf16"

# Bytes/param by execution dtype — for the memory-footprint pick only.
_DTYPE_DRAM_FIELD = {
    "bf16": "inference_dram_gb_bf16",
    "fp8": "inference_dram_gb_int8",   # fp8 ≈ 1 byte/param, same footprint class as int8
    "int8": "inference_dram_gb_int8",
}

# Weight bytes/param by dtype — applied ONLY to the BW-bound decode term in
# the 🔵 calibrated path. The 5090 decode was measured at bf16 (~2 B/param);
# an edge tier running int8 streams ~half the bytes/token, so decode latency
# scales by this ratio in addition to the BW ratio. Compute-bound terms
# (vision, prefill) do NOT get this factor — they do the same FLOPs at any
# precision; the throughput difference is already in peak_tops_for_dtype.
_BYTES_PER_PARAM = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


def _resolve_exec_dtype(vla: VLAModel, dtype: str | None) -> str:
    """Pick the execution dtype for the compute floors / peak-TOPS lookup.

    Explicit `dtype` wins; else derive from the model's dtype_path_default.
    Encodings (see VLAModel.dtype_path_default): 'int8' / 'int_only' /
    'int8+fp8' / 'int8+bf16' all execute the VLM stages at int8 (the action
    head's FP requirement only matters for dual-loop, which is deferred);
    'fp8+bf16' → fp8; bare 'bf16' → bf16.
    """
    if dtype:
        return dtype
    path = (vla.dtype_path_default or "").lower()
    if path.startswith("int8") or path == "int_only":
        return "int8"
    if path.startswith("fp8"):
        return "fp8"
    if path.startswith("bf16"):
        return "bf16"
    return "bf16"


def _dram_gb_for_dtype(vla: VLAModel, exec_dtype: str) -> float | None:
    field = _DTYPE_DRAM_FIELD.get(exec_dtype, "inference_dram_gb_bf16")
    return getattr(vla, field, None)


def project_vla(
    vla: VLAModel,
    hw: Hardware,
    *,
    dtype: str | None = None,
    n_action_tokens: int | None = None,
    npu_share: float | None = None,
    reference: Hardware = RTX_5090_REFERENCE,
) -> dict:
    """Project single-loop VLA per-action latency + action rate on `hw`.

    Returns a dict (mirrors project_llm's shape). For dual-loop architectures
    returns `{"deferred": True, ...}` — Phase 3b. For a model/dtype that won't
    run on `hw` returns `{"runs": False, ...}` with a reason.
    """
    share = npu_share if npu_share is not None else hw.npu_share_default
    share = max(share, 1e-6)
    n_tok = n_action_tokens if n_action_tokens is not None else vla.n_action_tokens
    exec_dtype = _resolve_exec_dtype(vla, dtype)

    base = {
        "vla": vla.key,
        "hw": hw.name,
        "dtype": exec_dtype,
        "architecture": vla.architecture,
        "n_action_tokens": n_tok,
        "npu_share": share,
    }

    # ── Phase 3a scope gate: single-loop only ──────────────────────────────
    if vla.architecture != "single_loop":
        return {**base, "deferred": True,
                "reason": (f"{vla.architecture} projection is Phase 3b (needs a "
                           "measured dual-loop anchor: flow-matching action "
                           "expert + duty-cycle + FP-required gate)")}

    # ── dtype runnability gate ──────────────────────────────────────────────
    # Each component must accept the execution dtype, and the silicon must have
    # nonzero peak at it (catches e.g. a bf16-only model on Mid, which is INT8-only).
    for comp in vla.components.values():
        if exec_dtype not in comp.dtype_required:
            return {**base, "runs": False,
                    "reason": (f"component '{comp.name}' requires "
                               f"{comp.dtype_required}; execution dtype "
                               f"'{exec_dtype}' not among them")}
    peak = hw_peak_tops_for_dtype(hw, exec_dtype)
    if peak <= 0:
        return {**base, "runs": False,
                "reason": f"{hw.name} has no {exec_dtype} compute (peak_tops=0)"}

    # ── memory gate ─────────────────────────────────────────────────────────
    dram_gb = _dram_gb_for_dtype(vla, exec_dtype)
    fits = None if dram_gb is None else (dram_gb + 0.5) < hw.mem_capacity_gb

    is_reference = (hw.name == reference.name and hw.tier_family == reference.tier_family)
    comp_meas = vla.measured_5090_components

    if vla.measured_5090_calibrated and comp_meas:
        if is_reference:
            # 🟢 measured — return the measured component latencies verbatim.
            # ms_per_action uses the authoritative measured e2e headline
            # (composing from components differs ~1% due to run-to-run).
            vla_source, regime = "measured", "single_loop_measured"
            vision_ms = comp_meas["vision_ms"]
            prefill_ms = comp_meas["llm_prefill_ms"]
            decode_ms_tok = comp_meas["decode_ms_per_token"]
            ms_per_action = (vla.measured_5090_ms_per_action
                             if vla.measured_5090_ms_per_action is not None
                             else vision_ms + prefill_ms + n_tok * decode_ms_tok)
        else:
            # 🔵 calibrated — latency-anchor scaling. Vision + LLM-prefill are
            # compute-bound (scale by their respective effective-compute ratios,
            # which use DIFFERENT util factors); decode is BW-bound (scale by
            # effective-BW ratio). The reference's measured dtype (bf16) is used
            # on the reference side; the target uses its execution dtype.
            vla_source, regime = "calibrated", "single_loop_calibrated_latency_scaled"
            ref_peak = hw_peak_tops_for_dtype(reference, _REFERENCE_DTYPE)

            ref_vis_eff = ref_peak * reference.compute_util_factor
            tgt_vis_eff = peak * hw.compute_util_factor
            vision_ms = comp_meas["vision_ms"] * (ref_vis_eff / tgt_vis_eff)

            ref_pf_eff = ref_peak * reference.llm_prefill_util_factor
            tgt_pf_eff = peak * hw.llm_prefill_util_factor
            prefill_ms = comp_meas["llm_prefill_ms"] * (ref_pf_eff / tgt_pf_eff)

            # decode BW-bound: target BW reduced by npu_share contention, AND
            # the weight-byte volume rescaled if the target executes at a lower
            # precision than the reference's measured bf16 (int8 edge ≈ half).
            bw_ratio = reference.effective_bandwidth_gbs / (hw.effective_bandwidth_gbs * share)
            bpp_factor = _BYTES_PER_PARAM.get(exec_dtype, 2.0) / _BYTES_PER_PARAM[_REFERENCE_DTYPE]
            decode_ms_tok = comp_meas["decode_ms_per_token"] * bw_ratio * bpp_factor

            ms_per_action = vision_ms + prefill_ms + n_tok * decode_ms_tok
    else:
        # 🟠 cross_class — first-principles roofline from catalog component
        # FLOPs / arithmetic intensity. Un-calibrated: no overhead model, so
        # this is a floor-ish estimate (trust less than 🟢/🔵).
        vla_source, regime = "cross_class", "single_loop_cross_class_roofline"
        vision = vla.components["vision_encoder"]
        llm = vla.components["llm_backbone"]

        vision_ms = vision.flops_per_call_g / (peak * hw.compute_util_factor)
        prefill_ms = llm.flops_per_call_g / (peak * hw.llm_prefill_util_factor)
        # decode bytes/token from AI: bytes = decode_flops / AI; decode_flops/tok
        # ≈ 2 × active LLM params (GPT-style matmul). BW-bound time = bytes / BW.
        decode_flops_g = 2.0 * llm.params_b
        ai = max(llm.arithmetic_intensity, 1e-6)
        bytes_per_tok_gb = decode_flops_g / ai
        decode_ms_tok = bytes_per_tok_gb / (hw.effective_bandwidth_gbs * share) * 1000.0

        ms_per_action = (vision_ms + prefill_ms + n_tok * decode_ms_tok
                         + hw.compute_overhead_ms)

    action_hz = 1000.0 / ms_per_action if ms_per_action > 0 else 0.0

    return {
        **base,
        "runs": True,
        "vla_source": vla_source,
        "regime": regime,
        "vision_ms": vision_ms,
        "llm_prefill_ms": prefill_ms,
        "vlm_forward_ms": vision_ms + prefill_ms,
        "decode_ms_per_token": decode_ms_tok,
        "ms_per_action": ms_per_action,
        "action_hz": action_hz,
        "dram_gb": dram_gb,
        "fits_in_memory": fits,
    }
