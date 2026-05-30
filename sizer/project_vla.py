"""VLA (Vision-Language-Action) projection — Phase 3a (single-loop) + 3b (dual-loop).

Projects per-action latency + action rate (Hz) for a `VLAModel` on a target
`Hardware` tier.

**Single-loop autoregressive** (NORA-3B, OpenVLA, BitVLA), composition:

    ms_per_action = vision_ms + llm_prefill_ms + n_action_tokens × decode_ms/token
    action_hz     = 1000 / ms_per_action

This mirrors the existing engine: vision + LLM prefill are compute-bound
forwards (cf. `project_vision`'s compute floor), action-token decode is
BW-bound (cf. `project_llm`'s decode ceiling).

**Dual-loop** (flow-matching action expert: NORA-1.5, π0.5) — Phase 3b. The VLM
backbone runs ONCE per chunk → frozen KV cache; a separate action expert runs
N denoise steps to emit an H-action chunk:

    chunk_latency_ms        = vlm_backbone_ms + N × denoise_step_ms
    ms_per_action           = chunk_latency_ms / H     (amortized control latency)
    fast_loop_only_hz       = H / (N × denoise_step_ms) (VLM reused/pipelined)

The two loops have CATEGORICALLY different physics, so they project differently:

  • Slow loop (VLM backbone) — compute-bound (vision ~12% / prefill ~23% util on
    the 5090). Genuinely silicon-dependent → latency-anchor scaled exactly like
    single-loop's vision+prefill (🔵 calibrated).
  • Fast loop (denoise step) — kernel-LAUNCH/dispatch-bound in stock eager HF
    (0.07% peak FLOP, 1.6% peak BW). The bottleneck is host-side dispatch, NOT
    silicon. So the measured step is carried UNCHANGED to edge as the **eager
    ceiling** (launch-bound ⇒ silicon-independent ⇒ edge eager ≈ 5090 eager),
    and a separate **physics floor** (roofline on the FP expert weights) gives
    the optimized best case. It is NEVER bandwidth-scaled — doing so (the
    single-loop AR-decode treatment) would produce ~340 ms/step nonsense and is
    exactly the footgun [backend] flagged in 4caf501.

  HARD GATE: flow-matching action experts REQUIRE FP (fp8/bf16) per QuantVLA —
  INT8 of the diffusion head breaks task success. An INT8-only tier (NPU Mid:
  fp8=bf16=0) therefore CANNOT run a dual-loop model at all → `runs: False`.

Dual-loop models WITHOUT a measured anchor (π0.5, openvla_7b_cached) still
defer — the projection needs the measured `measured_5090_dual_loop` topology.

**OFT parallel-chunk** (BitVLA) — Phase 3b. ONE VLM forward over
[image + prompt + H action-placeholder tokens + proprio] → an L1-regression head
reads the action-position hidden states in PARALLEL → H actions from a single
forward. NO autoregressive token loop, NO decode-per-token, NO AR-decode BW-wall:

    ms_per_action = (vision_ms + llm_forward_ms) / H        (amortized over chunk)

Both stages are PREFILL-shaped compute-bound forwards → latency-anchor scaled
exactly like single-loop's vision+prefill (🔵 calibrated). This is WHY OFT is
fast (BitVLA 65 Hz vs OpenVLA-7B's 7.9 Hz AR). No FP gate (BitVLA is int_only →
runs on INT8-only silicon). NB: ternary-kernel speedups (bitblas/LUT, 0.2-byte
weights) are a SEPARATE optimistic floor, not the measured bf16-dense reality —
the projection uses the measured latencies; see the BitVLA catalog notes.

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

    # ── architecture dispatch ───────────────────────────────────────────────
    if vla.architecture == "oft_parallel_chunk":
        return _project_oft_parallel(vla, hw, base, share=share, reference=reference)
    if vla.architecture != "single_loop":
        return _project_dual_loop(vla, hw, base, share=share, reference=reference)

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


# ── dual-loop (Phase 3b) ────────────────────────────────────────────────────

def _resolve_expert_dtype(vla: VLAModel, hw: Hardware) -> str | None:
    """Pick the FP execution dtype for the flow-matching action expert.

    The expert REQUIRES FP (fp8/bf16) per QuantVLA. Prefer the dtype_path's FP
    component, fall back to the other FP if the silicon only has one, return
    None if the silicon has NEITHER (INT8-only tier → the hard FP gate fires).
    """
    path = (vla.dtype_path_default or "").lower()
    preferred = "fp8" if "fp8" in path else ("bf16" if "bf16" in path else "fp8")
    order = [preferred] + [d for d in ("fp8", "bf16") if d != preferred]
    for d in order:
        if hw_peak_tops_for_dtype(hw, d) > 0:
            return d
    return None


def _denoise_edge_step(dl: dict, hw: Hardware, share: float,
                       reference: Hardware) -> tuple[float, float]:
    """Project one denoise step to `hw` via the launch+BW decomposition.

    The fast loop is partly kernel-LAUNCH/dispatch-bound (silicon-independent)
    and partly genuinely BW-bound. Naively BW-scaling the WHOLE measured step is
    the footgun [backend] flagged (→ ~340 ms/step nonsense); carrying it whole
    is right for a *pure* launch-bound step (NORA-1.5, 1.6% BW) but UNDER-projects
    a partial-BW step (π0.5, 13.4% BW). So decompose, using the measured effective
    traffic per step, and scale ONLY the BW-bound fraction by edge bandwidth:

        bytes/step    = denoise_step_effective_bw_gbs × step_ms        (real traffic)
        bw_floor_ref  = bytes/step ÷ reference effective BW            (BW-bound time @ ref)
        launch_const  = max(step_ms − bw_floor_ref, 0)                 (silicon-independent)
        edge_bw_floor = bytes/step ÷ (target effective BW × npu_share) (BW-bound time @ edge)

    Returns (eager_edge_step, optimized_floor_step):
      • eager = launch_const + edge_bw_floor  — conservative headline (stock eager:
        launch overhead survives + the BW fraction degrades on slower memory).
      • optimized_floor = edge_bw_floor       — compiled/fused best case (launch
        eliminated; only the genuine BW wall remains) — the headroom.

    No string-branching on `denoise_bottleneck`: a pure-launch step has tiny
    bytes → edge ≈ launch_const (≈ unchanged), big headroom; a partial-BW step
    has large bytes → edge_bw_floor dominates and scales hard. On the reference
    (ratio=1, share=1) both reduce to the measured step exactly. The label is
    carried to the result for the UI/deck, not used as control flow.
    """
    step_ms = dl["denoise_step_ms"]
    eff_bw = dl.get("denoise_step_effective_bw_gbs")
    if not eff_bw or eff_bw <= 0:
        return step_ms, step_ms                          # no BW telemetry → treat as launch-bound
    bytes_gb = eff_bw * step_ms / 1000.0                 # measured real traffic per step
    bw_floor_ref = bytes_gb / reference.effective_bandwidth_gbs * 1000.0
    launch_const = max(step_ms - bw_floor_ref, 0.0)      # silicon-independent overhead
    edge_bw_floor = bytes_gb / max(hw.effective_bandwidth_gbs * share, 1e-6) * 1000.0
    return launch_const + edge_bw_floor, edge_bw_floor


def _project_dual_loop(vla: VLAModel, hw: Hardware, base: dict, *,
                       share: float, reference: Hardware) -> dict:
    """Project dual-loop (flow-matching) per-action latency + action rate.

    Slow loop (VLM backbone, once/chunk) is 🔵 calibrated (latency-anchor scaled
    like single-loop's vision+prefill). Fast loop (denoise) is launch-bound: the
    measured step is carried UNCHANGED as the eager ceiling (headline) + a physics
    floor as headroom — never BW-scaled. FP-required head is a hard gate.
    """
    dl = vla.measured_5090_dual_loop
    comp_meas = vla.measured_5090_components

    # Un-calibrated dual-loop (π0.5, openvla_7b_cached): no measured topology yet.
    if not (vla.measured_5090_calibrated and dl and comp_meas):
        return {**base, "deferred": True,
                "reason": (f"{vla.architecture} projection needs a measured "
                           "dual-loop anchor (measured_5090_dual_loop): "
                           "flow-matching backbone-once + denoise-loop split. "
                           "Phase 3b absorbs these per-model as bake-offs land.")}

    H = dl["action_chunk_length"]
    N = dl["num_denoise_steps"]
    base = {**base, "n_action_tokens": H, "architecture": vla.architecture}

    # ── VLM-stage dtype + runnability ────────────────────────────────────────
    vlm_dtype = _resolve_exec_dtype(vla, None)        # int8 part of the path
    for comp in ("vision_encoder", "llm_backbone"):
        req = vla.components[comp].dtype_required
        if vlm_dtype not in req:
            return {**base, "runs": False,
                    "reason": (f"VLM component '{comp}' requires {req}; "
                               f"VLM dtype '{vlm_dtype}' not among them")}
    vlm_peak = hw_peak_tops_for_dtype(hw, vlm_dtype)
    if vlm_peak <= 0:
        return {**base, "runs": False,
                "reason": f"{hw.name} has no {vlm_dtype} compute (peak_tops=0)"}

    # ── FP-required action-expert gate (the hard QuantVLA gate) ──────────────
    expert_dtype = _resolve_expert_dtype(vla, hw)
    if expert_dtype is None:
        return {**base, "runs": False,
                "reason": ("flow-matching action expert REQUIRES FP (fp8/bf16) "
                           "per QuantVLA; "
                           f"{hw.name} has neither (INT8-only) — the FP-required "
                           "diffusion head is a hard gate that eliminates this tier")}

    # ── memory gate (VLM-dtype footprint) ────────────────────────────────────
    dram_gb = _dram_gb_for_dtype(vla, vlm_dtype)
    fits = None if dram_gb is None else (dram_gb + 0.5) < hw.mem_capacity_gb

    is_reference = (hw.name == reference.name and hw.tier_family == reference.tier_family)
    # Fast loop: launch+BW decomposition (handles pure-launch NORA-1.5 AND
    # partial-BW π0.5 without string-branching; reduces to measured on the ref).
    eager_step_ms, floor_step_ms = _denoise_edge_step(dl, hw, share, reference)

    if is_reference:
        # 🟢 measured — return the measured topology verbatim.
        vla_source, regime = "measured", "dual_loop_measured"
        vision_ms = comp_meas["vision_ms"]
        prefill_ms = comp_meas["llm_prefill_ms"]
        vlm_backbone_ms = dl["vlm_backbone_ms"]
        expert_dtype = _REFERENCE_DTYPE               # measured at bf16
        ms_per_action = dl["amortized_ms_per_action"]  # authoritative measured amortization
        chunk_latency_ms = dl["chunk_latency_ms"]
    else:
        # 🔵 calibrated VLM backbone (compute-bound, latency-anchor scaled) +
        # decomposed fast loop (launch_const survives, BW fraction scales).
        vla_source, regime = "calibrated", "dual_loop_calibrated_vlm_decomposed_expert"
        ref_peak = hw_peak_tops_for_dtype(reference, _REFERENCE_DTYPE)

        ref_vis_eff = ref_peak * reference.compute_util_factor
        tgt_vis_eff = vlm_peak * hw.compute_util_factor
        vision_ms = comp_meas["vision_ms"] * (ref_vis_eff / tgt_vis_eff)

        ref_pf_eff = ref_peak * reference.llm_prefill_util_factor
        tgt_pf_eff = vlm_peak * hw.llm_prefill_util_factor
        prefill_ms = comp_meas["llm_prefill_ms"] * (ref_pf_eff / tgt_pf_eff)

        vlm_backbone_ms = vision_ms + prefill_ms
        chunk_latency_ms = vlm_backbone_ms + N * eager_step_ms
        ms_per_action = chunk_latency_ms / H

    action_hz = 1000.0 / ms_per_action if ms_per_action > 0 else 0.0
    eager_loop_ms = N * eager_step_ms
    fast_loop_only_hz = (H / (eager_loop_ms / 1000.0)) if eager_loop_ms > 0 else 0.0

    # Optimized physics floor (headroom) — what the fast loop could hit compiled.
    floor_loop_ms = N * floor_step_ms
    floor_chunk_ms = vlm_backbone_ms + floor_loop_ms
    floor_ms_per_action = floor_chunk_ms / H
    floor_action_hz = 1000.0 / floor_ms_per_action if floor_ms_per_action > 0 else 0.0
    floor_fast_loop_only_hz = (H / (floor_loop_ms / 1000.0)) if floor_loop_ms > 0 else 0.0

    return {
        **base,
        "runs": True,
        "vla_source": vla_source,
        "regime": regime,
        "vlm_dtype": vlm_dtype,
        "expert_dtype": expert_dtype,
        # slow loop (VLM backbone, once/chunk)
        "vision_ms": vision_ms,
        "llm_prefill_ms": prefill_ms,
        "vlm_backbone_ms": vlm_backbone_ms,
        # fast loop — eager ceiling (headline, launch-bound, silicon-independent)
        "num_denoise_steps": N,
        "action_chunk_length": H,
        "denoise_step_ms": eager_step_ms,
        "denoise_loop_ms": eager_loop_ms,
        "denoise_bottleneck": dl.get("denoise_bottleneck"),
        "denoise_projection": "measured" if is_reference else "eager_launch_plus_bw_decomp",
        "chunk_latency_ms": chunk_latency_ms,
        "ms_per_action": ms_per_action,
        "action_hz": action_hz,
        "fast_loop_only_hz": fast_loop_only_hz,
        # fast loop — optimized physics floor (headroom; compiled/fused best case)
        "denoise_step_ms_optimized_floor": floor_step_ms,
        "ms_per_action_optimized_floor": floor_ms_per_action,
        "action_hz_optimized_floor": floor_action_hz,
        "fast_loop_only_hz_optimized_floor": floor_fast_loop_only_hz,
        "dram_gb": dram_gb,
        "fits_in_memory": fits,
    }


# ── OFT parallel-chunk (Phase 3b) ───────────────────────────────────────────

def _project_oft_parallel(vla: VLAModel, hw: Hardware, base: dict, *,
                          share: float, reference: Hardware) -> dict:
    """Project OFT parallel-chunk (BitVLA) per-action latency + action rate.

    ONE prefill-shaped VLM forward → H actions read in parallel by an
    L1-regression head. ms_per_action = (vision + llm_forward) / H. Both stages
    are compute-bound → latency-anchor scaled like single-loop's vision+prefill.
    NO decode term, NO FP gate (int_only runs on INT8-only silicon). The measured
    anchor is the bf16-dense reality; ternary-kernel speedups are NOT applied here.
    """
    of = vla.measured_5090_oft
    if not (vla.measured_5090_calibrated and of):
        return {**base, "deferred": True,
                "reason": (f"{vla.architecture} projection needs a measured OFT "
                           "anchor (measured_5090_oft): parallel forward + "
                           "chunk split. Phase 3b absorbs these as bake-offs land.")}

    H = of["action_chunk_length"]
    exec_dtype = base["dtype"]
    base = {**base, "n_action_tokens": H, "architecture": vla.architecture}

    # ── runnability (compute-bound forward, no FP gate) ──────────────────────
    for comp in ("vision_encoder", "llm_backbone"):
        req = vla.components[comp].dtype_required
        if exec_dtype not in req:
            return {**base, "runs": False,
                    "reason": (f"component '{comp}' requires {req}; execution "
                               f"dtype '{exec_dtype}' not among them")}
    peak = hw_peak_tops_for_dtype(hw, exec_dtype)
    if peak <= 0:
        return {**base, "runs": False,
                "reason": f"{hw.name} has no {exec_dtype} compute (peak_tops=0)"}

    dram_gb = _dram_gb_for_dtype(vla, exec_dtype)
    fits = None if dram_gb is None else (dram_gb + 0.5) < hw.mem_capacity_gb

    is_reference = (hw.name == reference.name and hw.tier_family == reference.tier_family)

    if is_reference:
        # 🟢 measured — return the measured forward split + amortized verbatim.
        vla_source, regime = "measured", "oft_parallel_chunk_measured"
        vision_ms = of["vision_ms"]
        llm_forward_ms = of["llm_forward_ms"]
        forward_ms = of["forward_ms"]
        ms_per_action = of["ms_per_action"]
    else:
        # 🔵 calibrated — both stages compute-bound, latency-anchor scaled (vision
        # by compute_util_factor, llm_forward by llm_prefill_util_factor — the
        # parallel forward is prefill-shaped). Amortize over the H-action chunk.
        vla_source, regime = "calibrated", "oft_parallel_chunk_calibrated_latency_scaled"
        ref_peak = hw_peak_tops_for_dtype(reference, _REFERENCE_DTYPE)

        ref_vis_eff = ref_peak * reference.compute_util_factor
        tgt_vis_eff = peak * hw.compute_util_factor
        vision_ms = of["vision_ms"] * (ref_vis_eff / tgt_vis_eff)

        ref_pf_eff = ref_peak * reference.llm_prefill_util_factor
        tgt_pf_eff = peak * hw.llm_prefill_util_factor
        llm_forward_ms = of["llm_forward_ms"] * (ref_pf_eff / tgt_pf_eff)

        forward_ms = vision_ms + llm_forward_ms
        ms_per_action = forward_ms / H

    action_hz = 1000.0 / ms_per_action if ms_per_action > 0 else 0.0

    return {
        **base,
        "runs": True,
        "vla_source": vla_source,
        "regime": regime,
        "vision_ms": vision_ms,
        "llm_forward_ms": llm_forward_ms,
        "vlm_forward_ms": forward_ms,            # the full parallel forward (vision + llm)
        "forward_ms": forward_ms,
        "action_chunk_length": H,
        "ms_per_action": ms_per_action,
        "action_hz": action_hz,
        "dram_gb": dram_gb,
        "fits_in_memory": fits,
    }
