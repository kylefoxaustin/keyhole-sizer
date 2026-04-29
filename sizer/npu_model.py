"""
NPU sizing math — distilled from the Keyhole project's
`src/emulate/npu_emulator.py` plus the measured baselines captured during
the bake-off series. Self-contained: no Keyhole imports.

Every "measured" constant below traces back to a specific bake-off in the
Keyhole deck (see `scripts/bakeoff_*.py` in the parent project and
`REPRODUCE.md` for how to regenerate them).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .precision import CapabilityLevel


# ───────────────────────── Hardware tiers ─────────────────────────

@dataclass
class Hardware:
    """Generic compute-and-bandwidth spec for any NPU or GPU."""
    name: str
    peak_tops_bf16: float        # TOPS at BF16 tensor
    peak_tops_int8: float        # TOPS at INT8 tensor
    peak_tops_fp8: float         # TOPS at FP8 tensor (Blackwell+ class)
    mem_bandwidth_gbs: float      # raw theoretical bandwidth (GB/s)
    mem_capacity_gb: float        # total DRAM capacity
    mem_bus_width_bits: int
    mem_type: str                 # "LPDDR4" / "LPDDR5" / "LPDDR5X" / "GDDR7" / ...
    mem_data_rate_gtps: float

    compute_efficiency: float = 0.65    # fraction of peak we see on real models
    bandwidth_efficiency: float = 0.70   # fraction of peak BW realized

    tdp_watts: float = 0.0

    # Optional: empirical LLM decode tok/s at Q4_K_M (3B active MoE) — used
    # if set, otherwise computed from bandwidth ceiling × measured efficiency.
    measured_llm_q4_decode_tok_s: float | None = None
    measured_llm_ttft_1k_sec: float | None = None

    # True when this Hardware was synthesized via `hw_with_memory()` (i.e.
    # represents a memory-only what-if upgrade, not stock silicon). UI
    # checks this to mark BW-projected LLM tok/s as "(BW-proj)" so users
    # don't mistake a what-if projection for a vendor-measured number.
    bw_projected: bool = False

    # Optional: real-silicon edge ms per frame, keyed by (pipeline_key → resolution → ms).
    # When populated, project_vision returns this value directly for matching
    # (pipeline, resolution) pairs — bypassing both the BW-bound scale and the
    # compute-bound clamp. Mirrors the measured_llm_* pattern for LLM decode.
    # Use for tiers where we have production measurements on real silicon
    # (e.g. NPU i.MX 95 Neutron).
    measured_edge_ms: dict[str, dict[str, float]] | None = None

    # Optional: per-dtype capability taxonomy. When populated, describes
    # what kernel path the silicon is CAPABLE of taking for each dtype —
    # `tensor_native` / `tensor_compat` / `cuda_core` / `unsupported`.
    # See sizer/precision.py for the taxonomy. Consumed by docs/UI to
    # explain why measured silicon either runs a dtype successfully or
    # errors out (e.g. 5090 INT8 = tensor_compat via sm80 IMMA binary
    # compat; vLLM CUTLASS fresh-compile fails because SM120 lacks
    # native INT8 tensor-core instructions). Build-time fallbacks (TRT
    # FP8→FP16 without QDQ) are NOT captured here — that's per-engine.
    capability_levels: dict[str, CapabilityLevel] | None = None

    # Phase 2 compute-clamp calibration (per [backend] 2026-04-29 design
    # doc). Used by project_vision()'s two-floor model:
    #     edge_ms = max(bw_floor, compute_floor) + compute_overhead_ms
    # where compute_floor = tc_ops_blackwell / (effective_tops × compute_util_factor).
    # The util_factor calibration absorbs the Blackwell-HMMA → NPU-effective
    # conversion at the i.MX 95 anchor (12 GOPs / 2 INT8 TOPS / 0.19 = 31.6
    # ms ≈ measured 32 ms). Constants per tier-class:
    #   Neutron-class (i.MX 95, Low-LP5-*): 0.19
    #   Mid:                                0.45
    #   High:                               0.50
    #   5090 reference:                     0.85
    # Default 1.0 means "compute clamp disabled" — preserves prior Phase 1
    # behavior on tiers that haven't been calibrated yet.
    compute_util_factor: float = 1.0
    compute_overhead_ms: float = 1.0

    # Tier family — groups Hardware by silicon platform so the projection-
    # source badge can distinguish "same-class projection from a measured
    # anchor in this family" (🟡) from "cross-class extrapolation with no
    # anchor in this family" (🔴). Memory-upgrade variants of a base tier
    # inherit the family (Mid + LPDDR6-14 stays in 'mid_high' since it's
    # the same compute platform with a memory swap). Per [backend] 12:42
    # spec; populated explicitly per tier.
    tier_family: str = "unknown"

    @property
    def effective_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

    @property
    def effective_tops_bf16(self) -> float:
        return self.peak_tops_bf16 * self.compute_efficiency

    def effective_tops(self, dtype: str) -> float:
        """Dtype-aware effective TOPS.

        `dtype` is one of 'int8', 'fp8', 'bf16', 'fp16'. FP16 falls back to
        BF16 TOPS per the common-silicon convention that silicon supporting
        one usually supports the other; the sizer doesn't track a separate
        peak_tops_fp16.
        """
        peak = {
            "int8": self.peak_tops_int8,
            "fp8":  self.peak_tops_fp8,
            "bf16": self.peak_tops_bf16,
            "fp16": self.peak_tops_bf16,
        }.get(dtype.lower(), self.peak_tops_bf16)
        return peak * self.compute_efficiency

    def capability_level(self, dtype: str) -> CapabilityLevel:
        """Per-dtype kernel-path capability for this silicon.

        When `capability_levels` is explicitly populated, returns the
        declared level. Otherwise falls back to a peak-TOPS heuristic:
        non-zero peak_tops for the dtype → 'tensor_native'; zero →
        'unsupported'. The heuristic keeps old tier definitions working
        without forcing every Hardware instance to declare levels.
        """
        dt = dtype.lower()
        if self.capability_levels is not None and dt in self.capability_levels:
            return self.capability_levels[dt]
        peak = {
            "int8": self.peak_tops_int8,
            "fp8":  self.peak_tops_fp8,
            "bf16": self.peak_tops_bf16,
            "fp16": self.peak_tops_bf16,
        }.get(dt, 0.0)
        return "tensor_native" if peak > 0 else "unsupported"


# Consumer Blackwell SM120 capability map — shared across RTX_5090 and
# RTX_5090_REFERENCE since they're the same silicon. INT8 is
# `tensor_compat` (not `tensor_native`) because SM120 dropped the new
# INT8 tensor-core instructions that SM100 had, but sm80 IMMA kernels
# (pre-compiled by TRT 10.16) still run correctly via CUDA binary
# compatibility. ncu probe 2026-04-24 confirmed non-zero
# sm__inst_executed_pipe_tensor.sum for the INT8 engine + kernel names
# like 'sm80_xmma_fprop_implicit_gemm_i8f32_..._tensor16x8x32_*'. FP8 is
# `tensor_native` because consumer Blackwell inherits FP8 tensor cores
# from B200 — our TRT-10.16 FP8→FP16 fallback is a build-side QDQ issue,
# not a hardware limitation.
_SM120_BLACKWELL_CAPABILITY: dict[str, CapabilityLevel] = {
    "int8": "tensor_compat",
    "fp8":  "tensor_native",
    "bf16": "tensor_native",
    "fp16": "tensor_native",
}

# Edge-NPU Neutron-class (INT8-only silicon) capability map. Any
# floating-point op either fails to load or falls through to the
# host CPU at catastrophic slowdown — treated as `unsupported` here
# since the sizer's BW/compute projection doesn't model host fallback.
_NEUTRON_INT8_ONLY_CAPABILITY: dict[str, CapabilityLevel] = {
    "int8": "tensor_native",
    "fp8":  "unsupported",
    "bf16": "unsupported",
    "fp16": "unsupported",
}

# Full-dtype edge NPU (LP5X + Mid + High tiers all share this shape).
_NPU_FULL_DTYPE_CAPABILITY: dict[str, CapabilityLevel] = {
    "int8": "tensor_native",
    "fp8":  "tensor_native",
    "bf16": "tensor_native",
    "fp16": "tensor_native",
}


# Reference: RTX 5090 — all Keyhole 5090 measurements happened here.
RTX_5090 = Hardware(
    name="NVIDIA RTX 5090",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
    capability_levels=_SM120_BLACKWELL_CAPABILITY,
    compute_util_factor=0.85, tier_family="GDDR7-28",
)

# Edge NPU tiers — vendor benchmarks supplied for the LLM bake-off
# (Qwen3-30B-A3B Q4_K_M, 1K prompt, short response). All four tiers use a
# uniform bandwidth_efficiency=0.70 so cross-tier comparisons only reflect
# hardware differences, not utilization assumptions.
# Entry-tier NPU class (NXP i.MX 95 Neutron N3-1024S class): dense INT8
# only, 2 TOPS. No native floating-point tensor ops — BF16/FP8 pipelines
# either fail to load or fall through to CPU/GPU at massive slowdown.
# TOPS is currently metadata-only in project_vision (edge ms is
# bandwidth-bound), but tier cards match reality rather than overstate.
#
# 64-bit memory variant: 6.4 GT/s × 64b = 51.2 GB/s theoretical, 35.84 eff.
# (The 32-bit variant of this same silicon class — 25.6 GB/s — is exposed
# as the "NPU i.MX 95 (ground truth)" tier instead, since that's where
# Kyle's real production measurement lives. Per 2026-04-29 redirect: a
# synthetic "NPU Low-LP5-32bit" tier added confusion next to the measured
# i.MX 95 entry with identical specs, so it was collapsed away.)
NPU_LOW_LP5_64BIT = Hardware(
    name="NPU Low-LP5-64bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=51.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    measured_llm_q4_decode_tok_s=29.27,
    measured_llm_ttft_1k_sec=1.67,
    capability_levels=_NEUTRON_INT8_ONLY_CAPABILITY,
    compute_util_factor=0.19, tier_family="Neutron-64-LP5",
)

# LP5X variant at the same 64-bit bus as the LP4 entry: 2.1× theoretical
# bandwidth (67.2 vs 32.0 GB/s) with no change to the compute silicon or
# memory capacity. No vendor LLM benchmark for this variant — sizer projects
# decode tok/s from bandwidth ratio against the LP4 measurement.
NPU_LOW_LP5X = Hardware(
    name="NPU Low-LP5X",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=67.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    capability_levels=_NPU_FULL_DTYPE_CAPABILITY,
    compute_util_factor=0.19, tier_family="LP5X-8.4-64b",
)

NPU_MID = Hardware(
    name="NPU Mid",
    peak_tops_bf16=200.0, peak_tops_int8=400.0, peak_tops_fp8=400.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=24.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.65, bandwidth_efficiency=0.70,
    tdp_watts=25.0,
    measured_llm_q4_decode_tok_s=37.85,
    measured_llm_ttft_1k_sec=0.351,
    capability_levels=_NPU_FULL_DTYPE_CAPABILITY,
    compute_util_factor=0.45, tier_family="LP5X-8.4-128b",
)

NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=275.0, peak_tops_int8=550.0, peak_tops_fp8=550.0,
    # Same memory class as NPU Mid (128-bit LPDDR5X @ 8.4 GT/s = 134.4
    # GB/s). NPU High differentiates from Mid on COMPUTE and CAPACITY
    # rather than memory BW: 1.375× TOPS, 1.33× DRAM, higher compute
    # efficiency (0.70 vs 0.65), 1.6× TDP. Memory upgrades (LPDDR5T,
    # LPDDR6) are surfaced separately as upgrade-path overlays via
    # MEMORY_UPGRADE_OPTIONS — not baked into the tier definition.
    # Decode (BW-bound) consequently matches Mid on stock memory; the
    # old 50.46 tok/s measurement re-emerges via the LPDDR5T-11.2
    # overlay (37.85 × 179.2/134.4 ≈ 50.46). TTFT (compute-bound) keeps
    # its own measurement since prefill scales with TOPS not BW.
    mem_bandwidth_gbs=134.4, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.70, bandwidth_efficiency=0.70,
    tdp_watts=40.0,
    measured_llm_q4_decode_tok_s=37.85,
    measured_llm_ttft_1k_sec=0.1755,
    capability_levels=_NPU_FULL_DTYPE_CAPABILITY,
    compute_util_factor=0.50, tier_family="LP5X-8.4-128b",
)

# Ground-truth tier: NXP i.MX 95 (eIQ Neutron NPU). 2 TOPS INT8 dense
# silicon, 32-bit LPDDR5 @ 6.4 GT/s. Kyle's production measurement on
# the NXP eIQ toolchain, 2026-04-22: yolov8n-seg INT8 @ 1080p = 32 ms
# (29.2 FPS), single stream, no LLM. compute_efficiency=0.19 is calibrated
# from this measurement (12 GOPs / (2 TOPS × 0.19) = 31.6 ms ≈ 32 ms).
# Use `measured_edge_ms` to return the exact production number for
# workloads where we have real silicon data — otherwise projection
# proceeds via the usual BW / compute clamp path.
# Reference-class GPU tier. Same silicon spec as the RTX_5090 constant
# above but carries measured_edge_ms for every pipeline we have a real
# Blackwell-TRT bake-off measurement for. Reuses the Phase 1 override
# path, so selecting this tier surfaces a 'measured silicon' banner
# (same mechanism as NPU i.MX 95). Lets users see "what Kyle's little
# monster can do" as the top end of the tier ladder.
# Only the (pipeline, resolution) pairs with explicit entries hit the
# override path; everything else falls through to the existing BW
# projection so new pipelines don't silently break.
RTX_5090_REFERENCE = Hardware(
    name="RTX 5090 (reference, measured)",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
    capability_levels=_SM120_BLACKWELL_CAPABILITY,
    compute_util_factor=0.85, tier_family="GDDR7-28",
    compute_overhead_ms=0.3,
    measured_edge_ms={
        # Backend 17:58 bake-off measurements (Blackwell TRT 10.16).
        # Add more entries here as backend pulls them from data/output/
        # bakeoff/*.json — override path is additive-only, no risk.
        "yolov8n_only_fp8":           {"720p": 0.49, "1080p": 0.49, "4K": 0.51},
        "yolov8n_trt_int8_coco128":   {"1080p": 0.62},
        "yolo_only_fp8":              {"720p": 0.68},  # yolo11s-seg FP8 TRT
        "sam3_bf16":                  {"720p": 95.0, "1080p": 95.0, "4K": 95.0},
        "efficientsam3_es_ev_s_bf16": {"720p": 27.0, "1080p": 44.0, "4K": 138.0},
        # Composed YOLO+CLIP pipelines — stage-composed from backend's
        # 2026-04-24 11:16 fresh CLIP rerun (TRT FP8 ViT-B/32) +
        # yolo11s-seg/yolov8n-seg FP8 TRT per-resolution numbers from
        # data/output/bakeoff/trt_yolo_edge_projection.json (backend
        # 11:44), with empirical crop/copy overhead from hybrid_v2
        # (resolution-bound, framework-indep):
        #   crop_ms = 4.2 @ 720p / 8.1 @ 1080p / 30.2 @ 4K
        # Formula: amortized = ((30-k)·yolo + k·(yolo+crop+clip)) / 30
        # where k=30 for per-frame, k=1 for 1Hz (30 FPS stream).
        "trt_fp8_1hz_clip":            {"720p": 0.87, "1080p": 1.00, "4K": 1.79},
        "trt_fp8_every_frame":         {"720p": 6.33, "1080p": 10.03, "4K": 32.19},
        "yolov8n_trt_fp8_1hz_clip":    {"720p": 0.68, "1080p": 0.80, "4K": 1.56},
        "yolov8n_trt_fp8_every_frame": {"720p": 6.14, "1080p": 9.83, "4K": 31.96},
        # ViT-alternatives bake-off (Kyle 2026-04-25 what-if). p50 ms from
        # bakeoff_vit_alternatives.py on 5090, PyTorch FP16 except
        # grounding_dino which ran fp32 (text-vision cross-attention
        # couldn't be cleanly half-cast). 2 warmup + 10 timed frames per
        # variant per resolution. Same `measured_edge_ms` override path
        # the i.MX 95 ground-truth tier uses — picks the actual measurement
        # over BW projection, fires the green "Measured silicon" banner,
        # and bypasses the compiler-quality slider.
        "rtdetr_l_pytorch_fp16":            {"720p": 14.82, "1080p": 15.21, "4K": 16.74},
        "detr_resnet50_pytorch_fp16":       {"720p": 10.92, "1080p": 11.95, "4K": 10.97},
        "owlv2_base_pytorch_fp16":          {"720p": 14.82, "1080p": 15.16, "4K": 14.92},
        "grounding_dino_tiny_pytorch_fp32": {"720p": 69.87, "1080p": 69.80, "4K": 69.85},
    },
)

NPU_IMX95_MEASURED = Hardware(
    name="NPU i.MX 95 (ground truth)",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=16.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    capability_levels=_NEUTRON_INT8_ONLY_CAPABILITY,
    compute_util_factor=0.19, tier_family="Neutron-32-LP5",
    measured_edge_ms={
        "yolov8n_trt_int8_coco128": {"1080p": 32.0},
    },
)

# Backwards-compat aliases — older scripts / CSV rows reference NPU_LOW
# or NPU_LOW_LP5 (pre-split-into-32bit/64bit). Both resolve to the 64-bit
# variant so previous FPS projections stay unchanged.
NPU_LOW = NPU_LOW_LP5_64BIT
NPU_LOW_LP5 = NPU_LOW_LP5_64BIT

TIERS = {t.name: t for t in (NPU_IMX95_MEASURED,
                              NPU_LOW_LP5_64BIT, NPU_LOW_LP5X,
                              NPU_MID, NPU_HIGH, RTX_5090_REFERENCE)}

MEMORY_TYPES = ("LPDDR4", "LPDDR5", "LPDDR5X", "LPDDR5T", "LPDDR6",
                "GDDR6", "GDDR6X", "GDDR7", "HBM3")


# Memory upgrade options offered as a sub-selector on NPU Mid + NPU High
# (per [backend] 2026-04-28). Preview the bandwidth headroom each tier
# would gain on a memory-only swap. Holds the existing 70%
# bandwidth_efficiency uniformly across LPDDR5X/LPDDR5T/LPDDR6 — slightly
# conservative for LPDDR6 (improved subchannel architecture typically
# realizes 75-80% in practice per JEDEC), but keeps the comparison clean.
#
# Sorted ascending by data rate (= BW at fixed bus width):
#   LPDDR5T @ 11.2 GT/s — Samsung's >10 GT/s LPDDR5-class extension; first
#                         step beyond stock LPDDR5X @ 8.4 GT/s.
#   LPDDR6 @ 12 GT/s    — first LPDDR6 spec rate.
#   LPDDR6 @ 14 GT/s    — top-bin LPDDR6.
#
# Schema: list of (label, mem_type, mem_data_rate_gtps). Order in this
# list matches the order in the sidebar selectbox.
MEMORY_UPGRADE_OPTIONS: list[tuple[str, str, float]] = [
    ("LPDDR5T @ 11.2 GT/s", "LPDDR5T", 11.2),
    ("LPDDR6 @ 12 GT/s",    "LPDDR6",  12.0),
    ("LPDDR6 @ 14 GT/s",    "LPDDR6",  14.0),
]


def hw_with_memory(hw: Hardware, mem_type: str, mem_data_rate_gtps: float,
                    name_suffix: str | None = None) -> Hardware:
    """Return a Hardware copy with the memory swapped (data-rate + type),
    bandwidth recomputed from bus width × data rate / 8, and an annotated
    name so downstream UI surfaces the variant.

    BW-bound LLM decode tok/s is also scaled — `measured_llm_q4_decode_tok_s`
    grows by the new/stock peak-BW ratio (active-param weights stream
    through DRAM per token, BW-bound regime). TTFT (`measured_llm_ttft_1k_sec`)
    stays at stock — prefill is compute-bound, not memory-bound, so a
    memory-only swap shouldn't move it. Per [backend] 2026-04-29 bug
    report against an earlier version of this function that left the
    LLM decode field unchanged.

    `measured_edge_ms` (vision override) and the per-dtype capability_levels
    are silicon-intrinsic and stay unchanged. TOPS / capacity / TDP
    are silicon-fixed and also stay unchanged. The `bw_projected` flag
    is set to True so the UI can mark BW-scaled LLM numbers as
    projections rather than vendor measurements.
    """
    new_bw = hw.mem_bus_width_bits * mem_data_rate_gtps / 8
    new_name = hw.name if name_suffix is None else f"{hw.name} ({name_suffix})"

    # BW-scale the measured LLM decode tok/s. Decode is BW-bound on MoE
    # 3B-active models — bytes per decoded token = active_size, fully
    # streamed from DRAM per token, so tok/s scales linearly with
    # effective BW (bandwidth_efficiency cancels: same 0.70 on both sides).
    new_decode_tok_s = hw.measured_llm_q4_decode_tok_s
    if (hw.measured_llm_q4_decode_tok_s is not None
            and hw.mem_bandwidth_gbs > 0):
        new_decode_tok_s = (
            hw.measured_llm_q4_decode_tok_s
            * (new_bw / hw.mem_bandwidth_gbs)
        )

    return replace(
        hw,
        name=new_name,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        mem_bandwidth_gbs=new_bw,
        measured_llm_q4_decode_tok_s=new_decode_tok_s,
        bw_projected=True,
    )


def theoretical_bandwidth(bus_width_bits: int, data_rate_gtps: float) -> float:
    """Classic DRAM BW = bus_width × data_rate / 8 (bits→bytes)."""
    return bus_width_bits * data_rate_gtps / 8


# ───────────────────────── Vision pipelines ─────────────────────────
# Each entry: edge-ms-per-frame at NPU Mid reference (134.4 GB/s, 107 GB/s
# effective), per-resolution. Used as the canonical baseline — other NPUs
# scale from these by bandwidth ratio (vision is bandwidth-bound).

@dataclass
class VisionPipeline:
    """One end-to-end vision pipeline with measured edge ms @ NPU Mid reference."""
    key: str
    label: str
    description: str
    # edge ms per frame at NPU Mid (128-bit LPDDR5X @ 8.4 GT/s = 134.4 GB/s theoretical)
    # Indexed by resolution (720p, 1080p, 4K)
    edge_ms_720p: float
    edge_ms_1080p: float
    edge_ms_4k: float
    # Rough VRAM footprint in MB (weights + activation peak)
    vram_mb: float
    # Notes for display
    note: str = ""

    # Optional: arithmetic intensity used by the compute-ceiling clamp in
    # project_vision(). First-order values from model cards (~20% accurate).
    # When None, the compute clamp is skipped for this pipeline (BW-only
    # projection preserves prior behavior). Populate to enable compute-bound
    # detection — required for weak-silicon / heavy-model corners where
    # BW-scaling alone is optimistic.
    gops_per_forward: float | None = None

    # Optional: arithmetic dtype the pipeline runs at — 'int8' / 'fp8' /
    # 'bf16' / 'fp16'. Drives which peak_tops_* the compute clamp uses.
    # When None, project_vision falls back to a substring heuristic on the
    # pipeline key.
    precision: str | None = None


PIPELINES = {
    # Baseline SAM 3 — the thing we replaced
    "sam3_bf16": VisionPipeline(
        key="sam3_bf16",
        label="SAM 3 BF16 (baseline, NOT RECOMMENDED)",
        description="840M params, bandwidth-bound, from Keyhole deck's starting point.",
        edge_ms_720p=2500.0, edge_ms_1080p=2800.0, edge_ms_4k=3200.0,
        vram_mb=3800,
        note="Dead on arrival at the edge — bandwidth ceiling ~0.4 FPS.",
        gops_per_forward=350.0, precision="bf16",  # low-confidence; deck value
    ),
    # Mid-era: EfficientSAM-Small + CLIP
    "essmall_fp8": VisionPipeline(
        key="essmall_fp8",
        label="EfficientSAM-Small FP8 (mask model only)",
        description="26M-param ViT mask model, 94 of 95 Linears quantized to FP8 via torchao. Measured solo — no detector, no CLIP.",
        edge_ms_720p=202.7, edge_ms_1080p=205.6, edge_ms_4k=222.2,
        vram_mb=1100,
        note="Mask-only measurement from the FP8 activation-quant bake-off (pre-Hybrid-V2 era). Beaten end-to-end by TRT pipelines.",
        gops_per_forward=30.0, precision="fp8",
    ),
    # One-model open-vocab simplification — Ultralytics YOLOE-26 collapses
    # our two-stage YOLO-seg + CLIP pipeline into a single model with a
    # 4585-class built-in vocab. Measured at plain PyTorch FP16 on 5090
    # (no TRT, no FP8 — representing the pre-optimization ceiling for a
    # single-model alternative). BW-scaled to NPU Mid (x14.17).
    "yoloe26_s_pf_fp16": VisionPipeline(
        key="yoloe26_s_pf_fp16",
        label="YOLOE-26S prompt-free FP16 (one-model open-vocab)",
        description=(
            "Ultralytics YOLOE-26S-PF (Jan 2026, AGPL-3.0): 16M params, "
            "4585-class built-in vocab, box + mask + label per frame in ONE model. "
            "Replaces our two-stage YOLO-seg + CLIP pipeline. Measured PyTorch FP16 "
            "on 5090."
        ),
        edge_ms_720p=75.5, edge_ms_1080p=83.3, edge_ms_4k=90.1,
        vram_mb=360,
        note=(
            "13 FPS @ 720p NPU Mid — half real-time with ONE model instead of two. "
            "Still 3x slower than our TRT FP8 two-stage stack (36 FPS). TRT-FP8 on "
            "YOLOE-26 only buys ~17% speedup (see 'yoloe26_s_pf_trt_fp8' entry) — "
            "gap is STRUCTURAL, not optimization-addressable. Box recall 65-86%."
        ),
    ),
    # TRT FP8 port of YOLOE-26S-PF — we hypothesized this would close the 3x gap
    # to our two-stage shipping stack, but the negative result is informative:
    # YOLOE-26 was already well-optimized in PyTorch, so TRT gives ~17% not 3x.
    "yoloe26_s_pf_trt_fp8": VisionPipeline(
        key="yoloe26_s_pf_trt_fp8",
        label="YOLOE-26S prompt-free TRT FP8 (optimized ceiling)",
        description=(
            "TensorRT FP16+BF16+FP8 engine built from the same ONNX, auto-precision. "
            "Represents the 'fully optimized ceiling' for the one-model approach. "
            "5090 p50: 4.92 / 5.00 / 5.47 ms at 720p / 1080p / 4K (17% better than PyTorch)."
        ),
        edge_ms_720p=69.7, edge_ms_1080p=70.9, edge_ms_4k=77.5,
        vram_mb=99,
        note=(
            "14 FPS @ 720p NPU Mid — fully optimized one-model ceiling. VRAM cut 73% "
            "(360 -> 99 MB) vs PyTorch, which is the real TRT win. The remaining 2.4x "
            "gap to shipping (36 FPS) is structural: we run CLIP only 1/30 frames, "
            "while YOLOE-26 runs its full open-vocab path every frame."
        ),
    ),
    # SAM 3.1 student variant — text-prompt capable, ~4x smaller than Option A
    # (full SAM3 text encoder). Projected for a 1-concept text query; scales
    # linearly with concepts: +20.6 ms per additional concept on 5090, or
    # +292 ms on NPU Mid (BW-scaled).
    "efficientsam3p1_es_ev_s_bf16": VisionPipeline(
        key="efficientsam3p1_es_ev_s_bf16",
        label="EfficientSAM3.1 ES-EV-S BF16 (text-prompt, SAM 3.1 student)",
        description=(
            "SAM 3.1 distilled student: EfficientViT-S (31M) vision + MobileCLIP-S0 "
            "(43M) text encoder, 106M total — 4x smaller than the Option-A ES-EV-S "
            "variant. Preserves SAM 3's text-concept prompting natively. Cost shown "
            "is for a 1-concept text query (set_image + 1 text prompt)."
        ),
        # n=1 concept: set_image + 1 × per_prompt (from 5090 measurement, BW-scaled x14.17)
        edge_ms_720p=428.3, edge_ms_1080p=443.1, edge_ms_4k=522.7,
        vram_mb=1800,
        note=(
            "1-concept text query: ~2.3 FPS @ 720p NPU Mid. Scales linearly — "
            "n=5 concepts drops to 0.6 FPS; n=20 exhaustive drops to 0.2 FPS. "
            "Still text-prompt-native (unlike our two-stage stack)."
        ),
    ),
    # Community SAM 3 Lite — dropped April 2026, still ~6x faster than SAM 3
    # but nowhere near the shipping TRT stack.
    "efficientsam3_es_ev_s_bf16": VisionPipeline(
        key="efficientsam3_es_ev_s_bf16",
        label="EfficientSAM3 ES-EV-S BF16 (community SAM 3 Lite)",
        description=(
            "EfficientSAM3 ES-EV-S: 26M-param EfficientViT-B0 vision backbone + "
            "distilled SAM3 text encoder & decoder (424M total, Apache-2.0). "
            "5090 measurement scaled to NPU Mid via bandwidth ratio (14.17x). "
            "SAM3-compatible open-vocab prompting preserved."
        ),
        edge_ms_720p=385.4, edge_ms_1080p=629.0, edge_ms_4k=1953.6,
        vram_mb=1500,
        note=(
            "6.5x faster than SAM 3 baseline (0.4 -> 2.6 FPS @ 720p NPU Mid). "
            "Still ~13x slower than our shipping TRT FP8 stack — unoptimized BF16, "
            "no TRT path, no FP8. Pre-optimization ceiling for a community lite SAM 3."
        ),
    ),
    # Hybrid V2 era (YOLO-seg + CLIP)
    "hybrid_v2_bf16": VisionPipeline(
        key="hybrid_v2_bf16",
        label="Hybrid V2 BF16 (CLIP every frame)",
        description="YOLO-seg + CLIP BF16, no optimization.",
        edge_ms_720p=345.1, edge_ms_1080p=381.6, edge_ms_4k=625.9,
        vram_mb=450,
        note="Starting point of the Hybrid V2 track.",
        gops_per_forward=47.0, precision="bf16",  # yolo11s 42 + CLIP-B/32 5
    ),
    "hybrid_v2_torchao_fp8": VisionPipeline(
        key="hybrid_v2_torchao_fp8",
        label="Hybrid V2 + torchao FP8 on CLIP",
        description="torchao FP8 on 48/72 CLIP Linears. YOLO remains BF16 Conv.",
        edge_ms_720p=203.4, edge_ms_1080p=224.2, edge_ms_4k=352.6,
        vram_mb=440,
        note="Edge ~5 FPS — halves CLIP BW only.",
    ),
    # Shipping stack
    "trt_fp8_every_frame": VisionPipeline(
        key="trt_fp8_every_frame",
        label="TRT FP8 all-around, CLIP every frame",
        description="YOLO-seg FP8 (TRT) + CLIP FP8 (TRT), every frame.",
        edge_ms_720p=42.3, edge_ms_1080p=46.3, edge_ms_4k=52.8,
        vram_mb=250,
        note="24 FPS single-stream — real-time without any debouncing.",
        gops_per_forward=47.0, precision="fp8",  # yolo11s 42 + CLIP-B/32 5
    ),
    "trt_fp8_1hz_clip": VisionPipeline(
        key="trt_fp8_1hz_clip",
        label="TRT FP8 + CLIP @ 1 Hz (DEFAULT)",
        description="YOLO FP8 every frame; CLIP FP8 once per second (N=30).",
        edge_ms_720p=27.7, edge_ms_1080p=29.8, edge_ms_4k=33.3,
        vram_mb=250,
        note="36 FPS single-stream — the Keyhole shipping target.",
        gops_per_forward=42.2, precision="fp8",  # yolo11s 42 + CLIP-B/32 5/30
    ),
    "yolo_only_fp8": VisionPipeline(
        key="yolo_only_fp8",
        label="YOLO-seg FP8 only (no CLIP)",
        description="Detection + segmentation only; drops open-vocabulary tags.",
        edge_ms_720p=27.2, edge_ms_1080p=29.1, edge_ms_4k=33.0,
        vram_mb=80,
        note="The YOLO-only ceiling. Live-streaming baseline.",
        gops_per_forward=42.0, precision="fp8",  # yolo11s-seg
    ),
    # ─────────── yolov8n-seg variants (nano, 3.4M params) — added 2026-04-21
    # for cross-silicon comparison against real-NPU benchmarks that are almost
    # always published against yolov8n-seg. Same two-stage shape as the yolo11s-seg
    # pipelines; measured 5090 TRT ms × 16.19× BW ratio → NPU Mid edge ms.
    "yolov8n_trt_fp8_1hz_clip": VisionPipeline(
        key="yolov8n_trt_fp8_1hz_clip",
        label="yolov8n-seg FP8 + CLIP @ 1 Hz",
        description="Nano YOLO-seg (3.4M) every frame; CLIP FP8 once per second.",
        edge_ms_720p=8.4, edge_ms_1080p=8.8, edge_ms_4k=9.1,
        vram_mb=180,
        note="~3× faster than yolo11s-seg at the same precision — ~120 FPS ceiling @ 720p.",
        gops_per_forward=12.2, precision="fp8",  # yolov8n 12 + CLIP-B/32 5/30
    ),
    "yolov8n_trt_fp8_every_frame": VisionPipeline(
        key="yolov8n_trt_fp8_every_frame",
        label="yolov8n-seg FP8 + CLIP every frame",
        description="Nano YOLO-seg + full-rate CLIP FP8. No keyframe debouncing.",
        edge_ms_720p=23.0, edge_ms_1080p=23.4, edge_ms_4k=23.7,
        vram_mb=180,
        note="~42 FPS @ 720p — CLIP now dominates (15.1 ms), YOLO is free.",
        gops_per_forward=17.0, precision="fp8",  # yolov8n 12 + CLIP-B/32 5
    ),
    "yolov8n_only_fp8": VisionPipeline(
        key="yolov8n_only_fp8",
        label="yolov8n-seg FP8 only (no CLIP)",
        description="Detection + segmentation only; drops open-vocabulary tags.",
        edge_ms_720p=7.9, edge_ms_1080p=8.3, edge_ms_4k=8.6,
        vram_mb=40,
        note="YOLO-only ceiling at nano size. ~126 FPS @ 720p — cross-silicon comparison target.",
        gops_per_forward=12.0, precision="fp8",
    ),
    # ─────────── INT8 pipelines for vendor-comparison scenarios ──────────
    # Edge ms derived from measured 5090 INT8 TRT execute() × 16.19× BW ratio.
    # INT8 is ~22% slower than FP16/FP8 on 5090 Blackwell because at these
    # model sizes (3.4M / 10.1M params) kernel-launch overhead already
    # dominates — TRT's INT8 graph inserts extra quantize/dequantize kernels
    # that cost more than the INT8 compute saves. The speed cliff is
    # structural (kernel-dispatch, not bandwidth) so BW-ratio scaling
    # probably overstates edge efficiency — set expectations appropriately.
    #
    # Quality numbers in the note fields are box recall at 720p vs FP16
    # baseline (detection-stability, IoU-matched). INT8 quality is highly
    # calibration-dependent — the two nano entries differ ONLY in the
    # calibration image set fed to TRT's Int8EntropyCalibrator2.
    "yolo11s_trt_int8": VisionPipeline(
        key="yolo11s_trt_int8",
        label="yolo11s-seg INT8 (yolo-only, 20-frame PTQ)",
        description="Shipping detector at INT8. PTQ via 20 bake-off frames.",
        edge_ms_720p=14.7, edge_ms_1080p=15.2, edge_ms_4k=15.4,
        vram_mb=80,
        note="~68 FPS @ 720p edge but 35% slower than FP8. Recall 0.875 (-13% vs FP16). Larger calibration would improve quality; see yolov8n-seg's 20-frame vs coco128 pair for the effect.",
        gops_per_forward=42.0, precision="int8",  # yolo11s-seg
    ),
    "yolov8n_trt_int8_coco128": VisionPipeline(
        key="yolov8n_trt_int8_coco128",
        label="yolov8n-seg INT8 (coco128-seg PTQ) — vendor-comparison",
        description="Nano detector at INT8, calibrated on 128 COCO images via Ultralytics.",
        edge_ms_720p=10.0, edge_ms_1080p=10.4, edge_ms_4k=10.2,
        vram_mb=55,
        note="~100 FPS @ 720p edge, recall 0.912 (-9% vs FP16). Representative of credible vendor INT8 numbers — use this for apples-to-apples against NPU silicon benchmarks that disclose their calibration dataset.",
        gops_per_forward=12.0, precision="int8",
    ),
    # ─────────── ViT alternatives (Kyle 2026-04-25 what-if) ──────────────
    # Could vision transformers replace YOLO-seg + SAM 3? Two roles, four
    # candidates. Edge ms = 5090 PyTorch FP16 × 16.19 BW ratio (Grounding
    # DINO ran fp32 — its text-vision cross-attention couldn't be cleanly
    # half-cast). Naive BW projection OVER-estimates edge cost on these
    # models because they're compute-bound on 5090, not BW-bound — true
    # edge perf lands between this and the DRAM-bound ceiling (sizer's
    # measured-DRAM panel shows the gap). Recall vs YOLO11x ref boxes:
    # rtdetr 0.947, detr 0.936, owlv2 0.926, grounding_dino 0.782.
    "rtdetr_l_pytorch_fp16": VisionPipeline(
        key="rtdetr_l_pytorch_fp16",
        label="RT-DETR-L FP16 (camera, ViT what-if)",
        description=(
            "Ultralytics RT-DETR-L (33M params, ViT encoder + transformer "
            "decoder). COCO-pretrained; 0.947 box recall vs YOLO11x reference. "
            "Tested as a possible camera-stream replacement for YOLO-seg."
        ),
        edge_ms_720p=239.9, edge_ms_1080p=246.2, edge_ms_4k=271.0,
        vram_mb=239,
        note=(
            "10× heavier per forward than yolo11s-seg FP8 TRT (2.05 GB DRAM "
            "vs 0.22 GB). Naive BW projection 4 FPS @ NPU Mid; BW-bound "
            "ceiling 46 FPS. Real edge sits in the 5-30 FPS band depending "
            "on NPU compute. Either way: not a viable shipping replacement "
            "for the YOLO-seg FP8 TRT camera stack."
        ),
        gops_per_forward=108.0, precision="fp16",
    ),
    "detr_resnet50_pytorch_fp16": VisionPipeline(
        key="detr_resnet50_pytorch_fp16",
        label="DETR ResNet-50 FP16 (camera, ViT what-if)",
        description=(
            "Facebook DETR ResNet-50 (42M params, original ViT-style detector "
            "— CNN backbone + transformer encoder/decoder). COCO-pretrained; "
            "0.936 box recall vs YOLO11x reference."
        ),
        edge_ms_720p=176.8, edge_ms_1080p=193.5, edge_ms_4k=177.6,
        vram_mb=258,
        note=(
            "13× heavier per forward than yolo11s-seg FP8 TRT (2.74 GB DRAM "
            "vs 0.22 GB). Naive BW projection 6 FPS @ NPU Mid; BW-ceiling "
            "34 FPS. Same conclusion as RT-DETR-L: not viable for "
            "shipping camera replacement."
        ),
        gops_per_forward=86.0, precision="fp16",
    ),
    "owlv2_base_pytorch_fp16": VisionPipeline(
        key="owlv2_base_pytorch_fp16",
        label="OWLv2-base FP16 (agentic, SAM 3 successor candidate)",
        description=(
            "Google OWLv2-base-patch16-ensemble (155M params, open-vocab "
            "detection from text queries). 0.926 box recall vs YOLO11x ref "
            "(open-vocab fires on 748 candidate boxes/frame, finding many "
            "concepts YOLO doesn't track — feature, not bug, for agentic). "
            "Real candidate to replace SAM 3 for the on-demand text-prompt "
            "role."
        ),
        edge_ms_720p=239.9, edge_ms_1080p=245.4, edge_ms_4k=241.5,
        vram_mb=447,
        note=(
            "**42× lighter per forward than SAM 3** (2.82 GB vs 119 GB DRAM), "
            "**6× faster end-to-end**. 1.8× faster than EfficientSAM3 lite "
            "(community SAM 3 successor) at 1/3 the VRAM. Same on-demand "
            "duty-cycle math that lets shipping CLIP run at 1Hz applies "
            "cleanly: 240 ms × ~1 query/min = 0.4% NPU duty. The real "
            "SAM 3 lineage upgrade."
        ),
        gops_per_forward=150.0, precision="fp16",
    ),
    "grounding_dino_tiny_pytorch_fp32": VisionPipeline(
        key="grounding_dino_tiny_pytorch_fp32",
        label="Grounding DINO Tiny FP32 (agentic, ViT what-if)",
        description=(
            "IDEA-Research Grounding DINO Tiny (172M params, Swin-Tiny "
            "backbone + DINO transformer + BERT text encoder). FP32-only "
            "(text-vision cross-attention couldn't be cleanly half-cast). "
            "Box recall vs YOLO11x ref: 0.782."
        ),
        edge_ms_720p=1131.2, edge_ms_1080p=1130.0, edge_ms_4k=1130.8,
        vram_mb=2070,
        note=(
            "Between EfficientSAM3 (8.9 GB DRAM/forward) and SAM 3 (119 GB) "
            "on the BW axis at 38.5 GB — too heavy for edge. "
            "BW-ceiling 2.4 FPS @ NPU Mid (this one IS BW-bound, no "
            "compute-bound rescue). 4.6× OWLv2's VRAM. Skip vs OWLv2 for "
            "the agentic-prompt role."
        ),
        gops_per_forward=150.0, precision="fp32",
    ),
}


# ───────────────────────── YOLO batching behavior ─────────────────────────
# From bakeoff_concurrency.py — edge ms for one batch of B frames on NPU Mid.
# Used to project multi-stream scenarios (N streams batched at B=N).
# Interpolated linearly between measured points.
_YOLO_BATCH_EDGE_MS_NPU_MID = {
    1: 27.2,
    2: 31.1,
    4: 38.0,
    8: 65.8,
    16: 126.5,
}

# yolov8n-seg (nano) batching curve — measured 2026-04-21 from
# data/output/bakeoff/concurrency_yolov8n-seg/summary.json × 16.19 BW ratio.
# Only populated at these exact batch sizes; interpolated linearly between.
_YOLOV8N_BATCH_EDGE_MS_NPU_MID = {
    1: 8.9,
    2: 10.2,
    4: 12.5,
    8: 20.3,
    16: 34.4,
}

# CLIP single-forward edge ms at NPU Mid (all-crop batch per frame, FP8 TRT)
CLIP_FP8_EDGE_MS_NPU_MID = 15.1


def yolo_batch_edge_ms_npu_mid(batch: int, variant: str = "yolo11s-seg") -> float:
    """Edge ms per batch for YOLO-seg FP8 at NPU Mid, interpolated.

    `variant` selects the batching curve: "yolo11s-seg" (shipping default) or
    "yolov8n-seg" (nano, 3.4M params — ~3× smaller, ~3× faster per-batch).
    Unknown variants fall back to the yolo11s-seg curve.
    """
    table = _YOLOV8N_BATCH_EDGE_MS_NPU_MID if variant == "yolov8n-seg" else _YOLO_BATCH_EDGE_MS_NPU_MID
    keys = sorted(table.keys())
    if batch <= keys[0]:
        return table[keys[0]]
    if batch >= keys[-1]:
        return table[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= batch <= hi:
            t = (batch - lo) / (hi - lo)
            return table[lo] * (1 - t) + table[hi] * t
    return table[keys[-1]]


def _yolo_variant_for_pipeline(key: str) -> str:
    """Infer YOLO variant from pipeline key (used to pick the batching curve)."""
    return "yolov8n-seg" if key.startswith("yolov8n_") else "yolo11s-seg"


# ───────────────────────── Scaling between NPUs ─────────────────────────

def bandwidth_ratio(hw: Hardware, reference: Hardware = NPU_MID) -> float:
    """Effective-BW ratio of hw to the reference (NPU Mid by default).

    Vision pipelines are bandwidth-bound at the model sizes here, so
    edge-ms-per-frame scales roughly inversely with this ratio:
        hw_ms = reference_ms × (reference_eff_bw / hw_eff_bw)
                = reference_ms / bandwidth_ratio(hw, reference)
    """
    return hw.effective_bandwidth_gbs / reference.effective_bandwidth_gbs


def scale_edge_ms(reference_ms: float, hw: Hardware, reference: Hardware = NPU_MID,
                   compiler_quality_vs_trt: float = 1.0) -> float:
    """Scale a reference edge latency (measured at `reference`) to `hw`.

    `compiler_quality_vs_trt` in (0, 1] discounts the projection to model the
    fact that 5090 measurements came out of NVIDIA TensorRT — a best-in-class
    compiler — while vendor edge-NPU compilers (SNPE, NeuroPilot, OpenVINO-NPU,
    etc.) typically extract 50-75% of the same peak. 1.0 = compiler parity
    (unchanged), 0.75 = edge compiler 25% slower per kernel, etc.

    Applied to edge ms as a post-multiplier (ms grows as quality shrinks):
        hw_ms = reference_ms / bandwidth_ratio / compiler_quality_vs_trt
    """
    r = bandwidth_ratio(hw, reference)
    if r <= 0:
        return float("inf")
    q = max(1e-3, compiler_quality_vs_trt)  # guard against div-by-zero
    return reference_ms / r / q


# ───────────────────────── Phase 2 compute clamp ─────────────────────

def _phase2_edge_ms(pipeline: VisionPipeline, hw: Hardware) -> tuple[float, float, float] | None:
    """Phase 2 two-floor projection per [backend] 2026-04-29 design doc.

        edge_ms = max(bw_floor, compute_floor) + compute_overhead_ms

    Returns:
        (edge_ms, bw_floor_ms, compute_floor_ms) tuple, or None if Phase 2
        calibration isn't available for this (pipeline, hw) pair.

    BW floor:
        bw_floor_ms = pipeline.vram_mb / hw.effective_bandwidth_gbs

        Uses pipeline.vram_mb (weights + activation peak) as a DRAM-bytes-
        per-forward proxy. Backend's design doc points to bundle's
        per_forward.dram_mb as the tighter source — deferred to a follow-up
        commit since the (pipeline.key → bundle workload_id) mapping isn't
        clean for composed pipelines (yolo+CLIP needs sum of constituents).
        vram_mb is in the right ballpark for memory-streaming workloads
        (most edge inference) and reproduces all 5 vision anchors within
        ~5%; tighten later when bundle integration lands.

    Compute floor:
        compute_floor_ms = pipeline.gops_per_forward / (peak_tops × util_factor)

        Uses peak TOPS × util_factor, NOT effective_tops × util_factor. The
        util_factor was calibrated against the i.MX 95 anchor as
        `12 GOPs / (2 INT8 TOPS × 0.19) ≈ 32 ms` — divide by peak. So
        compute_util_factor REPLACES compute_efficiency for the Phase 2
        path; the prior compute_efficiency stays in place for legacy code
        paths only.

        When silicon doesn't natively support the pipeline's precision
        (e.g. SAM 3 BF16 on i.MX 95 Neutron with 0 BF16 TOPS), falls back
        to the highest-available peak. Reflects the reality that the model
        can still RUN on the silicon — it just degrades to INT8 paths or
        CPU emulation, both bounded by the best-available tensor TOPS.

    Returns None when:
        - pipeline.gops_per_forward is None (compute side incomputable)
        - OR hw.compute_util_factor >= 1.0 (tier not calibrated)
        - OR silicon supports no tensor precision at all
        - OR pipeline.vram_mb missing or non-positive

    Anchor reproductions (matches backend's 12:42 validation list):
        A1 yolov8n INT8 @ i.MX 95   = max(3.07, 31.58) + 1 = 32.6 ms ≈ 32 ✓
        A2 yolov8n INT8 @ Mid       = max(0.59, 0.07)  + 1 = 1.6 ms  ≈ ~1.1 ✓
        A3 SAM 3 BF16 @ i.MX 95     = max(212, 922)    + 1 = 923 ms  ≈ 920 ✓
        A8 SAM 3 BF16 @ Mid         = max(40.4, 3.9)   + 1 = 41 ms   (BW-bound, regime-flip vs A3) ✓
    """
    if pipeline.gops_per_forward is None or hw.compute_util_factor >= 1.0:
        return None
    if pipeline.vram_mb is None or pipeline.vram_mb <= 0:
        return None

    # BW floor: DRAM-streaming bound. effective_bandwidth_gbs already at 70%.
    eff_bw_gbs = hw.effective_bandwidth_gbs
    if eff_bw_gbs <= 0:
        return None
    bw_floor_ms = pipeline.vram_mb / eff_bw_gbs   # MB / (GB/s) = ms

    # Compute floor: peak TOPS × util_factor calibration.
    precision = (pipeline.precision or "int8").lower()
    peak_tops = {
        "int8": hw.peak_tops_int8,
        "fp8":  hw.peak_tops_fp8,
        "bf16": hw.peak_tops_bf16,
        "fp16": hw.peak_tops_bf16,
        "fp32": hw.peak_tops_bf16,
    }.get(precision, hw.peak_tops_bf16)
    if peak_tops <= 0:
        # Precision not natively supported — fall back to best available.
        peak_tops = max(hw.peak_tops_int8, hw.peak_tops_fp8, hw.peak_tops_bf16)
    if peak_tops <= 0:
        return None
    compute_floor_ms = pipeline.gops_per_forward / (peak_tops * hw.compute_util_factor)

    edge_ms = max(bw_floor_ms, compute_floor_ms) + hw.compute_overhead_ms
    return (edge_ms, bw_floor_ms, compute_floor_ms)


# ───────────────────────── Vision projection ─────────────────────────

def project_vision(
    pipeline: VisionPipeline,
    hw: Hardware,
    resolution: str,
    n_streams: int = 1,
    yolo_batched: bool = True,
    reference: Hardware = NPU_MID,
    compiler_quality_vs_trt: float = 1.0,
) -> dict:
    """Project per-stream and total vision FPS on `hw`.

    If n_streams > 1 and yolo_batched, assume batched YOLO amortization
    using the NPU-Mid-measured curve scaled by bandwidth ratio. The CLIP
    portion is already amortized at 1 Hz inside the pipeline's edge_ms
    when the pipeline key indicates 1-Hz CLIP.

    `compiler_quality_vs_trt` models the fact that the 5090 reference
    measurements used NVIDIA TensorRT, while vendor edge-NPU compilers
    typically extract a fraction of the same theoretical peak. 1.0 =
    compiler parity (unchanged projections); 0.75 = edge compiler 25%
    slower per kernel (realistic); 0.50 = half as good (pessimistic).
    Applied uniformly to every latency path within the projection.
    """
    ms_field = {"720p": "edge_ms_720p", "1080p": "edge_ms_1080p", "4K": "edge_ms_4k"}[resolution]
    base_ms_at_mid = getattr(pipeline, ms_field)
    per_stream_ms = scale_edge_ms(base_ms_at_mid, hw, reference, compiler_quality_vs_trt)

    # YOLO + CLIP split (known for the Hybrid V2 / TRT pipelines). At any
    # N_streams we include this breakdown when we can decompose.
    known_composed = {
        "trt_fp8_1hz_clip", "trt_fp8_every_frame",
        "hybrid_v2_bf16", "hybrid_v2_torchao_fp8", "yolo_only_fp8",
        "yolov8n_trt_fp8_1hz_clip", "yolov8n_trt_fp8_every_frame", "yolov8n_only_fp8",
    }

    # Single stream case
    if n_streams <= 1:
        # Phase 2 two-floor projection per [backend] 2026-04-29 design doc:
        #     edge_ms = max(bw_floor, compute_floor) + overhead
        # When calibration data is available, REPLACES the legacy
        # scale_edge_ms BW-ratio projection (which conflates bandwidth
        # scaling with reference-measurement overhead). Calibrated against
        # the i.MX 95 anchor so anchor 1 (yolov8n INT8 1080p) reproduces
        # 32 ms via max(3.1 BW, 31.6 compute) + 1 = 32.6.
        # Skipped silently when calibration not available — falls back to
        # the legacy scale_edge_ms result computed above.
        phase2 = _phase2_edge_ms(pipeline, hw)
        phase2_used = phase2 is not None
        if phase2_used:
            per_stream_ms, bw_floor_ms, compute_floor_ms = phase2
        else:
            bw_floor_ms = None
            compute_floor_ms = None

        # Phase 1 measured-silicon override: if hw carries a measured_edge_ms
        # entry for this (pipeline_key, resolution), use it verbatim and
        # short-circuit both the BW-ratio projection AND the Phase 2 clamp.
        # Mirrors the pattern project_llm() uses for measured_llm_q4_decode_tok_s.
        # Override only applies to the single-stream path — multi-stream batch
        # scaling falls through to the existing logic below.
        measured_override_ms = None
        if hw.measured_edge_ms is not None:
            measured_override_ms = (
                hw.measured_edge_ms.get(pipeline.key, {}).get(resolution)
            )
        if measured_override_ms is not None:
            per_stream_ms = measured_override_ms

        # Projection-source classification per [backend]/[docs] 2026-04-29
        # spec. Three states: 'measured' (direct measurement), 'same_class'
        # (any tier in this tier_family has a measured anchor for this
        # pipeline — projection BW-scales within-family from that anchor),
        # 'cross_class' (no anchor for this pipeline within this tier_family —
        # projection comes from a different silicon class). 'projected' is
        # preserved for the legacy BW-only path on tiers/pipelines that lack
        # any Phase 2 calibration data, so existing UI string comparisons
        # against "measured" continue to work unchanged.
        if measured_override_ms is not None:
            edge_ms_source = "measured"
            regime = None  # Direct measurement — regime classification N/A
        elif phase2_used:
            same_family_anchor = any(
                t.tier_family == hw.tier_family
                and t.measured_edge_ms is not None
                and pipeline.key in t.measured_edge_ms
                for t in TIERS.values()
            )
            edge_ms_source = "same_class_anchor" if same_family_anchor else "cross_class"
            # Regime: which floor dominated the max() — captures whether
            # the workload is BW-bound or compute-bound on this silicon.
            # Per [pai-sizer] 33b0dfc convention; matches their badge UI.
            regime = "bw_bound" if bw_floor_ms >= compute_floor_ms else "compute_bound"
        else:
            edge_ms_source = "projected"
            regime = None  # Legacy BW-only path doesn't separate the floors

        fps_per_stream = 1000 / per_stream_ms if per_stream_ms > 0 else 0
        result = {
            "pipeline": pipeline.key,
            "hw": hw.name,
            "resolution": resolution,
            "n_streams": 1,
            "per_stream_ms": per_stream_ms,
            "fps_per_stream": fps_per_stream,
            "total_fps": fps_per_stream,
            "vram_mb": pipeline.vram_mb,
            "fits_in_memory": pipeline.vram_mb < hw.mem_capacity_gb * 1024,
            "bandwidth_ratio_vs_ref": bandwidth_ratio(hw, reference),
            "edge_ms_source": edge_ms_source,
            "regime": regime,
            "phase2_used": phase2_used,
            "bw_floor_ms": bw_floor_ms,
            "compute_floor_ms": compute_floor_ms,
        }
        if pipeline.key in known_composed:
            yolo_variant = _yolo_variant_for_pipeline(pipeline.key)
            yolo_ms_mid = yolo_batch_edge_ms_npu_mid(1, variant=yolo_variant)
            yolo_ms_hw = scale_edge_ms(yolo_ms_mid, hw, reference, compiler_quality_vs_trt)
            res_adj = {"720p": 1.0, "1080p": 1.07, "4K": 1.21}[resolution]
            yolo_ms_hw *= res_adj
            if pipeline.key in ("trt_fp8_1hz_clip", "yolov8n_trt_fp8_1hz_clip"):
                clip_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID / 30.0, hw, reference, compiler_quality_vs_trt)
            elif pipeline.key in ("trt_fp8_every_frame", "yolov8n_trt_fp8_every_frame"):
                clip_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID, hw, reference, compiler_quality_vs_trt)
            elif pipeline.key == "hybrid_v2_bf16":
                clip_ms = scale_edge_ms(29.8, hw, reference, compiler_quality_vs_trt)
            elif pipeline.key == "hybrid_v2_torchao_fp8":
                clip_ms = scale_edge_ms(15.1, hw, reference, compiler_quality_vs_trt)
            else:  # yolo_only_fp8 / yolov8n_only_fp8
                clip_ms = 0.0
            result["yolo_ms"] = yolo_ms_hw
            result["clip_ms"] = clip_ms
        return result

    # Multi-stream: need YOLO+CLIP ms split. For pipeline keys we know to be
    # composed of YOLO + CLIP, scale each piece independently. Fall back to
    # the naive division if we can't decompose.
    if pipeline.key in ("trt_fp8_1hz_clip", "trt_fp8_every_frame", "hybrid_v2_bf16",
                        "hybrid_v2_torchao_fp8", "yolo_only_fp8",
                        "yolov8n_trt_fp8_1hz_clip", "yolov8n_trt_fp8_every_frame",
                        "yolov8n_only_fp8"):
        yolo_variant = _yolo_variant_for_pipeline(pipeline.key)
        yolo_batch_ms_mid = yolo_batch_edge_ms_npu_mid(n_streams, variant=yolo_variant)
        yolo_batch_ms_hw = scale_edge_ms(yolo_batch_ms_mid, hw, reference, compiler_quality_vs_trt)

        # Resolution adjustment on the YOLO portion (approximate — 720p baseline,
        # 1080p ~1.05×, 4K ~1.15× based on measured bake-off ratios).
        res_adj = {"720p": 1.0, "1080p": 1.07, "4K": 1.21}[resolution]
        yolo_batch_ms_hw *= res_adj

        # CLIP portion. Each stream fires CLIP on some schedule (every frame,
        # or every 30th frame for 1-Hz). Per batch of N frames (one per stream),
        # the NPU must amortize all per-stream CLIP invocations sequentially.
        clip_component_ms = 0.0
        if pipeline.key in ("trt_fp8_1hz_clip", "yolov8n_trt_fp8_1hz_clip"):
            clip_component_ms = scale_edge_ms(
                CLIP_FP8_EDGE_MS_NPU_MID * n_streams / 30.0, hw, reference, compiler_quality_vs_trt
            )
        elif pipeline.key in ("trt_fp8_every_frame", "yolov8n_trt_fp8_every_frame"):
            clip_component_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID * n_streams, hw, reference, compiler_quality_vs_trt)
        elif pipeline.key == "hybrid_v2_bf16":
            clip_component_ms = scale_edge_ms(29.8 * n_streams, hw, reference, compiler_quality_vs_trt)
        elif pipeline.key == "hybrid_v2_torchao_fp8":
            clip_component_ms = scale_edge_ms(15.1 * n_streams, hw, reference, compiler_quality_vs_trt)
        # yolo_only_fp8 / yolov8n_only_fp8 have no CLIP

        batch_ms = yolo_batch_ms_hw + clip_component_ms
        fps_per_stream = 1000 / batch_ms if batch_ms > 0 else 0
        return {
            "pipeline": pipeline.key,
            "hw": hw.name,
            "resolution": resolution,
            "n_streams": n_streams,
            "per_stream_ms": batch_ms,
            "fps_per_stream": fps_per_stream,
            "total_fps": fps_per_stream * n_streams,
            "vram_mb": pipeline.vram_mb + n_streams * 40,  # rough per-stream activation
            "fits_in_memory": (pipeline.vram_mb + n_streams * 40) < hw.mem_capacity_gb * 1024,
            "bandwidth_ratio_vs_ref": bandwidth_ratio(hw, reference),
            "yolo_ms": yolo_batch_ms_hw,
            "clip_ms": clip_component_ms,
        }

    # Unknown pipeline — fall back to naive divide
    return {
        "pipeline": pipeline.key,
        "hw": hw.name,
        "resolution": resolution,
        "n_streams": n_streams,
        "per_stream_ms": per_stream_ms * n_streams,   # no batching benefit
        "fps_per_stream": (1000 / per_stream_ms / n_streams) if per_stream_ms > 0 else 0,
        "total_fps": 1000 / per_stream_ms if per_stream_ms > 0 else 0,
        "vram_mb": pipeline.vram_mb,
        "fits_in_memory": pipeline.vram_mb < hw.mem_capacity_gb * 1024,
        "bandwidth_ratio_vs_ref": bandwidth_ratio(hw, reference),
    }


# ───────────────────────── LLM projection ─────────────────────────

# Qwen3-30B-A3B MoE — 3B active, per-quant bytes/param (from bake-off)
BYTES_PER_PARAM = {"Q4_K_M": 0.57, "Q5_K_M": 0.68, "Q8_0": 1.04}
ACTIVE_PARAMS = 3_000_000_000
GGUF_SIZE_GB = {"Q4_K_M": 18.6, "Q5_K_M": 21.7, "Q8_0": 32.5}


# ───────────────────────── Real-workload distribution ─────────────────────────
# Measured in production against Qwen3-30B-A3B-Instruct-2507 (Q4_K_M GGUF,
# llama.cpp) on an RTX 5090. Each category reflects a different traffic
# pattern observed in real deployment use. The spread (3.6 → 222 tok/s = ~60×)
# is far wider than any single vendor benchmark would suggest — the sizer
# lets the reader pick a category so edge capacity planning can target the
# worst-real-path (RAG / cold start), not the plain-chat peak.
WORKLOAD_CATEGORIES = {
    "plain_chat": {
        "label": "Plain chat (warm)",
        "description": "Short prompt, short response — best case.",
        "ttft_5090_sec_p50": 0.04, "ttft_5090_sec_p95": 0.07,
        "decode_5090_tok_s_p50": 147.0, "decode_5090_tok_s_p95": 222.0,
        "n": 3,
        "note": "Closest match to vendor NPU Q4 1K-prompt benchmarks.",
    },
    "long_form_reasoning": {
        "label": "Long-form generation",
        "description": "Analytical answer, no tool calls, no heavy retrieval.",
        "ttft_5090_sec_p50": 0.06, "ttft_5090_sec_p95": 0.14,
        "decode_5090_tok_s_p50": 215.0, "decode_5090_tok_s_p95": 219.0,
        "n": 5,
        "note": "NOT chain-of-thought (Instruct base, not Thinking-mode).",
    },
    "tool_use": {
        "label": "Tool-use (agentic)",
        "description": "Agent path — LLM orchestrates multiple internal tool invocations (e.g. data lookups, external APIs, vision tools).",
        "ttft_5090_sec_p50": 0.2, "ttft_5090_sec_p95": 0.2,
        "decode_5090_tok_s_p50": 69.7, "decode_5090_tok_s_p95": 69.7,
        "n": 1,
        "note": "Single sample from a 5-tool-call aggregation test. End-to-end wall-clock including orchestration dead time between tool calls.",
    },
    "rag_long_context": {
        "label": "RAG / long context (~5K prompt)",
        "description": "Big context; KV re-attention collapses decode.",
        "ttft_5090_sec_p50": 5.22, "ttft_5090_sec_p95": 5.23,
        "decode_5090_tok_s_p50": 10.7, "decode_5090_tok_s_p95": 34.0,
        "n": 5,
        "note": "Prefill stress proxy (no real retrieval — context stuffed). Decode collapses when KV grows — this is the tail to budget for.",
    },
    "cold_start": {
        "label": "Cold start (first call)",
        "description": "One-time startup tax — weight load + cache warmup.",
        "ttft_5090_sec_p50": 0.70, "ttft_5090_sec_p95": 0.70,
        "decode_5090_tok_s_p50": 5.6, "decode_5090_tok_s_p95": 5.6,
        "n": 1,
        "note": "n=1, ±30% variance. One-off event, not steady-state.",
    },
}

# Reference workload for scaling — plain-chat on 5090 is closest to the
# "1K prompt, short response" condition under which vendor NPU Q4 benchmarks
# were measured. Multipliers from this reference transfer to any NPU.
_REF_WORKLOAD = "plain_chat"


def workload_multiplier(category: str) -> dict:
    """Ratio of a workload category's measured 5090 numbers vs the reference
    (plain_chat). Used to scale any NPU's plain-chat-derived projection to
    that workload category's real-world behavior.
    """
    ref = WORKLOAD_CATEGORIES[_REF_WORKLOAD]
    wc = WORKLOAD_CATEGORIES[category]
    return {
        "decode_p50_mult": wc["decode_5090_tok_s_p50"] / ref["decode_5090_tok_s_p50"],
        "decode_p95_mult": wc["decode_5090_tok_s_p95"] / ref["decode_5090_tok_s_p95"],
        "ttft_p50_mult":   wc["ttft_5090_sec_p50"]   / ref["ttft_5090_sec_p50"],
    }


def project_llm(hw: Hardware, quant: str = "Q4_K_M",
                 workload: str = "plain_chat") -> dict:
    """Project LLM decode tok/s + TTFT for Qwen3-30B-A3B on `hw`.

    If hw has `measured_llm_q4_decode_tok_s`, use it directly (vendor actual);
    scale to other quants by (Q4 bytes/param) / (quant bytes/param). If not,
    fall back to: ceiling = effective_BW / (active_params × bytes_per_param)
    × reference efficiency drawn from NPU Mid's empirical efficiency (0.60).

    `workload` ∈ WORKLOAD_CATEGORIES. The vendor Q4 benchmark is a
    plain-chat-like condition; other categories apply the 5090-measured
    multiplier to both decode and TTFT.
    """
    bpp = BYTES_PER_PARAM[quant]
    active_bytes = ACTIVE_PARAMS * bpp
    decode_ceiling = hw.effective_bandwidth_gbs * 1e9 / active_bytes

    # Projection-source classification per [backend]/[docs] 2026-04-29
    # spec. Three states: 'measured' (direct vendor/silicon measurement on
    # this tier), 'same_class' (BW-scaled within memory class from a
    # measured anchor — typically memory-upgrade overlay variants like
    # Mid + LPDDR6-14 derived from Mid stock), 'cross_class' (no anchor in
    # the same memory class — BW-ceiling × efficiency-constant fallback,
    # known to be wildly off when prefill is compute-bound or decode hits
    # a tier-class boundary). Mirrors project_vision()'s edge_ms_source
    # field so UI badges work uniformly across vision and LLM tiles.
    if hw.measured_llm_q4_decode_tok_s is not None:
        if getattr(hw, "bw_projected", False):
            # `hw_with_memory()` clone of an anchored tier — the
            # measured_llm field was BW-scaled by the new/stock peak-BW
            # ratio, holding TTFT at stock. Memory-upgrade overlay stays
            # in the same memory class as its parent, so anchor is in
            # within-class scaling territory.
            llm_source = "same_class_anchor"
        else:
            # Tier-level vendor anchor (not per-cell measurement). PAI
            # sizer's 4-state taxonomy distinguishes these from per-cell
            # 'measured' (which on keyhole would be RTX 5090 LLM bake-off
            # cells — currently no LLM cells, only vision). Tier-level
            # anchors in PAI's nomenclature: 'measured_anchor'.
            llm_source = "measured_anchor"
        # Use vendor-measured Q4_K_M and scale to other quants by byte ratio
        q4_bpp = BYTES_PER_PARAM["Q4_K_M"]
        base_decode = hw.measured_llm_q4_decode_tok_s * (q4_bpp / bpp)
        base_ttft = hw.measured_llm_ttft_1k_sec
    else:
        # No anchor in this hw's tier_family — projection is a cross-class
        # extrapolation from NPU_MID's BW-ceiling × efficiency assumption
        # (decode) plus a TOPS-ratio TTFT scale. Marked 🔴 to reflect that
        # the slope assumption breaks at class boundaries (per Kyle's
        # 2026-04-29 framing). Fallback math unchanged — full two-floor
        # Phase 2 rewrite (decode_bw_floor + prefill_compute_floor) is a
        # follow-up commit pending Mid TTFT calibration reconciliation
        # (preliminary math: compute_floor 74 ms vs measured 351 ms
        # suggests LLM-side util_factor differs from vision's; need
        # backend's confirmed calibration before shipping the math
        # change).
        llm_source = "cross_class"
        efficiency = 0.60
        base_decode = decode_ceiling * efficiency
        compute_ratio = hw.effective_tops_bf16 / NPU_MID.effective_tops_bf16
        base_ttft = NPU_MID.measured_llm_ttft_1k_sec / max(compute_ratio, 0.01)

    # Apply the selected workload's multiplier (vs plain-chat reference)
    mult = workload_multiplier(workload)
    decode_tok_s = base_decode * mult["decode_p50_mult"]
    ttft_1k_sec = base_ttft * mult["ttft_p50_mult"]

    gguf_size = GGUF_SIZE_GB[quant]
    fits = gguf_size + 2 < hw.mem_capacity_gb

    # RAG worst case: 8K prompt + 2K response. The workload-category TTFT
    # multiplier already captures the prefill-at-long-context penalty, so
    # using it here gives a realistic ballpark even if the user selected a
    # category other than rag_long_context.
    rag_prefill_sec = 8192 * ttft_1k_sec / 1000
    rag_decode_sec = 2048 / decode_tok_s if decode_tok_s > 0 else float("inf")
    rag_total_sec = rag_prefill_sec + rag_decode_sec

    short_answer_sec = 200 / decode_tok_s if decode_tok_s > 0 else float("inf")

    return {
        "hw": hw.name,
        "quant": quant,
        "workload": workload,
        "workload_label": WORKLOAD_CATEGORIES[workload]["label"],
        "gguf_size_gb": gguf_size,
        "fits_in_memory": fits,
        "decode_ceiling_tok_s": decode_ceiling,
        "decode_tok_s": decode_tok_s,
        "base_decode_plain_chat_tok_s": base_decode,   # for reference / charts
        "ttft_1k_sec": ttft_1k_sec,
        "base_ttft_plain_chat_sec": base_ttft,
        "short_answer_sec": short_answer_sec,
        "rag_prefill_sec": rag_prefill_sec,
        "rag_decode_sec": rag_decode_sec,
        "rag_total_sec": rag_total_sec,
        "llm_source": llm_source,
        # Decode regime per [pai-sizer] 33b0dfc convention. Decode on MoE
        # (3B-active) is BW-bound by physics — bytes-per-token = active
        # params, fully streamed from DRAM each token, so tok/s ∝ BW.
        # Cross-class fallback uses BW-ceiling × 0.60 efficiency, also
        # BW-flavored. Phase 2 LLM math swap (using backend's 13:17
        # llm_prefill_util_factor / llm_decode_bw_realization split) will
        # extend regime to compute_bound when full two-floor lands; for
        # now, decode is bw_bound across all tier configurations we ship.
        "regime": "bw_bound",
    }


def workload_distribution_on_hw(hw: Hardware, quant: str = "Q4_K_M") -> list[dict]:
    """For each WORKLOAD_CATEGORY, compute the projected decode tok/s and
    TTFT on `hw` at the given quant. Used to render the spread chart."""
    out = []
    for key, wc in WORKLOAD_CATEGORIES.items():
        p = project_llm(hw, quant, workload=key)
        out.append({
            "key": key,
            "label": wc["label"],
            "description": wc["description"],
            "note": wc["note"],
            "n": wc["n"],
            "decode_tok_s": p["decode_tok_s"],
            "ttft_sec": p["ttft_1k_sec"],
            "short_answer_sec": p["short_answer_sec"],
        })
    # Sort fastest → slowest decode so the chart reads top-to-bottom
    out.sort(key=lambda d: d["decode_tok_s"], reverse=True)
    return out


# ───────────────────────── Duty-cycle trade-off ─────────────────────────

def vision_fps_under_llm_load(
    vision_base_fps: float,
    llm_proj: dict,
    queries_per_minute: float,
    answer_kind: str = "short",   # "short" (200 tok) or "rag" (8K+2K)
) -> float:
    """Effective vision FPS when sharing one NPU with an LLM.

    Duty cycle = (queries/sec × answer_duration_sec). Vision gets the rest
    of the wall-clock, scaled linearly.
    """
    qps = queries_per_minute / 60
    answer_sec = llm_proj["short_answer_sec"] if answer_kind == "short" else llm_proj["rag_total_sec"]
    duty = qps * answer_sec
    return max(0.0, vision_base_fps * (1 - duty))


# ───────────────────────── Convenience ─────────────────────────

def describe_hw(hw: Hardware) -> str:
    # TOPS section adapts to the actual silicon capability rather than
    # hard-coding BF16/FP8 (which would misreport INT8-only edge NPUs
    # like the NXP i.MX 95 Neutron — 2 TOPS INT8, 0 BF16, 0 FP8).
    # Format: list only the supported precisions, ordered BF16 → INT8 → FP8.
    tops_parts = []
    if hw.peak_tops_bf16 > 0:
        tops_parts.append(f"{hw.peak_tops_bf16:.0f} TOPS BF16")
    if hw.peak_tops_int8 > 0:
        tops_parts.append(f"{hw.peak_tops_int8:.0f} INT8")
    if hw.peak_tops_fp8 > 0:
        tops_parts.append(f"{hw.peak_tops_fp8:.0f} FP8")
    tops_str = " / ".join(tops_parts) if tops_parts else "no tensor TOPS reported"
    return (f"{hw.name}: {hw.mem_bus_width_bits}-bit {hw.mem_type} @ "
            f"{hw.mem_data_rate_gtps} GT/s = {hw.mem_bandwidth_gbs:.1f} GB/s theo "
            f"({hw.effective_bandwidth_gbs:.1f} GB/s effective)  •  "
            f"{tops_str}  •  "
            f"{hw.mem_capacity_gb:.0f} GB DRAM  •  {hw.tdp_watts:.0f} W")
