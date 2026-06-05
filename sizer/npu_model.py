"""
NPU sizing math — distilled from the Keyhole project's
`src/emulate/npu_emulator.py` plus the measured baselines captured during
the bake-off series. Self-contained: no Keyhole imports.

Every "measured" constant below traces back to a specific bake-off in the
Keyhole deck (see `scripts/bakeoff_*.py` in the parent project and
`REPRODUCE.md` for how to regenerate them).
"""
from __future__ import annotations

from dataclasses import dataclass

from ratchet import (
    Hardware,
    IMX95_MEASURED, NPU_LOW_LP5_64BIT, NPU_LOW_LP5X, NPU_MID, NPU_HIGH,
    RTX_5090_REFERENCE,
    hw_with_memory, MEMORY_UPGRADE_OPTIONS,
    hw_supports_dtype, hw_peak_tops_for_dtype,
)

from .precision import CapabilityLevel


# ───────────────────────── Hardware tiers (ratchet) ─────────────────────────
# Hardware, the tier instances, hw_with_memory, MEMORY_UPGRADE_OPTIONS, and the
# capability tables are now owned by ratchet (the shared SoC sizing engine).
# keyhole composes its VISIBLE video ladder from ratchet's canonical registry;
# surface-specific measurements (LLM anchors + vision edge-ms) are re-attached
# to these shared instances at import by sizer/measured.py
# (attach_keyhole_anchors_to_ratchet_tiers).

TIERS = {hw.name: hw for hw in (
    IMX95_MEASURED,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    NPU_HIGH,
    RTX_5090_REFERENCE,
)}

# Backwards-compat aliases (older scripts / CSV rows reference NPU_LOW /
# NPU_LOW_LP5 from before the LP5 32/64-bit split).
NPU_LOW = NPU_LOW_LP5_64BIT
NPU_LOW_LP5 = NPU_LOW_LP5_64BIT


# ─── Capability + measurement adapters ───
# ratchet's Hardware doesn't carry keyhole's capability_level() method,
# effective_tops_bf16 property, or measured_edge_ms field; these surface-side
# helpers bridge keyhole's call sites onto ratchet's equivalents.

# keyhole queries int8/fp8/bf16/fp16; ratchet conflates bf16/fp16 (+ adds q4_km).
_DTYPE_TO_RATCHET_CAP_KEY = {
    "int8": "int8", "fp8": "fp8", "bf16": "bf16/fp16", "fp16": "bf16/fp16",
    "nvfp4": "nvfp4", "fp4": "nvfp4", "mxfp4": "nvfp4",  # FP4 compute dtype (ratchet >=0.2.5)
}

# keyhole's catalog key for the Skippy MoE Q4 fine-tune — the implicit
# tier-level LLM anchor (was the legacy measured_llm_q4_decode_tok_s flat
# field; now in ratchet's measured_decode_overrides, re-attached by measured.py).
_MOE_KEY = "skippy_finetune"


def capability_level(hw: Hardware, dtype: str) -> CapabilityLevel:
    """Per-dtype capability string for keyhole, sourced from ratchet's canonical
    capability tables. ratchet's capability_levels is dict[str, CapabilityInfo]
    keyed int8/fp8/'bf16/fp16'/q4_km; its CapabilityLevel enum .value matches
    keyhole's string taxonomy. Falls back to a peak-TOPS heuristic when the
    tier declares no table."""
    cap_key = _DTYPE_TO_RATCHET_CAP_KEY.get(dtype.lower(), dtype.lower())
    caps = hw.capability_levels
    if caps is not None and cap_key in caps:
        return caps[cap_key].level.value
    peak = hw_peak_tops_for_dtype(hw, dtype)
    return "tensor_native" if peak > 0 else "unsupported"


def _measured_edge_ms(hw: Hardware, pipeline_key: str, resolution: str):
    """keyhole's measured edge-ms float for (pipeline, resolution), read from
    ratchet's measured_vision_overrides ({ms_per_inference, fps}); None if
    absent. Replaces direct hw.measured_edge_ms access (re-attached by
    measured.py)."""
    mvo = hw.measured_vision_overrides
    if not mvo:
        return None
    cell = mvo.get(pipeline_key, {}).get(resolution)
    return cell.get("ms_per_inference") if cell else None


def _has_vision_measurements(hw: Hardware) -> bool:
    """True when the tier carries any measured vision overrides."""
    return bool(hw.measured_vision_overrides)


def _get_measured(hw: Hardware, model_key: str, quant: str):
    """Per-cell LLM measurement (keyhole's model -> quant shape) on ratchet's
    Hardware. ratchet's get_measured_llm_cell is workload-keyed and unused here."""
    ml = hw.measured_llm
    if not ml:
        return None
    return ml.get(model_key, {}).get(quant)


# Importing measured runs attach_keyhole_anchors_to_ratchet_tiers() at module
# load, populating the ratchet tier instances above with keyhole's LLM + vision
# measurements before any projection. (measured.py imports only ratchet + stdlib
# — no circular import back into npu_model.)
from . import measured as _keyhole_measured  # noqa: E402,F401


# Edge-anchor cap calibration constant (per [backend] 2026-05-04 16:19).
# Used by the edge-anchor cap in project_vision to back-solve implied edge
# DRAM streaming from a slower-tier measurement: ref_bw_only = ref_ms - this.
# Sources: Low-LP5X back-solves to ~0.25-0.3 ms on ResNet-50; i.MX 95
# back-solves to ~0.4 on yolov8n. 0.3 is a reasonable single-constant fit.
# Per-tier-class refinement (Neutron 0.4 / LP5X 0.25 / Mid 0.2) is a future
# config-dict change if needed; 0.3 is the global default for now.
ASSUMED_EDGE_OVERHEAD_MS = 0.3

MEMORY_TYPES = ("LPDDR4", "LPDDR5", "LPDDR5X", "LPDDR5T", "LPDDR6",
                "GDDR6", "GDDR6X", "GDDR7", "HBM3")


# MEMORY_UPGRADE_OPTIONS and hw_with_memory are now owned by ratchet (imported
# at the top of this module). ratchet's hw_with_memory BW-scales the tier's
# measured_decode_overrides (where keyhole's MoE anchors now live, re-attached
# by measured.py) and holds prefill at stock — the same physics keyhole's local
# version applied to its flat fields. The 5090 measured_llm bundle isn't a
# memory-upgrade target (GDDR7, not an LPDDR swap), so it needs no scaling.


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

    # Per-forward DRAM bytes streamed (in MB) from the bundle's ncu profile.
    # Distinct from vram_mb (which is "weights + activation peak" footprint
    # used for capacity checks): this is the actual sustained streaming
    # quantity that Phase 2's BW floor needs. For cache-friendly workloads
    # (yolov8n / ResNet-50), the two are similar order of magnitude. For
    # heavy ViT models with attention layers (SAM 3, EfficientSAM3) the
    # streaming exceeds footprint by 30×+ due to cache thrashing — TLB
    # / L2 / SRAM misses force weights tiles to re-stream multiple times
    # per forward. Per [backend] 2026-04-29 12:39 audit, the bundle's
    # `per_forward.dram_mb` field is the canonical source. When populated,
    # _phase2_edge_ms uses this for the BW floor; when None, falls back
    # to vram_mb (back-compat for un-audited pipelines).
    dram_per_forward_mb: float | None = None


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
        # ncu-measured DRAM streaming on 5090: 118975 MB per forward
        # (bundle entry sam3_bf16_reference). Massively exceeds the 3.8 GB
        # footprint due to cache thrashing on the giant ViT-H backbone —
        # weight tiles re-stream multiple times per forward. Pre-2026-05-04,
        # Phase 2's BW floor used vram_mb=3800 which gave Mid + SAM 3 =
        # 18 FPS (faster than 5090's measured 10.5 FPS — physically
        # impossible). With ncu DRAM, Mid projects 0.6 FPS at default
        # 75% NPU share — physically consistent (5090 is ~13× the BW).
        dram_per_forward_mb=118975.0,
    ),
    # Mid-era: EfficientSAM-Small + CLIP
    # ── Bundle DRAM audit pass 2026-05-04 (per [backend] 13:37 mapping) ──
    # Each pipeline populated below with dram_per_forward_mb derived from
    # the bundle's per_forward.dram_mb (ncu profile on 5090). Heavy-ViT
    # entries had 10×+ vram-vs-streaming divergence due to attention
    # cache thrashing; CNN-class entries had streaming ≈ footprint and
    # don't strictly need this — populated for explicitness where the
    # bundle has a clean mapping.
    "essmall_fp8": VisionPipeline(
        key="essmall_fp8",
        label="EfficientSAM-Small FP8 (mask model only)",
        description="26M-param ViT mask model, 94 of 95 Linears quantized to FP8 via torchao. Measured solo — no detector, no CLIP.",
        edge_ms_720p=202.7, edge_ms_1080p=205.6, edge_ms_4k=222.2,
        vram_mb=1100,
        note="Mask-only measurement from the FP8 activation-quant bake-off (pre-Hybrid-V2 era). Beaten end-to-end by TRT pipelines.",
        gops_per_forward=30.0, precision="fp8",
        dram_per_forward_mb=34359.0,  # bundle efficientsam_small (precision unspecified; FP8 variant may stream less)
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
        dram_per_forward_mb=841.0,  # bundle yoloe26_pytorch_fp16
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
        dram_per_forward_mb=498.0,  # bundle yoloe26_trt_fp8
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
        dram_per_forward_mb=3886.0,  # bundle efficientsam3p1_es_ev_s__set_image (heavier path; text_prompt is 3575)
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
        dram_per_forward_mb=8934.0,  # bundle efficientsam3_es_ev_s
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
        dram_per_forward_mb=652.0,  # composed: yolo_seg_fp16_trt 219 + clip_trt 433
    ),
    "hybrid_v2_torchao_fp8": VisionPipeline(
        key="hybrid_v2_torchao_fp8",
        label="Hybrid V2 + torchao FP8 on CLIP",
        description="torchao FP8 on 48/72 CLIP Linears. YOLO remains BF16 Conv.",
        edge_ms_720p=203.4, edge_ms_1080p=224.2, edge_ms_4k=352.6,
        vram_mb=440,
        note="Edge ~5 FPS — halves CLIP BW only.",
        dram_per_forward_mb=650.0,  # composed: yolo_seg_fp8_trt 217 + clip_trt 433
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
        dram_per_forward_mb=650.0,  # composed: yolo_seg_fp8_trt 217 + clip_trt 433
    ),
    "trt_fp8_1hz_clip": VisionPipeline(
        key="trt_fp8_1hz_clip",
        label="TRT FP8 + CLIP @ 1 Hz (DEFAULT)",
        description="YOLO FP8 every frame; CLIP FP8 once per second (N=30).",
        edge_ms_720p=27.7, edge_ms_1080p=29.8, edge_ms_4k=33.3,
        vram_mb=250,
        note="36 FPS single-stream — the Keyhole shipping target.",
        gops_per_forward=42.2, precision="fp8",  # yolo11s 42 + CLIP-B/32 5/30
        dram_per_forward_mb=231.4,  # composed: yolo_seg_fp8_trt 217 + clip_trt/30 14.4
    ),
    "yolo_only_fp8": VisionPipeline(
        key="yolo_only_fp8",
        label="YOLO-seg FP8 only (no CLIP)",
        description="Detection + segmentation only; drops open-vocabulary tags.",
        edge_ms_720p=27.2, edge_ms_1080p=29.1, edge_ms_4k=33.0,
        vram_mb=80,
        note="The YOLO-only ceiling. Live-streaming baseline.",
        gops_per_forward=42.0, precision="fp8",  # yolo11s-seg
        dram_per_forward_mb=217.0,  # bundle yolo_seg_fp8_trt
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
        dram_per_forward_mb=120.4,  # composed: yolo_seg_yolov8n-seg_fp8_trt 106 + clip_trt/30 14.4
    ),
    "yolov8n_trt_fp8_every_frame": VisionPipeline(
        key="yolov8n_trt_fp8_every_frame",
        label="yolov8n-seg FP8 + CLIP every frame",
        description="Nano YOLO-seg + full-rate CLIP FP8. No keyframe debouncing.",
        edge_ms_720p=23.0, edge_ms_1080p=23.4, edge_ms_4k=23.7,
        vram_mb=180,
        note="~42 FPS @ 720p — CLIP now dominates (15.1 ms), YOLO is free.",
        gops_per_forward=17.0, precision="fp8",  # yolov8n 12 + CLIP-B/32 5
        dram_per_forward_mb=539.0,  # composed: yolo_seg_yolov8n-seg_fp8_trt 106 + clip_trt 433
    ),
    "yolov8n_only_fp8": VisionPipeline(
        key="yolov8n_only_fp8",
        label="yolov8n-seg FP8 only (no CLIP)",
        description="Detection + segmentation only; drops open-vocabulary tags.",
        edge_ms_720p=7.9, edge_ms_1080p=8.3, edge_ms_4k=8.6,
        vram_mb=40,
        note="YOLO-only ceiling at nano size. ~126 FPS @ 720p — cross-silicon comparison target.",
        gops_per_forward=12.0, precision="fp8",
        dram_per_forward_mb=106.0,  # bundle yolo_seg_yolov8n-seg_fp8_trt
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
        dram_per_forward_mb=217.0,  # bundle has no INT8 yolo entry; use yolo_seg_fp8_trt 217 as proxy (similar streaming)
    ),
    "yolov8n_trt_int8_coco128": VisionPipeline(
        key="yolov8n_trt_int8_coco128",
        label="yolov8n-seg INT8 (coco128-seg PTQ) — vendor-comparison",
        description="Nano detector at INT8, calibrated on 128 COCO images via Ultralytics.",
        edge_ms_720p=10.0, edge_ms_1080p=10.4, edge_ms_4k=10.2,
        vram_mb=55,
        note="~100 FPS @ 720p edge, recall 0.912 (-9% vs FP16). Representative of credible vendor INT8 numbers — use this for apples-to-apples against NPU silicon benchmarks that disclose their calibration dataset.",
        gops_per_forward=12.0, precision="int8",
        dram_per_forward_mb=106.0,  # bundle yolo_seg_yolov8n-seg_fp8_trt as proxy for INT8 (similar streaming)
    ),
    # ResNet-50v1 INT8 image classification — canonical vendor-comparison
    # benchmark for the 100-TOPS edge-NPU class. Added 2026-05-01 to support
    # Kyle's anchor measurement (1125 inf/s on Low-LP5X-class silicon).
    # Constant-time at 224×224 input regardless of camera resolution.
    "resnet50v1_int8_224": VisionPipeline(
        key="resnet50v1_int8_224",
        label="ResNet-50v1 INT8 (image classification, 224×224)",
        description="Standard ImageNet image-classification benchmark — 25.5M params, 4.1 GFLOPs/forward. Canonical vendor-comparison shape; constant-time at 224×224 input regardless of camera resolution.",
        # 5090 measured 0.325 ms (TRT INT8 PTQ, [backend] 4caa000). Mid
        # reference below = 5090 ms × BW ratio (1523/94 = 16.2) ≈ 5.3 ms
        # in pure BW projection — but since the workload is BW-bound on
        # Mid (Phase 2 max(BW=1.0, compute=0.046) + 1 ms = 2.0 ms), use
        # that as the canonical Mid edge_ms. Same value across all three
        # resolutions since the model input is fixed at 224×224.
        edge_ms_720p=2.0, edge_ms_1080p=2.0, edge_ms_4k=2.0,
        # vram_mb refined from 30 (rough estimate) to ncu-measured 94.15
        # per [backend] 4caa000 (RTX 5090 ncu profile, 10 forwards × 110
        # kernels). This is "DRAM bytes streamed per forward" via the
        # 5090's TRT engine with Blackwell's L2 cache behavior. May not
        # transfer perfectly to edge silicon with smaller caches — but
        # it's the most authoritative measurement we have, and Phase 2's
        # BW floor uses this number directly. Anchored cells (Low-LP5X +
        # 5090) are unaffected via Phase 1 override; un-anchored Mid/High
        # projections become more conservative (758 → ~500 FPS on Mid).
        vram_mb=94,
        note="Image classification (no bbox/seg head). 1000-class softmax output. Common vendor benchmark — useful as a 'compute-light, BW-modest' anchor on the 100-TOPS edge-NPU class.",
        gops_per_forward=4.1, precision="int8",
        dram_per_forward_mb=94.15,  # bundle resnet50_int8_trt__224
    ),
    # ─────────── 4-bit-weight CNN variants (Kyle 2026-05-14) ────────────
    # Added to unlock the spec's `{mid,high}_int8.{resnet50_w4,yolov8n_w4}`
    # anchor cells (per [docs] 2026-05-14 20:56 — `_w4`/`_w8` are
    # genuinely different weight bit-widths, not naming shorthand).
    # Compute path is still INT8; weights are 4-bit quantized. 5090-side
    # bake-offs for 4-bit weights don't exist yet ([backend] 21:03 noted
    # this as KH-P3-003 methodology-validation candidate, not urgent) —
    # the headline number for these pipelines on edge silicon comes from
    # the private NPU anchor secrets hot-swap when populated; falls back
    # to a Phase 2 BW-projection with halved DRAM streaming (4-bit weights
    # are ~half the bytes of 8-bit) when secrets is missing.
    "yolov8n_trt_int4_coco128": VisionPipeline(
        key="yolov8n_trt_int4_coco128",
        label="yolov8n-seg INT4 weights (coco128-seg PTQ)",
        description=(
            "Nano detector with 4-bit weights × INT8 compute. Same "
            "architecture as the 8-bit-weight INT8 variant; bytes-per-"
            "forward is roughly halved (only the weight stream shrinks; "
            "activations stay at INT8). Surfaces the spec's "
            "{mid,high}_int8.yolov8n_w4 anchor cells when secrets is "
            "populated."
        ),
        # Conservative placeholders — real numbers come from anchor
        # secrets hot-swap when populated. Halving the 8-bit-variant's
        # DRAM and using same compute (4-bit-weight × INT8-act doesn't
        # reduce compute, just memory traffic).
        edge_ms_720p=6.0, edge_ms_1080p=6.2, edge_ms_4k=6.0,
        vram_mb=28,
        note=(
            "Same architecture as yolov8n_trt_int8_coco128 — only the "
            "weight quant differs (4-bit vs 8-bit). Headline edge_ms on "
            "NPU Mid/High overrides to anchor when populated in "
            ".streamlit/secrets.toml (cnn_anchors.{mid,high}_int8.yolov8n_w4)."
        ),
        gops_per_forward=12.0, precision="int8",
        dram_per_forward_mb=53.0,  # ~half of 8-bit-weight 106 MB
    ),
    "resnet50v1_int4_224": VisionPipeline(
        key="resnet50v1_int4_224",
        label="ResNet-50v1 INT4 weights (image classification, 224×224)",
        description=(
            "ImageNet classifier with 4-bit weights × INT8 compute. Same "
            "25.5M-param architecture as the 8-bit-weight INT8 variant; "
            "DRAM streaming roughly halved due to compressed weights. "
            "Surfaces the spec's {mid,high}_int8.resnet50_w4 anchor "
            "cells when secrets is populated."
        ),
        edge_ms_720p=1.0, edge_ms_1080p=1.0, edge_ms_4k=1.0,
        vram_mb=47,
        note=(
            "Same architecture as resnet50v1_int8_224 — only the weight "
            "quant differs. Headline edge_ms on NPU Mid/High overrides "
            "to anchor when populated in .streamlit/secrets.toml "
            "(cnn_anchors.{mid,high}_int8.resnet50_w4)."
        ),
        gops_per_forward=4.1, precision="int8",
        dram_per_forward_mb=47.0,  # ~half of 8-bit-weight 94.15 MB
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
        dram_per_forward_mb=2052.0,  # bundle rtdetr_l__720p
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
        dram_per_forward_mb=2737.0,  # bundle detr__720p
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
        dram_per_forward_mb=2819.0,  # bundle owlv2__720p
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
        dram_per_forward_mb=38508.0,  # bundle grounding_dino__720p (note: 38 GB at fp32; weight stream + activations)
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

def _phase2_edge_ms(pipeline: VisionPipeline, hw: Hardware,
                     npu_share: float = 1.0) -> tuple[float, float, float] | None:
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
    # DRAM-per-forward streaming amount: prefer ncu-measured bundle data
    # (`pipeline.dram_per_forward_mb`) when populated, fall back to the
    # legacy `vram_mb` footprint estimate. The two diverge for heavy ViT
    # models where cache thrashing forces weight re-streaming (SAM 3
    # streams 119 GB/forward via ncu vs its 3.8 GB footprint). Per
    # [backend] 2026-04-29 audit, the bundle's per_forward.dram_mb is
    # the canonical source for sustained streaming.
    dram_mb = pipeline.dram_per_forward_mb if pipeline.dram_per_forward_mb else pipeline.vram_mb
    if dram_mb is None or dram_mb <= 0:
        return None

    # BW floor: DRAM-streaming bound. effective_bandwidth_gbs already at 70%
    # bandwidth_efficiency. NPU_share scales further down (1.0 = idle SoC,
    # 0.75 = typical contention, etc.). Compute floor is unaffected by
    # NPU_share — TOPS doesn't share the memory bus.
    eff_bw_gbs = hw.effective_bandwidth_gbs * max(npu_share, 1e-6)
    if eff_bw_gbs <= 0:
        return None
    bw_floor_ms = dram_mb / eff_bw_gbs   # MB / (GB/s) = ms

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
    npu_share: float | None = None,
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
    # NPU_share factor — applies to BW-bound paths only. None falls back
    # to the tier's npu_share_default (1.0 for 5090, 0.75 for NPU tiers).
    share = npu_share if npu_share is not None else hw.npu_share_default
    share = max(share, 1e-6)

    ms_field = {"720p": "edge_ms_720p", "1080p": "edge_ms_1080p", "4K": "edge_ms_4k"}[resolution]
    base_ms_at_mid = getattr(pipeline, ms_field)
    per_stream_ms = scale_edge_ms(base_ms_at_mid, hw, reference, compiler_quality_vs_trt)
    # Legacy scale_edge_ms is BW-projected, so npu_share scales it linearly
    # (slower wall-clock at lower share). Phase 2 path applies share inside
    # _phase2_edge_ms instead.
    if share < 1.0:
        per_stream_ms = per_stream_ms / share

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
        phase2 = _phase2_edge_ms(pipeline, hw, npu_share=share)
        phase2_used = phase2 is not None
        if phase2_used:
            per_stream_ms, bw_floor_ms, compute_floor_ms = phase2
        else:
            bw_floor_ms = None
            compute_floor_ms = None

        # Defensive 5090-anchored cap (per [backend] 2026-05-04 13:52
        # recommendation, Option 2 — refined to decompose BW + overhead).
        # When 5090 has a measured anchor for this (pipeline_key,
        # resolution) cell, the target tier's edge_ms cannot be faster
        # than what physics + measured-overhead allow.
        #
        # Backend's original formula was `5090_measured × BW_ratio` —
        # which over-clamps when 5090 is overhead-bound (small workloads
        # where 5090's measurement is mostly kernel-launch latency, not
        # BW-streaming). Example: ResNet-50 5090 measured 0.325 ms vs
        # 5090 BW floor 0.062 ms (5× overhead-padded). `5090 × BW_ratio`
        # propagates the overhead × 21.59 BW gap to Mid, yielding 7 ms
        # — wildly over-pessimistic since edge NPU overhead doesn't
        # scale with BW (kernel launch is per-launch, not per-byte).
        #
        # Decomposed formula:
        #   5090_bw_floor    = dram / 5090_eff_bw
        #   5090_overhead    = max(0, 5090_measured - 5090_bw_floor)
        #   target_bw_floor  = dram / (target_eff_bw × share)
        #   cap_ms           = target_bw_floor + 5090_overhead
        #
        # Right reading: target's BW-bound latency PLUS at-least-as-much
        # overhead as 5090 had. Edge NPU overhead is typically ≥ 5090's
        # (smaller scheduling surface, different ISA), so this is a
        # conservative LOWER bound on target_ms.
        # Skip for the 5090 itself.
        clamped_5090 = False
        ref_5090 = None
        if (phase2_used and hw.tier_family != "GDDR7-28"
                and _has_vision_measurements(RTX_5090_REFERENCE)
                and pipeline.dram_per_forward_mb is not None):
            ref_5090 = _measured_edge_ms(RTX_5090_REFERENCE, pipeline.key, resolution)
            if ref_5090 is not None:
                bw_5090_eff = (RTX_5090_REFERENCE.effective_bandwidth_gbs
                               * RTX_5090_REFERENCE.npu_share_default)
                bw_target_eff = hw.effective_bandwidth_gbs * share
                if bw_5090_eff > 0 and bw_target_eff > 0:
                    bw_floor_5090 = pipeline.dram_per_forward_mb / bw_5090_eff
                    overhead_5090 = max(0.0, ref_5090 - bw_floor_5090)
                    bw_floor_target = pipeline.dram_per_forward_mb / bw_target_eff
                    cap_ms = bw_floor_target + overhead_5090
                    if per_stream_ms < cap_ms:
                        # Mark as 'meaningfully clamped' only when cap raises
                        # the result by ≥10% — avoids labeling cases where
                        # 5090's overhead is small relative to target's BW
                        # floor (SAM 3: 17 ms 5090 overhead vs 1687 ms Mid
                        # BW floor = 1% nominal cap firing on a fundamentally
                        # BW-bound workload). The cap still applies as a
                        # safety net; just doesn't override the regime.
                        if cap_ms > per_stream_ms * 1.10:
                            clamped_5090 = True
                        per_stream_ms = cap_ms

        # Edge-anchor cap (per [backend] 2026-05-04 16:19). Mirror image of
        # the 5090 cap above: the 5090 cap is a LOWER bound on target_ms
        # (Phase 2 was too optimistic); this is an UPPER bound on target_ms
        # (Phase 2 was too pessimistic). Fires when a slower-BW edge tier
        # has a measured anchor for this (pipeline, resolution) cell — back-
        # solves the implied edge-streaming DRAM from the anchor and uses
        # it to bound the faster-BW target's projection.
        #
        # Background: pipeline.dram_per_forward_mb comes from 5090 ncu
        # profiles, which include Blackwell L2 cache-thrashing artifacts
        # that don't transfer to edge silicon with smaller caches. ResNet-50
        # 5090 streamed 94 MB; back-solving from Low-LP5X's 0.889 ms anchor
        # implies edge silicon streams ~28 MB (close to the 25 MB weight
        # footprint). Without this cap, Mid Phase 2 over-counts DRAM and
        # projects ResNet-50 at 428 FPS — slower than Low-LP5X's measured
        # 843 FPS, a physics violation flagged by Kyle 2026-05-04.
        #
        # Formula:
        #   ref_bw_only_ms       = max(0, ref_ms - ASSUMED_EDGE_OVERHEAD_MS)
        #   ref_implied_dram_mb  = ref_bw_only_ms × ref_eff_bw × ref_share
        #   target_bw_floor      = implied_dram / (target_eff_bw × share)
        #   cap_ms               = target_bw_floor + ASSUMED_EDGE_OVERHEAD_MS
        #
        # Robustness: implied_dram is clamped to min(implied, bundle_dram).
        # When the ref measurement is compute-bound (e.g. yolov8n on i.MX 95),
        # the naive back-solve grossly inflates implied DRAM; clamping to
        # bundle DRAM degrades the cap to a no-op rather than a wrong-direction
        # bound. When multiple refs apply, the tightest cap (smallest cap_ms)
        # wins — typically the BW-bound ref since it produces the smallest
        # implied DRAM.
        clamped_edge_anchor = False
        # Only fire when Phase 2 was BW-bound. Cap exists to fix BW-floor
        # over-estimation (5090 cache thrashing inflates bundle DRAM); it
        # has nothing useful to say when target is compute-bound. Without
        # this guard, a compute-bound i.MX 95 anchor for yolov8n could
        # spuriously inflate Low-LP5-64bit's projection from a real 31 ms
        # compute floor to a fake 4 ms BW-derived cap.
        phase2_bw_bound = (phase2_used and bw_floor_ms is not None
                           and compute_floor_ms is not None
                           and bw_floor_ms >= compute_floor_ms)
        if (phase2_bw_bound and pipeline.dram_per_forward_mb is not None):
            best_cap_ms = None
            for ref_hw in TIERS.values():
                if ref_hw is hw:
                    continue
                if ref_hw.tier_family == "GDDR7-28":
                    continue  # 5090 handled by separate cap above
                if not _has_vision_measurements(ref_hw):
                    continue
                ref_ms = _measured_edge_ms(ref_hw, pipeline.key, resolution)
                if ref_ms is None:
                    continue
                # Anchors are measured at 100% NPU share (no SoC contention
                # during benchmark) — mirrors the Phase 1 measured-override
                # convention `per_stream_ms = measured_override_ms / share`
                # which divides anchor by current share to model contention.
                # So ref_eff_bw is the silicon's max realized BW, NOT scaled
                # by npu_share_default.
                ref_eff_bw = ref_hw.effective_bandwidth_gbs
                target_eff_bw = hw.effective_bandwidth_gbs * share
                if ref_eff_bw <= 0 or target_eff_bw <= 0:
                    continue
                # Only slower-BW anchors bind faster-BW targets. Same-or-
                # faster ref doesn't constrain target's projection.
                if ref_eff_bw >= target_eff_bw:
                    continue
                ref_bw_only_ms = max(0.0, ref_ms - ASSUMED_EDGE_OVERHEAD_MS)
                ref_implied_dram_mb = ref_bw_only_ms * ref_eff_bw
                # Clamp to bundle DRAM — when ref is compute-bound, naive
                # implied_dram is inflated; bundle is the upper bound on
                # plausible streaming.
                ref_implied_dram_mb = min(ref_implied_dram_mb,
                                          pipeline.dram_per_forward_mb)
                target_bw_floor = ref_implied_dram_mb / target_eff_bw
                cap_ms = target_bw_floor + ASSUMED_EDGE_OVERHEAD_MS
                if best_cap_ms is None or cap_ms < best_cap_ms:
                    best_cap_ms = cap_ms
            if best_cap_ms is not None and per_stream_ms > best_cap_ms:
                # Threshold-gated regime label: only mark 'clamped' when
                # cap lowers Phase 2 result by ≥10% (FPS rises by ≥10%).
                # Mirrors the 5090 cap's threshold. Avoids label flips on
                # cells where Phase 2 was already in the right ballpark.
                if per_stream_ms > best_cap_ms * 1.10:
                    clamped_edge_anchor = True
                per_stream_ms = best_cap_ms

        # Phase 1 measured-silicon override: if hw carries a measured_edge_ms
        # entry for this (pipeline_key, resolution), use it verbatim and
        # short-circuit both the BW-ratio projection AND the Phase 2 clamp.
        # Mirrors the pattern project_llm() uses for measured_llm_q4_decode_tok_s.
        # Override only applies to the single-stream path — multi-stream batch
        # scaling falls through to the existing logic below.
        measured_override_ms = _measured_edge_ms(hw, pipeline.key, resolution)
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
            # Anchor was measured at 100% NPU share. At < 100%, the BW-
            # bound component scales linearly. Source stays 🟢 measured —
            # NPU_share is treated as an orthogonal axis (operating point)
            # from source (data pedigree). The tile-level "(@ X% NPU)"
            # marker is the visual signal that the user is looking at a
            # what-if. Mirrors [pai-sizer] e521a70 + matches [docs] 14:38
            # framing.
            if share < 1.0:
                per_stream_ms = measured_override_ms / share
            edge_ms_source = "measured"
            regime = None  # Direct measurement — regime classification N/A
        elif phase2_used:
            same_family_anchor = any(
                t.tier_family == hw.tier_family
                and pipeline.key in (t.measured_vision_overrides or {})
                for t in TIERS.values()
            )
            edge_ms_source = "same_class_anchor" if same_family_anchor else "cross_class"
            # Regime: which floor dominated the max() — captures whether
            # the workload is BW-bound or compute-bound on this silicon.
            # Per [pai-sizer] 33b0dfc convention; matches their badge UI.
            # Special case: if the 5090-anchored cap fired, the workload
            # is overhead-bound on 5090 reference — surface as a 4th regime
            # state per [backend] 2026-05-04 13:52 recommendation.
            if clamped_5090:
                regime = "overhead_bound_5090_clamped"
            elif clamped_edge_anchor:
                regime = "overhead_bound_edge_anchor_clamped"
            else:
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
            "npu_share": share,
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
                 workload: str = "plain_chat",
                 npu_share: float | None = None,
                 model_key: str | None = None) -> dict:
    """Project LLM decode tok/s + TTFT for the selected model on `hw`.

    `model_key` (added 2026-05-01) — when provided, project_llm consults
    `hw.measured_llm[model_key][quant]` first for richer per-(model, quant)
    anchors (e.g. RTX 5090 + Qwen 2.5 7B Q4 = 183.9 tok/s measured directly).
    When None, falls back to the legacy Skippy-MoE-Q4-implicit path via the
    flat `measured_llm_q4_decode_tok_s` field. Cross-app schema parity with
    PAI sizer's e69237b.

    `npu_share` ∈ (0, 1] — fraction of the NPU's effective DRAM bandwidth
    available to this workload, modeling SoC contention (display, camera,
    audio paths competing for the same memory bus). Scales BW-bound
    paths only (decode tok/s, prefill BW floor). Compute-bound paths
    (TTFT compute_floor) are unaffected — TOPS doesn't share the memory
    bus. When None, falls back to `hw.npu_share_default` (1.0 for 5090,
    0.75 for NPU tiers per [docs] 2026-04-29 14:38).

    Resolution order:
      1. hw.measured_llm[model_key][quant] → 🟢 measured (per-cell)
      2. hw.measured_llm_q4_decode_tok_s legacy path (Skippy-MoE-Q4 implicit) → 🟢 measured_anchor
         OR 🟡 same_class_anchor when hw.bw_projected is True (memory-upgrade variant)
      3. Cross-class fallback (two-floor max(BW, compute) + overhead) → 🟠 cross_class

    `workload` ∈ WORKLOAD_CATEGORIES. The vendor Q4 benchmark is a
    plain-chat-like condition; other categories apply the 5090-measured
    multiplier to both decode and TTFT.
    """
    bpp = BYTES_PER_PARAM[quant]
    active_bytes = ACTIVE_PARAMS * bpp
    # NPU_share factor — applies to all BW-bound paths below. Compute
    # floors are NOT scaled (TOPS doesn't share the memory bus).
    share = npu_share if npu_share is not None else hw.npu_share_default
    share = max(share, 1e-6)
    decode_ceiling = hw.effective_bandwidth_gbs * 1e9 / active_bytes

    # Projection-source classification per [backend]/[docs] 2026-04-29
    # spec. Resolution order:
    #   1. hw.measured_llm[model_key][quant] → 🟢 measured (per-cell anchor)
    #   2. hw.measured_llm_q4_decode_tok_s legacy → 🟢 measured_anchor / 🟡 same_class_anchor
    #   3. Cross-class fallback → 🟠 cross_class
    #
    # Phase 1a — per-(model, quant) anchor. Added 2026-05-01 via the
    # measured_llm dict (PAI's e69237b shape). Today this fires only on
    # RTX 5090 for the 5 Qwen + Skippy MoE Q4 cells; expandable as more
    # measurements come in.
    per_cell_anchor = None
    if model_key and hw.measured_llm and model_key in hw.measured_llm:
        per_cell_anchor = hw.measured_llm[model_key].get(quant)
    if per_cell_anchor is not None:
        llm_source = "measured" if not getattr(hw, "bw_projected", False) else "same_class_anchor"
        base_decode = per_cell_anchor["decode_tok_s"]
        # Derive TTFT @ 1K from prefill_tok_s if present; else fall back to
        # the tier's flat-field TTFT (best-available proxy when prefill
        # rate isn't supplied for this cell).
        prefill_rate = per_cell_anchor.get("prefill_tok_s")
        if prefill_rate and prefill_rate > 0:
            base_ttft = 1024.0 / prefill_rate
        else:
            # Fall back to the tier's MoE anchor prefill (migrated to ratchet's
            # measured_prefill_overrides) when this cell carries no prefill rate.
            _moe_prefill = (hw.measured_prefill_overrides or {}).get(_MOE_KEY)
            base_ttft = (1024.0 / _moe_prefill) if _moe_prefill else 0.0
        # NPU_share scaling on the BW-bound decode component. Source flag
        # stays 🟢 per orthogonal-axis convention (527fc9b); the tile
        # marker "(@ X% NPU)" surfaces the operating-point what-if.
        if share < 1.0:
            base_decode *= share
    elif (hw.measured_decode_overrides or {}).get(_MOE_KEY) is not None:
        # Tier-level Skippy-MoE-Q4 anchor — migrated from keyhole's legacy flat
        # fields to ratchet's measured_decode_overrides / measured_prefill_overrides
        # (re-attached by measured.py, keyed _MOE_KEY). Used as the implicit
        # tier anchor for any model, scaled to other quants by byte ratio.
        if getattr(hw, "bw_projected", False):
            # Memory-upgrade clone: ratchet's hw_with_memory BW-scaled the
            # measured_decode_overrides for the upgraded bandwidth (within-
            # class scaling), holding prefill at stock.
            llm_source = "same_class_anchor"
        else:
            # Tier-level vendor anchor (not a per-cell measurement) — PAI's
            # 4-state taxonomy calls this 'measured_anchor'.
            llm_source = "measured_anchor"
        q4_bpp = BYTES_PER_PARAM["Q4_K_M"]
        base_decode = hw.measured_decode_overrides[_MOE_KEY] * (q4_bpp / bpp)
        _moe_prefill = (hw.measured_prefill_overrides or {}).get(_MOE_KEY)
        base_ttft = (1024.0 / _moe_prefill) if _moe_prefill else 0.0
        # NPU_share scaling on the anchored path: anchor was measured at
        # 100% NPU access (idle SoC). At < 100%, the BW-bound decode
        # scales linearly. Source classification stays 🟢 measured_anchor
        # — NPU_share is treated as an orthogonal axis (operating point)
        # from source (data pedigree). The tile-level "(@ X% NPU)" marker
        # is the visual signal that the user is looking at a what-if.
        # Mirrors [pai-sizer] e521a70 + matches [docs] 14:38 framing
        # ("🟢 measured: NPU_share has no effect — the measurement was at
        # a specific real-world contention level baked in"). TTFT
        # (compute-bound) stays unchanged in either interpretation.
        if share < 1.0:
            base_decode *= share
    else:
        # No anchor in this hw's tier_family — Phase 2 cross-class two-floor
        # projection per [backend] 2026-04-29 13:17 calibration + 13:31
        # confirmation. Replaces the prior "BW × 0.60 efficiency + TOPS-
        # ratio TTFT" heuristic with first-principles physics:
        #
        #   decode_floor_ms_per_tok = active_params_GB / (eff_BW × decode_realization)
        #   prefill_compute_ms      = (prompt_tokens × gops_per_token) / (peak_tops × prefill_util)
        #   prefill_bw_ms           = active_params_GB / eff_BW   (weights load once)
        #   ttft_ms                 = max(prefill_bw, prefill_compute) + overhead
        #
        # Calibration constants per [backend] 13:17 + matches PAI sizer's
        # 0a5e94a:
        #   llm_prefill_util_factor: 0.10 default (Mid anchor calibrated;
        #     vision's 0.45 would under-predict by ~4.5×)
        #   llm_decode_bw_realization: 1.0 default (pure BW ceiling — Mid
        #     + MoE 0.66 realization is captured in the anchor itself, not
        #     extrapolated to other models)
        #
        # Anchor for first-principles math: gops_per_token ≈ 2 × active_params_billions
        # (matmul-bound forward, GPT-style transformer FLOP estimate).
        # Qwen3-30B-A3B = 2 × 3 = 6 GFLOPs/tok ≈ backend's 6.5 with attention.
        # ACTIVE_PARAMS is raw param count (3e9), so divide by 1e9 to get
        # billions, then × 2 for the matmul FLOP factor → 6 GFLOPs/token.
        # Unit balance: GFLOPs / (TFLOPs × util) → ms directly.
        llm_source = "cross_class"
        active_params_gb = active_bytes / 1e9     # GB streamed per decode token
        gops_per_token = 2 * (ACTIVE_PARAMS / 1e9)  # = 6 GFLOPs/tok for MoE 3B-active

        # Decode: BW-floor with realization factor (default 1.0 = pure
        # ceiling, anchor-driven elsewhere). NPU_share scales the
        # effective BW available to the workload.
        decode_bw_floor_tok_s = decode_ceiling * hw.llm_decode_bw_realization * share
        base_decode = decode_bw_floor_tok_s

        # Prefill (TTFT @ 1K): max(BW_load_floor, compute_floor) + overhead.
        # GFLOPs / (TOPS × util) gives ms; peak_tops_bf16 is in TOPS units
        # so the math is unit-balanced without further conversion.
        # Fallback for INT8-only silicon (e.g. i.MX 95 Neutron has 0 BF16
        # TOPS): drop to best-available tensor peak. Reflects that the
        # workload still RUNS — just degraded to INT8 path — with the
        # compute floor reflecting what the silicon CAN execute.
        # NPU_share scales the BW floor only; the compute floor is
        # unaffected since TOPS doesn't share the memory bus.
        peak_tops_compute = max(hw.peak_tops_bf16,
                                 hw.peak_tops_int8,
                                 hw.peak_tops_fp8,
                                 1e-9)
        prefill_compute_ms = (1024 * gops_per_token) / (peak_tops_compute * hw.llm_prefill_util_factor)
        prefill_bw_ms = (active_params_gb / (hw.effective_bandwidth_gbs * share)) * 1000
        ttft_ms = max(prefill_bw_ms, prefill_compute_ms) + hw.compute_overhead_ms
        base_ttft = ttft_ms / 1000  # ms → sec for downstream consumers

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
        "regime": "bw_bound",
        "npu_share": share,
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
