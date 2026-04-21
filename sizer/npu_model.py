"""
NPU sizing math — distilled from the Keyhole project's
`src/emulate/npu_emulator.py` plus the measured baselines captured during
the bake-off series. Self-contained: no Keyhole imports.

Every "measured" constant below traces back to a specific bake-off in the
Keyhole deck (see `scripts/bakeoff_*.py` in the parent project and
`REPRODUCE.md` for how to regenerate them).
"""
from __future__ import annotations

from dataclasses import dataclass, field


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

    @property
    def effective_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

    @property
    def effective_tops_bf16(self) -> float:
        return self.peak_tops_bf16 * self.compute_efficiency


# Reference: RTX 5090 — all Keyhole 5090 measurements happened here.
RTX_5090 = Hardware(
    name="NVIDIA RTX 5090",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
)

# Edge NPU tiers — vendor benchmarks supplied for the LLM bake-off
# (Qwen3-30B-A3B Q4_K_M, 1K prompt, short response). All four tiers use a
# uniform bandwidth_efficiency=0.70 so cross-tier comparisons only reflect
# hardware differences, not utilization assumptions.
NPU_LOW_LP4 = Hardware(
    name="NPU Low-LP4",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=32.0, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR4", mem_data_rate_gtps=4.0,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    measured_llm_q4_decode_tok_s=29.27,
    measured_llm_ttft_1k_sec=1.67,
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
)

NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=275.0, peak_tops_int8=550.0, peak_tops_fp8=550.0,
    mem_bandwidth_gbs=179.2, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=11.2,
    compute_efficiency=0.70, bandwidth_efficiency=0.70,
    tdp_watts=40.0,
    measured_llm_q4_decode_tok_s=50.46,
    measured_llm_ttft_1k_sec=0.1755,
)

# Backwards-compat alias — some older scripts / CSV rows still reference NPU_LOW.
NPU_LOW = NPU_LOW_LP4

TIERS = {t.name: t for t in (NPU_LOW_LP4, NPU_LOW_LP5X, NPU_MID, NPU_HIGH)}

MEMORY_TYPES = ("LPDDR4", "LPDDR5", "LPDDR5X", "LPDDR5T", "GDDR6", "GDDR6X", "GDDR7", "HBM3")


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


PIPELINES = {
    # Baseline SAM 3 — the thing we replaced
    "sam3_bf16": VisionPipeline(
        key="sam3_bf16",
        label="SAM 3 BF16 (baseline, NOT RECOMMENDED)",
        description="840M params, bandwidth-bound, from Keyhole deck's starting point.",
        edge_ms_720p=2500.0, edge_ms_1080p=2800.0, edge_ms_4k=3200.0,
        vram_mb=3800,
        note="Dead on arrival at the edge — bandwidth ceiling ~0.4 FPS.",
    ),
    # Mid-era: EfficientSAM-Small + CLIP
    "essmall_fp8": VisionPipeline(
        key="essmall_fp8",
        label="EfficientSAM-Small FP8 (mask model only)",
        description="26M-param ViT mask model, 94 of 95 Linears quantized to FP8 via torchao. Measured solo — no detector, no CLIP.",
        edge_ms_720p=202.7, edge_ms_1080p=205.6, edge_ms_4k=222.2,
        vram_mb=1100,
        note="Mask-only measurement from the FP8 activation-quant bake-off (pre-Hybrid-V2 era). Beaten end-to-end by TRT pipelines.",
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
    ),
    "trt_fp8_1hz_clip": VisionPipeline(
        key="trt_fp8_1hz_clip",
        label="TRT FP8 + CLIP @ 1 Hz (SHIPPING)",
        description="YOLO FP8 every frame; CLIP FP8 once per second (N=30).",
        edge_ms_720p=27.7, edge_ms_1080p=29.8, edge_ms_4k=33.3,
        vram_mb=250,
        note="36 FPS single-stream — the Keyhole shipping target.",
    ),
    "yolo_only_fp8": VisionPipeline(
        key="yolo_only_fp8",
        label="YOLO-seg FP8 only (no CLIP)",
        description="Detection + segmentation only; drops open-vocabulary tags.",
        edge_ms_720p=27.2, edge_ms_1080p=29.1, edge_ms_4k=33.0,
        vram_mb=80,
        note="The YOLO-only ceiling. Live-streaming baseline.",
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

# CLIP single-forward edge ms at NPU Mid (all-crop batch per frame, FP8 TRT)
CLIP_FP8_EDGE_MS_NPU_MID = 15.1


def yolo_batch_edge_ms_npu_mid(batch: int) -> float:
    """Edge ms per batch for YOLO-seg FP8 at NPU Mid, interpolated."""
    keys = sorted(_YOLO_BATCH_EDGE_MS_NPU_MID.keys())
    if batch <= keys[0]:
        return _YOLO_BATCH_EDGE_MS_NPU_MID[keys[0]]
    if batch >= keys[-1]:
        return _YOLO_BATCH_EDGE_MS_NPU_MID[keys[-1]]
    for lo, hi in zip(keys, keys[1:]):
        if lo <= batch <= hi:
            t = (batch - lo) / (hi - lo)
            return _YOLO_BATCH_EDGE_MS_NPU_MID[lo] * (1 - t) + _YOLO_BATCH_EDGE_MS_NPU_MID[hi] * t
    return _YOLO_BATCH_EDGE_MS_NPU_MID[keys[-1]]


# ───────────────────────── Scaling between NPUs ─────────────────────────

def bandwidth_ratio(hw: Hardware, reference: Hardware = NPU_MID) -> float:
    """Effective-BW ratio of hw to the reference (NPU Mid by default).

    Vision pipelines are bandwidth-bound at the model sizes here, so
    edge-ms-per-frame scales roughly inversely with this ratio:
        hw_ms = reference_ms × (reference_eff_bw / hw_eff_bw)
                = reference_ms / bandwidth_ratio(hw, reference)
    """
    return hw.effective_bandwidth_gbs / reference.effective_bandwidth_gbs


def scale_edge_ms(reference_ms: float, hw: Hardware, reference: Hardware = NPU_MID) -> float:
    """Scale a reference edge latency (measured at `reference`) to `hw`."""
    r = bandwidth_ratio(hw, reference)
    return reference_ms / r if r > 0 else float("inf")


# ───────────────────────── Vision projection ─────────────────────────

def project_vision(
    pipeline: VisionPipeline,
    hw: Hardware,
    resolution: str,
    n_streams: int = 1,
    yolo_batched: bool = True,
    reference: Hardware = NPU_MID,
) -> dict:
    """Project per-stream and total vision FPS on `hw`.

    If n_streams > 1 and yolo_batched, assume batched YOLO amortization
    using the NPU-Mid-measured curve scaled by bandwidth ratio. The CLIP
    portion is already amortized at 1 Hz inside the pipeline's edge_ms
    when the pipeline key indicates 1-Hz CLIP.
    """
    ms_field = {"720p": "edge_ms_720p", "1080p": "edge_ms_1080p", "4K": "edge_ms_4k"}[resolution]
    base_ms_at_mid = getattr(pipeline, ms_field)
    per_stream_ms = scale_edge_ms(base_ms_at_mid, hw, reference)

    # YOLO + CLIP split (known for the Hybrid V2 / TRT pipelines). At any
    # N_streams we include this breakdown when we can decompose.
    known_composed = {
        "trt_fp8_1hz_clip", "trt_fp8_every_frame",
        "hybrid_v2_bf16", "hybrid_v2_torchao_fp8", "yolo_only_fp8",
    }

    # Single stream case
    if n_streams <= 1:
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
        }
        if pipeline.key in known_composed:
            # YOLO single-stream edge ms at this HW (batch=1 curve)
            yolo_ms_mid = yolo_batch_edge_ms_npu_mid(1)
            yolo_ms_hw = scale_edge_ms(yolo_ms_mid, hw, reference)
            res_adj = {"720p": 1.0, "1080p": 1.07, "4K": 1.21}[resolution]
            yolo_ms_hw *= res_adj
            # CLIP contribution — see n_streams>1 branch for same breakdown
            if pipeline.key == "trt_fp8_1hz_clip":
                clip_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID / 30.0, hw, reference)
            elif pipeline.key == "trt_fp8_every_frame":
                clip_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID, hw, reference)
            elif pipeline.key == "hybrid_v2_bf16":
                clip_ms = scale_edge_ms(29.8, hw, reference)
            elif pipeline.key == "hybrid_v2_torchao_fp8":
                clip_ms = scale_edge_ms(15.1, hw, reference)
            else:  # yolo_only_fp8
                clip_ms = 0.0
            result["yolo_ms"] = yolo_ms_hw
            result["clip_ms"] = clip_ms
        return result

    # Multi-stream: need YOLO+CLIP ms split. For pipeline keys we know to be
    # composed of YOLO + CLIP, scale each piece independently. Fall back to
    # the naive division if we can't decompose.
    if pipeline.key in ("trt_fp8_1hz_clip", "trt_fp8_every_frame", "hybrid_v2_bf16",
                        "hybrid_v2_torchao_fp8", "yolo_only_fp8"):
        # YOLO portion (scales with batch size, then scale to target HW BW)
        yolo_batch_ms_mid = yolo_batch_edge_ms_npu_mid(n_streams)
        yolo_batch_ms_hw = scale_edge_ms(yolo_batch_ms_mid, hw, reference)

        # Resolution adjustment on the YOLO portion (approximate — 720p baseline,
        # 1080p ~1.05×, 4K ~1.15× based on measured bake-off ratios).
        res_adj = {"720p": 1.0, "1080p": 1.07, "4K": 1.21}[resolution]
        yolo_batch_ms_hw *= res_adj

        # CLIP portion. Each stream fires CLIP on some schedule (every frame,
        # or every 30th frame for 1-Hz). Per batch of N frames (one per stream),
        # the NPU must amortize all per-stream CLIP invocations sequentially.
        clip_component_ms = 0.0
        if pipeline.key == "trt_fp8_1hz_clip":
            # 1 Hz = each stream calls CLIP once per 30 frames. Per batch of N
            # frames, expected CLIP calls = N/30, so per-batch CLIP cost at the
            # NPU = (N/30) × 15.1 ms. Per-FRAME amortized cost is 0.5 ms,
            # independent of N — but per-BATCH cost scales with N, which is
            # what we add to batch_ms here.
            clip_component_ms = scale_edge_ms(
                CLIP_FP8_EDGE_MS_NPU_MID * n_streams / 30.0, hw, reference
            )
        elif pipeline.key == "trt_fp8_every_frame":
            # CLIP runs on every stream every frame — stays linear in N
            clip_component_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID * n_streams, hw, reference)
        elif pipeline.key == "hybrid_v2_bf16":
            clip_component_ms = scale_edge_ms(29.8 * n_streams, hw, reference)
        elif pipeline.key == "hybrid_v2_torchao_fp8":
            clip_component_ms = scale_edge_ms(15.1 * n_streams, hw, reference)
        # yolo_only_fp8 has no CLIP

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

    if hw.measured_llm_q4_decode_tok_s is not None:
        # Use vendor-measured Q4_K_M and scale to other quants by byte ratio
        q4_bpp = BYTES_PER_PARAM["Q4_K_M"]
        base_decode = hw.measured_llm_q4_decode_tok_s * (q4_bpp / bpp)
        base_ttft = hw.measured_llm_ttft_1k_sec
    else:
        # Fall back to NPU-Mid-class efficiency (~60% of BW ceiling)
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
    return (f"{hw.name}: {hw.mem_bus_width_bits}-bit {hw.mem_type} @ "
            f"{hw.mem_data_rate_gtps} GT/s = {hw.mem_bandwidth_gbs:.1f} GB/s theo "
            f"({hw.effective_bandwidth_gbs:.1f} GB/s effective)  •  "
            f"{hw.peak_tops_bf16:.0f} TOPS BF16 / {hw.peak_tops_fp8:.0f} FP8  •  "
            f"{hw.mem_capacity_gb:.0f} GB DRAM  •  {hw.tdp_watts:.0f} W")
