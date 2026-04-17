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
    bandwidth_efficiency: float = 0.80   # fraction of peak BW realized

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

# Three edge NPU tiers — vendor benchmarks supplied for the LLM bake-off
# (Qwen3-30B-A3B Q4_K_M, 1K prompt, short response).
NPU_LOW = Hardware(
    name="NPU Low",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=32.0, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR4", mem_data_rate_gtps=4.0,
    compute_efficiency=0.60, bandwidth_efficiency=0.75,
    tdp_watts=10.0,
    measured_llm_q4_decode_tok_s=29.27,
    measured_llm_ttft_1k_sec=1.67,
)

NPU_MID = Hardware(
    name="NPU Mid",
    peak_tops_bf16=200.0, peak_tops_int8=400.0, peak_tops_fp8=400.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=24.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.65, bandwidth_efficiency=0.80,
    tdp_watts=25.0,
    measured_llm_q4_decode_tok_s=37.85,
    measured_llm_ttft_1k_sec=0.351,
)

NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=275.0, peak_tops_int8=550.0, peak_tops_fp8=550.0,
    mem_bandwidth_gbs=179.2, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=11.2,
    compute_efficiency=0.70, bandwidth_efficiency=0.80,
    tdp_watts=40.0,
    measured_llm_q4_decode_tok_s=50.46,
    measured_llm_ttft_1k_sec=0.1755,
)

TIERS = {t.name: t for t in (NPU_LOW, NPU_MID, NPU_HIGH)}

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
        label="EfficientSAM-Small FP8 (torchao)",
        description="26M-param ViT mask model + CLIP; activation FP8 halves BW.",
        edge_ms_720p=202.7, edge_ms_1080p=205.6, edge_ms_4k=222.2,
        vram_mb=1100,
        note="Edge ~5 FPS. Second-generation; beaten by Hybrid V2.",
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

    # Single stream case
    if n_streams <= 1:
        fps_per_stream = 1000 / per_stream_ms if per_stream_ms > 0 else 0
        return {
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

        # CLIP portion — cost is N × single_clip_ms / N (each stream pays its own
        # CLIP invocation per frame or per Nth frame). For 1-Hz CLIP pipelines,
        # the CLIP amortization is already baked in; scale by bandwidth.
        clip_component_ms = 0.0
        if pipeline.key == "trt_fp8_1hz_clip":
            # 0.5 ms amortized per stream per frame × n_streams CLIP bursts at 1 Hz
            # They round-robin so the per-batch cost is ~N×0.5 ms, which spread over
            # N-frame batches = 0.5 ms/frame. Effectively constant.
            clip_component_ms = scale_edge_ms(CLIP_FP8_EDGE_MS_NPU_MID / 30.0, hw, reference)
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


def project_llm(hw: Hardware, quant: str = "Q4_K_M") -> dict:
    """Project LLM decode tok/s + TTFT for Qwen3-30B-A3B on `hw`.

    If hw has `measured_llm_q4_decode_tok_s`, use it directly (vendor actual);
    scale to other quants by (Q4 bytes/param) / (quant bytes/param). If not,
    fall back to: ceiling = effective_BW / (active_params × bytes_per_param)
    × reference efficiency drawn from NPU Mid's empirical efficiency (0.60).
    """
    bpp = BYTES_PER_PARAM[quant]
    active_bytes = ACTIVE_PARAMS * bpp
    decode_ceiling = hw.effective_bandwidth_gbs * 1e9 / active_bytes

    if hw.measured_llm_q4_decode_tok_s is not None:
        # Use vendor-measured Q4_K_M and scale to other quants by byte ratio
        q4_bpp = BYTES_PER_PARAM["Q4_K_M"]
        decode_tok_s = hw.measured_llm_q4_decode_tok_s * (q4_bpp / bpp)
        ttft_1k_sec = hw.measured_llm_ttft_1k_sec
    else:
        # Fall back to NPU-Mid-class efficiency (~60% of BW ceiling)
        efficiency = 0.60
        decode_tok_s = decode_ceiling * efficiency
        # TTFT: scale NPU Mid reference (0.351 s) by compute ratio
        compute_ratio = hw.effective_tops_bf16 / NPU_MID.effective_tops_bf16
        ttft_1k_sec = NPU_MID.measured_llm_ttft_1k_sec / max(compute_ratio, 0.01)

    gguf_size = GGUF_SIZE_GB[quant]
    fits = gguf_size + 2 < hw.mem_capacity_gb   # +2 GB KV/compute buffers

    # RAG worst case: 8K prompt + 2K response
    rag_prefill_sec = 8192 / (1000 / (ttft_1k_sec * 1000))   # tok_s = 1000 / (ttft × 1000) … simplify
    rag_prefill_sec = 8192 * ttft_1k_sec / 1000
    rag_decode_sec = 2048 / decode_tok_s if decode_tok_s > 0 else float("inf")
    rag_total_sec = rag_prefill_sec + rag_decode_sec

    short_answer_sec = 200 / decode_tok_s if decode_tok_s > 0 else float("inf")

    return {
        "hw": hw.name,
        "quant": quant,
        "gguf_size_gb": gguf_size,
        "fits_in_memory": fits,
        "decode_ceiling_tok_s": decode_ceiling,
        "decode_tok_s": decode_tok_s,
        "ttft_1k_sec": ttft_1k_sec,
        "short_answer_sec": short_answer_sec,
        "rag_prefill_sec": rag_prefill_sec,
        "rag_decode_sec": rag_decode_sec,
        "rag_total_sec": rag_total_sec,
    }


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
