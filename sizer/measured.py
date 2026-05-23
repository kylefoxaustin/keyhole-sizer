"""
Loader + lookup for ncu measurements vendored from the keyhole repo.

The bundle JSON is produced by `keyhole/scripts/export_ncu_for_sizer.py`
and copied to `sizer/measured/sizer_bundle.json`. Each entry covers one
NVTX-wrapped workload (YOLO-seg FP8 TRT, CLIP TRT, MobileSAM, etc.) with
per-forward DRAM bytes that transfer 1:1 to any NPU bandwidth budget.

The PIPELINES dict in `npu_model.py` describes COMPOSITE pipelines
(YOLO-detect + CLIP-enrich at 1 Hz, etc.). The PIPELINE_TO_NCU map
below decomposes each pipeline into its measured components and how
often each fires per frame.

Usage:
    from sizer.measured import measured_dram_per_frame

    bytes_per_frame = measured_dram_per_frame("trt_fp8_1hz_clip")
    if bytes_per_frame is not None:
        ss_ddr_gbs_avg = bytes_per_frame * fps / 1e9
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


_BUNDLE_PATH = Path(__file__).resolve().parent / "sizer_bundle.json"


# ───────────────── Pipeline composition (sizer key → ncu components) ──
#
# Each list entry: (ncu_workload_id, fires_per_frame).
# Pipelines absent from this map have no measured backing yet — the
# CSV row falls back to the approximated values.

PIPELINE_TO_NCU: dict[str, list[tuple[str, float]]] = {
    # Shipping stack — YOLO-seg FP8 every frame + CLIP FP8 once per 30 frames
    "trt_fp8_1hz_clip":      [("yolo_seg_fp8_trt", 1.0), ("clip_trt", 1.0 / 30.0)],
    "trt_fp8_every_frame":   [("yolo_seg_fp8_trt", 1.0), ("clip_trt", 1.0)],
    "yolo_only_fp8":         [("yolo_seg_fp8_trt", 1.0)],
    # yolov8n-seg variants — same two-stage shape, nano YOLO + CLIP FP8.
    # ncu workload id encodes the variant so this mapping uses the
    # distinct yolo_seg_yolov8n-seg_fp8_trt range.
    "yolov8n_trt_fp8_1hz_clip":    [("yolo_seg_yolov8n-seg_fp8_trt", 1.0), ("clip_trt", 1.0 / 30.0)],
    "yolov8n_trt_fp8_every_frame": [("yolo_seg_yolov8n-seg_fp8_trt", 1.0), ("clip_trt", 1.0)],
    "yolov8n_only_fp8":            [("yolo_seg_yolov8n-seg_fp8_trt", 1.0)],
    # YOLOE-26 one-model variants
    "yoloe26_s_pf_fp16":     [("yoloe26_pytorch_fp16", 1.0)],
    "yoloe26_s_pf_trt_fp8":  [("yoloe26_trt_fp8", 1.0)],
    # EfficientSAM3 community Lite (mask-only, no detector counted here)
    "efficientsam3_es_ev_s_bf16": [("efficientsam3_es_ev_s", 1.0)],
    # essmall_fp8 in the sizer is a measurement of EfficientSAM-Small alone
    # in the FP8 era; the closest ncu sibling is the BF16 EfficientSAM-Small
    # contestant from sam_variants. Bandwidth-side they're comparable.
    "essmall_fp8":           [("efficientsam_small", 1.0)],
    # Hybrid V2 era — older, approximated via current YOLO + CLIP measurements
    "hybrid_v2_torchao_fp8": [("yolo_seg", 1.0), ("clip_trt", 1.0)],
    "hybrid_v2_bf16":        [("yolo_seg", 1.0), ("clip_trt", 1.0)],
    # SAM 3 BF16 reference — surgical kernel-replay capture (sam3_refs target).
    # The sizer's sam3_bf16 pipeline is detector-free (SAM 3 does its own
    # concept-matching), so the mapping is just the one NVTX range.
    "sam3_bf16":             [("sam3_bf16_reference", 1.0)],
    # EfficientSAM3.1 kernel-replay capture — two NVTX ranges per frame
    # (ViT encoder via __set_image, then decoder via __text_prompt).
    # Sum both for per-frame DRAM bytes.
    "efficientsam3p1_es_ev_s_bf16": [
        ("efficientsam3p1_es_ev_s__set_image",   1.0),
        ("efficientsam3p1_es_ev_s__text_prompt", 1.0),
    ],
    # ViT alternatives (Kyle 2026-04-25 what-if). 720p-only ncu sweep —
    # one DRAM-per-forward sample per variant, mirrors efficientsam3p1's
    # 720p-only scope-decision pattern.
    "rtdetr_l_pytorch_fp16":            [("rtdetr_l__720p", 1.0)],
    "detr_resnet50_pytorch_fp16":       [("detr__720p", 1.0)],
    "owlv2_base_pytorch_fp16":          [("owlv2__720p", 1.0)],
    "grounding_dino_tiny_pytorch_fp32": [("grounding_dino__720p", 1.0)],
}


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    """Load the vendored ncu bundle. Empty if file missing (graceful no-op)."""
    if not _BUNDLE_PATH.exists():
        return {"workloads": [], "missing_from_ncu": [], "known_gaps": []}
    return json.loads(_BUNDLE_PATH.read_text())


@lru_cache(maxsize=64)
def _workload_by_id(workload_id: str) -> Optional[dict]:
    for w in _load_bundle().get("workloads", []):
        if w["workload_id"] == workload_id:
            return w
    return None


def measured_dram_per_frame(pipeline_key: str) -> Optional[float]:
    """Per-frame DRAM bytes for a sizer pipeline, summed across components.

    Returns None if no NCU mapping exists for this pipeline (caller
    falls back to the approximation).
    """
    components = PIPELINE_TO_NCU.get(pipeline_key)
    if not components:
        return None
    total = 0.0
    for ncu_id, fires_per_frame in components:
        w = _workload_by_id(ncu_id)
        if w is None:
            return None  # partial mapping == no measured number
        total += w["per_forward"]["dram_bytes"] * fires_per_frame
    return total


def measured_components(pipeline_key: str) -> Optional[list[dict]]:
    """Per-component breakdown for diagnostic / CSV provenance.

    Returns list of {ncu_workload_id, fires_per_frame, dram_bytes_per_fire,
    n_forwards_in_bakeoff} or None if no mapping.
    """
    components = PIPELINE_TO_NCU.get(pipeline_key)
    if not components:
        return None
    out = []
    for ncu_id, fires in components:
        w = _workload_by_id(ncu_id)
        if w is None:
            return None
        out.append({
            "ncu_workload_id":     ncu_id,
            "fires_per_frame":     fires,
            "dram_bytes_per_fire": w["per_forward"]["dram_bytes"],
            "n_forwards_in_bakeoff": w["n_forwards"],
        })
    return out


def bundle_metadata() -> dict:
    """Provenance for CSV: timestamp, host, git SHA of the source sweep."""
    b = _load_bundle()
    return {
        "ncu_bundle_timestamp":  b.get("export_timestamp_iso", "unknown"),
        "ncu_measurement_host":  b.get("measurement_host", "unknown"),
        "ncu_n_workloads":       b.get("n_workloads", 0),
        "ncu_npu_tier_assumed":  b.get("npu_tier_used_for_projection", {}).get("name", "unknown"),
    }


# ─────────────── ratchet retrofit: attach keyhole anchors to ratchet tiers ───
#
# keyhole's per-tier LLM + vision measurements used to live inline on its own
# Hardware tier definitions. Now that the tiers come from ratchet's canonical
# registry (phase 3, Option C), this module re-attaches keyhole's
# surface-specific measurements onto ratchet's shared tier instances at import
# — the same pattern PAI sizer uses for its 5090 measurements. These numbers
# are keyhole-specific (vendor anchors, keyhole bake-offs) and deliberately do
# NOT enter ratchet's canonical registry.
from ratchet import (  # noqa: E402
    IMX95_MEASURED,
    NPU_HIGH,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    RTX_5090_REFERENCE,
)

# MoE catalog key in keyhole's llm_models.py (the legacy flat fields were the
# implicit Skippy-MoE-Q4 anchor; migrated here to ratchet's per-model dicts).
_MOE_KEY = "skippy_finetune"

# Tier-level Skippy-MoE-Q4 anchors — migrated from the legacy flat fields
# (measured_llm_q4_decode_tok_s / measured_llm_ttft_1k_sec). prefill_tok_s =
# 1024 / ttft_1k_sec (keyhole's 1K = 1024 tokens convention). Tuples
# (tier, decode_tok_s, prefill_tok_s) — ratchet's Hardware is an unhashable
# dataclass, so we can't key by instance.
_KEYHOLE_MOE_ANCHORS = [
    (NPU_LOW_LP5_64BIT,  29.27,  613.2),   # vendor anchor (NOT Skippy-specific)
    (NPU_MID,            37.85, 2917.4),
    (NPU_HIGH,           37.85, 5835.3),
    (RTX_5090_REFERENCE, 249.8, 6228.0),
]

# Genuine per-(model, quant) 5090 LLM bundle cells (NOT MoE-flat duplicates).
# keyhole's project_llm reads measured_llm[model_key][quant] (model → quant →
# cell). The MoE (skippy_finetune) anchor is intentionally absent here — it
# lives in measured_decode_overrides as the single canonical location.
_KEYHOLE_5090_MEASURED_LLM = {
    "qwen25_7b_dense": {
        "Q4_K_M": {"decode_tok_s": 183.9, "prefill_tok_s": 7226.0},
        "Q5_K_M": {"decode_tok_s": 170.0, "prefill_tok_s": 7215.0},
        "Q8_0":   {"decode_tok_s": 137.2, "prefill_tok_s": 7478.0},
    },
    "qwen25_32b_dense": {
        "Q4_K_M": {"decode_tok_s": 52.7, "prefill_tok_s": 1936.0},
        "Q5_K_M": {"decode_tok_s": 47.7, "prefill_tok_s": 1888.0},
    },
    "qwen25_14b_dense": {
        "Q4_K_M": {"decode_tok_s": 125.7, "prefill_tok_s": 5117.2},
    },
    "llama_3_1_8b_dense": {
        "Q4_K_M": {"decode_tok_s": 171.0, "prefill_tok_s": 10162.0},
    },
    "mistral_7b_v03_dense": {
        "Q4_K_M": {"decode_tok_s": 182.7, "prefill_tok_s": 10217.0},
    },
}

# keyhole vision measurements (ms-per-inference), keyed pipeline → resolution.
# Migrated from the per-tier `measured_edge_ms` fields to ratchet's
# `measured_vision_overrides` via the wrap-the-leaf transform below.
_KEYHOLE_VISION_MS = [
    (NPU_LOW_LP5X, {
        "yolov8n_trt_int8_coco128": {"1080p": 2.0},
        "resnet50v1_int8_224": {"720p": 0.889, "1080p": 0.889, "4K": 0.889},
    }),
    (IMX95_MEASURED, {
        "yolov8n_trt_int8_coco128": {"1080p": 32.0},
    }),
    (RTX_5090_REFERENCE, {
        "yolov8n_only_fp8":           {"720p": 0.49, "1080p": 0.49, "4K": 0.51},
        "yolov8n_trt_int8_coco128":   {"1080p": 0.62},
        "resnet50v1_int8_224":        {"720p": 0.325, "1080p": 0.325, "4K": 0.325},
        "yolo_only_fp8":              {"720p": 0.68},
        "sam3_bf16":                  {"720p": 95.0, "1080p": 95.0, "4K": 95.0},
        "efficientsam3_es_ev_s_bf16": {"720p": 27.0, "1080p": 44.0, "4K": 138.0},
        "trt_fp8_1hz_clip":            {"720p": 0.87, "1080p": 1.00, "4K": 1.79},
        "trt_fp8_every_frame":         {"720p": 6.33, "1080p": 10.03, "4K": 32.19},
        "yolov8n_trt_fp8_1hz_clip":    {"720p": 0.68, "1080p": 0.80, "4K": 1.56},
        "yolov8n_trt_fp8_every_frame": {"720p": 6.14, "1080p": 9.83, "4K": 31.96},
        "rtdetr_l_pytorch_fp16":            {"720p": 14.82, "1080p": 15.21, "4K": 16.74},
        "detr_resnet50_pytorch_fp16":       {"720p": 10.92, "1080p": 11.95, "4K": 10.97},
        "owlv2_base_pytorch_fp16":          {"720p": 14.82, "1080p": 15.16, "4K": 14.92},
        "grounding_dino_tiny_pytorch_fp32": {"720p": 69.87, "1080p": 69.80, "4K": 69.85},
    }),
]


def _wrap_edge_ms(ms_map: dict) -> dict:
    """{pipeline: {res: ms}} -> {pipeline: {res: {ms_per_inference, fps}}}."""
    return {
        pipeline: {
            res: {"ms_per_inference": ms, "fps": (1000.0 / ms if ms > 0 else 0.0)}
            for res, ms in res_map.items()
        }
        for pipeline, res_map in ms_map.items()
    }


def attach_keyhole_anchors_to_ratchet_tiers() -> None:
    """Re-attach keyhole's surface-specific measurements onto ratchet's tier
    instances. Idempotent; mutates the shared instances at import (the only
    sanctioned post-construction mutation, same as PAI's reference-tier attach).
    """
    # LLM tier-level MoE anchors → measured_decode/prefill_overrides (keyed
    # skippy_finetune). Augments any canonical entry (e.g. ratchet's NPU_MID
    # carries qwen3_30b_a3b_moe) — by-key lookup, so both coexist harmlessly.
    for tier, decode, prefill in _KEYHOLE_MOE_ANCHORS:
        tier.measured_decode_overrides = {
            **(tier.measured_decode_overrides or {}), _MOE_KEY: decode}
        tier.measured_prefill_overrides = {
            **(tier.measured_prefill_overrides or {}), _MOE_KEY: prefill}

    # 5090 per-(model, quant) bundle cells → measured_llm.
    RTX_5090_REFERENCE.measured_llm = {
        **(RTX_5090_REFERENCE.measured_llm or {}), **_KEYHOLE_5090_MEASURED_LLM}

    # Vision: wrap-the-leaf + MERGE into measured_vision_overrides. The merge
    # preserves ratchet's canonical entries (e.g. i.MX 95 yolov8n @ 1920x1080)
    # and adds keyhole's pipelines/resolutions; keyhole's vision code reads its
    # own 720p/1080p/4K keys.
    for tier, ms_map in _KEYHOLE_VISION_MS:
        merged = dict(tier.measured_vision_overrides or {})
        for pipeline, res_map in _wrap_edge_ms(ms_map).items():
            merged[pipeline] = {**merged.get(pipeline, {}), **res_map}
        tier.measured_vision_overrides = merged


attach_keyhole_anchors_to_ratchet_tiers()
