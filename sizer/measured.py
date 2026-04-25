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
