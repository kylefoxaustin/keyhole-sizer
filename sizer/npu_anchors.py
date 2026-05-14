"""Private NPU + CNN anchor loader.

Numbers live in Streamlit secrets (.streamlit/secrets.toml locally; Cloud
Secrets in production). This module exposes typed accessors with graceful
fallback when secrets aren't set (returns None → app falls back to
projection or shows 'not measured').

Bandwidth derivation: stored peak_bw_gbps × bw_share_frac × bw_efficiency_frac
gives the achieved bandwidth used to back out bytes-per-token. The
share_frac is overridable at call time so the UI's 100/75/50/25%
share-selector can re-derive on the fly without re-reading secrets.

Spec: personal-ai-framework `docs/private_anchor_secrets_spec.md`
(commit 65bf89c, schema-locked 2026-05-14). Mirrored on PAI sizer side
per [docs] 13:24 + 13:31 bus messages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


# Badge color by source — matches keyhole-sizer's _render_source_banner convention.
BADGE_FOR_SOURCE = {
    "measured":     "🟢",
    "vendor_spec":  "🟡",
    "projected":    "🟠",
    # unknown / missing source → no badge
}


@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    prefill_tokps: float
    mem_gb: float
    seqlen: int
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override: Optional[float] = None) -> float:
        """BW available to NPU, applying any UI share override."""
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac

    def bytes_per_token(self, share_override: Optional[float] = None) -> float:
        """Memory bytes moved per decoded token (BW-bound decode model)."""
        if self.tokps <= 0:
            return 0.0
        return self.achieved_bw_gbps(share_override) * 1e9 / self.tokps


@dataclass(frozen=True)
class CNNAnchor:
    ms_per_inference: float
    fps: float
    mem_mb: float
    input_res: str
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override: Optional[float] = None) -> float:
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac


def _try_get(section: str, sub: str, key: str) -> Optional[dict]:
    """Defensive .get-chain for st.secrets — returns None on any miss."""
    try:
        return dict(st.secrets[section][sub][key])
    except Exception:
        return None


def load_llm_anchor(tier: str, precision: str, model_key: str) -> Optional[LLMAnchor]:
    """tier in {'mid','high'}, precision in {'int8','fp'}, model_key e.g. 'qwen3_30b_a3b_moe'.

    Returns None if the entry isn't in secrets — caller falls back to projection.
    """
    sub = f"{tier}_{precision}"
    data = _try_get("npu_llm_anchors", sub, model_key)
    if data is None or data.get("tokps", 0) <= 0:
        return None
    return LLMAnchor(
        tokps=float(data["tokps"]),
        prefill_tokps=float(data.get("prefill_tokps", 0.0)),
        mem_gb=float(data.get("mem_gb", 0.0)),
        seqlen=int(data.get("seqlen", 0)),
        source=str(data.get("source", "")),
        measured_date=str(data.get("measured_date", "")),
        peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
        bw_share_frac=float(data.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(data.get("bw_efficiency_frac", 0.70)),
        notes=str(data.get("notes", "")),
    )


def load_cnn_anchor(tier: str, precision: str, cnn_key: str) -> Optional[CNNAnchor]:
    """tier in {'mid','high'}, precision in {'int8','fp'}, cnn_key e.g. 'resnet50_w4'."""
    sub = f"{tier}_{precision}"
    data = _try_get("cnn_anchors", sub, cnn_key)
    if data is None or data.get("ms_per_inference", 0) <= 0:
        return None
    fps = float(data.get("fps", 0.0))
    if fps <= 0 and data.get("ms_per_inference", 0) > 0:
        fps = 1000.0 / float(data["ms_per_inference"])
    return CNNAnchor(
        ms_per_inference=float(data["ms_per_inference"]),
        fps=fps,
        mem_mb=float(data.get("mem_mb", 0.0)),
        input_res=str(data.get("input_res", "")),
        source=str(data.get("source", "")),
        measured_date=str(data.get("measured_date", "")),
        peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
        bw_share_frac=float(data.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(data.get("bw_efficiency_frac", 0.70)),
        notes=str(data.get("notes", "")),
    )
