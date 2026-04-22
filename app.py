"""
keyhole-sizer — interactive NPU sizing sandbox for the Keyhole bake-off findings.

Launch:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sizer.npu_model import (
    Hardware, TIERS, MEMORY_TYPES, PIPELINES, NPU_MID,
    WORKLOAD_CATEGORIES, BYTES_PER_PARAM,
    describe_hw, project_vision, project_llm,
    theoretical_bandwidth, vision_fps_under_llm_load,
    workload_distribution_on_hw,
)
from sizer.platform_budget import (
    vision_workload_row, llm_workload_row, rows_to_csv_str,
)
from sizer.measured import (
    measured_dram_per_frame, measured_components, bundle_metadata,
)
from sizer.kpi_breakdown import (
    pipeline_kpi_row, all_pipeline_kpi_rows, kpi_rows_to_xlsx,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _full_matrix_csv() -> str:
    """Generate the full preset-HW × pipeline × resolution × stream-count matrix,
    plus every LLM (quant × workload × answer_kind) combo. Cached for an hour.
    Same output as `scripts/export_platform_matrix.py`."""
    rows: list[dict] = []
    for hw in TIERS.values():
        for pipeline in PIPELINES.values():
            for res in ("720p", "1080p", "4K"):
                for n in (1, 2, 4, 8, 16):
                    try:
                        rows.append(vision_workload_row(pipeline, hw, res, n_streams=n))
                    except Exception:
                        pass
    for hw in TIERS.values():
        for quant in BYTES_PER_PARAM:
            for workload in WORKLOAD_CATEGORIES:
                for answer_kind in ("short", "rag"):
                    try:
                        rows.append(llm_workload_row(
                            hw, quant, workload=workload,
                            queries_per_minute=2.0, answer_kind=answer_kind,
                        ))
                    except Exception:
                        pass
    return rows_to_csv_str(rows)

st.set_page_config(
    page_title="keyhole-sizer",
    page_icon="🎯",
    layout="wide",
)


# ───────────────────────── Shared-password gate ─────────────────────────
# Password read from Streamlit Cloud secrets (secrets.toml has PASSWORD="...").
# If no secret is configured (e.g. local dev), the gate is bypassed.

def _password_gate() -> bool:
    try:
        expected = st.secrets["PASSWORD"]
    except (KeyError, FileNotFoundError):
        return True  # no secret configured → open access (local dev)

    if st.session_state.get("_authed"):
        return True

    # Minimal login screen
    st.markdown("### 🎯 keyhole-sizer")
    st.markdown(
        "This sandbox is shared-password protected. Enter the access password "
        "to continue. If you don't have one, ping Kyle."
    )
    with st.form("_auth", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if pw == expected:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _password_gate():
    st.stop()


# ───────────────────────── Header ─────────────────────────

st.title("🎯 keyhole-sizer")
with st.expander("ℹ About this sizer", expanded=False):
    st.markdown(
        "Interactive sandbox for the Keyhole bake-off findings. "
        "Tune NPU spec, pipeline, concurrency, and LLM load to see live "
        "FPS / tok/s / duty-cycle projections. All numbers trace back to "
        "measured bake-offs (`github.com/kylefoxaustin/keyhole`, see "
        "`REPRODUCE.md`).\n\n"
        "⚠️ Assumes vision/LLM time-slice on the NPU — concurrent BW "
        "contention not modeled."
    )


# ───────────────────────── Helpers: chart theme + pipeline strip ─────────────────────────

# Sizer's dark-theme palette for Plotly figures. Applied consistently so
# axis ticks, titles, and gridlines render as readable near-white-on-navy
# instead of Plotly's default gray-on-dark (which has low contrast).
_CHART_FG = "#EAEDF4"
_CHART_BG = "#0F192E"
_CHART_GRID = "#334155"


def _apply_chart_theme(fig):
    """Apply the sizer's dark-theme axis + legend colors to a Plotly figure.

    Called AFTER the figure's primary update_layout() so per-chart titles,
    heights, margins, and layout tweaks are preserved. Uses update_xaxes /
    update_yaxes / update_layout which MERGE rather than replace, so any
    chart that already sets e.g. tickfont=dict(size=13) keeps its size.

    Fixes: axis ticks / axis titles / gridlines inheriting Plotly's default
    gray-on-dark palette instead of the figure-level font color.
    """
    fig.update_xaxes(
        color=_CHART_FG,
        gridcolor=_CHART_GRID,
        title_font=dict(color=_CHART_FG),
    )
    fig.update_yaxes(
        color=_CHART_FG,
        gridcolor=_CHART_GRID,
        title_font=dict(color=_CHART_FG),
    )
    fig.update_layout(
        legend=dict(font=dict(color=_CHART_FG)),
    )


def _render_pipeline_strip(stages: list[tuple[str, bool]]):
    """Horizontal pipeline flow mirroring the Keyhole deck's exec summary.
    Highlighted boxes are indigo; dim boxes are neutral. Arrows between."""
    box_highlight = (
        "background:#6366F1; color:#FFFFFF; border:1.5px solid #6366F1; "
        "font-weight:600;"
    )
    # "Fixed infrastructure" boxes — always-on pipeline stages that don't vary with
    # the user's pipeline selection. Bright text on a mid-slate so they read as
    # "present, supporting" not "disabled/placeholder."
    box_dim = (
        "background:#334155; color:#EAEDF4; border:1.5px solid #475569;"
    )
    arrow = (
        '<div style="display:flex; align-items:center; color:#6366F1; '
        'font-size:24px; padding:0 4px;">→</div>'
    )
    parts = []
    for i, (label, hl) in enumerate(stages):
        style = box_highlight if hl else box_dim
        parts.append(
            f'<div style="{style} border-radius:10px; padding:10px 14px; '
            f'min-width:115px; flex:1; text-align:center; font-size:12.5px; '
            f'line-height:1.35; white-space:pre-line;">{label}</div>'
        )
        if i < len(stages) - 1:
            parts.append(arrow)
    html = (
        '<div style="display:flex; flex-wrap:nowrap; align-items:stretch; '
        f'gap:2px; margin:12px 0 8px;">{"".join(parts)}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _stages_for_pipeline(pipeline_key: str, llm_enabled: bool,
                         llm_workload: str, quant: str) -> list[tuple[str, bool]]:
    """Build the 5-stage flow based on the user's current selections.
    Highlight = 'this box is driven by the user's pipeline / LLM choice'."""
    mapping = {
        "sam3_bf16":             ("YOLO 11x", False,       "SAM 3 BF16", True),
        "essmall_fp8":           ("YOLO 11x", False,       "EfficientSAM-Small FP8", True),
        "efficientsam3_es_ev_s_bf16": ("YOLO 11x", False,  "EfficientSAM3 ES-EV-S\nBF16", True),
        "efficientsam3p1_es_ev_s_bf16": ("(text-prompt)", False, "EfficientSAM3.1 ES-EV-S\nBF16 (n=1 concept)", True),
        "yoloe26_s_pf_fp16":          ("YOLOE-26S-PF FP16\n(one model)", True, "(open-vocab built-in)", False),
        "yoloe26_s_pf_trt_fp8":       ("YOLOE-26S-PF TRT FP8\n(optimized ceiling)", True, "(open-vocab built-in)", False),
        "hybrid_v2_bf16":        ("YOLO-seg BF16", True,   "CLIP BF16", True),
        "hybrid_v2_torchao_fp8": ("YOLO-seg BF16", True,   "CLIP FP8 (torchao)", True),
        "trt_fp8_every_frame":   ("YOLO-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\nevery frame", True),
        "trt_fp8_1hz_clip":      ("YOLO-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\n@ 1 Hz", True),
        "yolo_only_fp8":         ("YOLO-seg FP8 (TRT)", True, "(no CLIP)", False),
        "yolov8n_trt_fp8_every_frame":  ("yolov8n-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\nevery frame", True),
        "yolov8n_trt_fp8_1hz_clip":     ("yolov8n-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\n@ 1 Hz", True),
        "yolov8n_only_fp8":             ("yolov8n-seg FP8 (TRT)", True, "(no CLIP)", False),
        "yolo11s_trt_int8":              ("yolo11s-seg INT8 (TRT)\n20-frame PTQ", True, "(no CLIP)", False),
        "yolov8n_trt_int8_coco128":      ("yolov8n-seg INT8 (TRT)\ncoco128-seg PTQ", True, "(no CLIP)", False),
        "yolov8n_trt_int8_20frame":      ("yolov8n-seg INT8 (TRT)\n20-frame PTQ ⚠", True, "(no CLIP)", False),
    }
    det_label, det_hl, enr_label, enr_hl = mapping.get(
        pipeline_key, ("?", False, "?", False)
    )
    stages: list[tuple[str, bool]] = [
        ("FFmpeg\ningest", False),
        (det_label, det_hl),
        (enr_label, enr_hl),
        ("SQLite\n+ FTS5", False),
    ]
    if llm_enabled:
        wl_label = WORKLOAD_CATEGORIES[llm_workload]["label"]
        stages.append((f"Qwen3-30B-A3B {quant}\n{wl_label}", True))
    else:
        stages.append(("NLQ / LLM\n(off)", False))
    return stages

# ───────────────────────── Sidebar: hardware + workload ─────────────────────────

with st.sidebar:
    st.header("Edge NPU")
    tier = st.selectbox(
        "Tier preset",
        options=("NPU Low-LP5-32bit", "NPU Low-LP5-64bit", "NPU Low-LP5X",
                 "NPU Mid", "NPU High", "Custom"),
        index=3,  # lands on 'NPU Mid'
        help="Low-LP5-32bit = 32-bit LPDDR5 @ 6.4 GT/s (25.6 GB/s theoretical), dense INT8-only silicon. "
             "Low-LP5-64bit = 64-bit LPDDR5 @ 6.4 GT/s (51.2 GB/s, 2× the 32-bit variant). "
             "Low-LP5X = same 64-bit bus on LPDDR5X @ 8.4 GT/s (67.2 GB/s, 1.3× Low-LP5-64bit). "
             "Mid = 128-bit LPDDR5X @ 8.4 GT/s (Keyhole shipping target, BF16/FP8-capable). "
             "High = 128-bit LPDDR5X @ 11.2 GT/s (vendor high-bin). "
             "Custom = roll your own. All presets assume 70% bandwidth efficiency.",
        key="tier",
    )

    if tier == "Custom":
        st.markdown("---")
        st.caption("Custom NPU")
        bus_width = st.select_slider(
            "Memory bus width (bits)",
            options=[64, 96, 128, 192, 256, 384, 512],
            value=128,
            key="custom_bus_width",
        )
        mem_type = st.selectbox("Memory type", options=MEMORY_TYPES, index=2,
                                 key="custom_mem_type")
        data_rate = st.slider("Data rate (GT/s)", min_value=2.0, max_value=32.0,
                               value=8.4, step=0.1,
                               key="custom_data_rate")
        theoretical_bw = theoretical_bandwidth(bus_width, data_rate)
        st.caption(f"→ theoretical BW: **{theoretical_bw:.1f} GB/s**")
        bw_eff = st.slider("Bandwidth efficiency", 0.50, 0.95, 0.70, 0.01,
                            help="Fraction of theoretical BW realized on real workloads. "
                                 "Presets use 0.70 (matches the tier cards above).",
                            key="custom_bw_eff")
        # Both sliders share the same 50-1000 range so equal numeric values
        # sit at equal horizontal positions. FP8 on Blackwell-class silicon
        # is typically 2× BF16, but some NPUs are 1:1 or even missing FP8 —
        # decouple the sliders so the user can model either.
        tops_bf16 = st.slider("Peak BF16 TOPS", 50, 1000, 200, 10,
                               key="custom_tops_bf16")
        tops_fp8 = st.slider("Peak FP8 TOPS", 0, 1000, min(1000, int(tops_bf16 * 2)), 10,
                              help="Set to 0 if silicon doesn't support FP8 natively. "
                                   "Blackwell-class is typically 2× BF16.",
                              key="custom_tops_fp8")
        compute_eff = st.slider("Compute efficiency", 0.40, 0.85, 0.65, 0.01,
                                 key="custom_compute_eff")
        mem_cap = st.slider("DRAM capacity (GB)", 2, 64, 8, 1,
                             key="custom_mem_cap")
        tdp = st.slider("TDP (W)", 2, 150, 25, 1,
                         key="custom_tdp")

        hw = Hardware(
            name="Custom NPU",
            peak_tops_bf16=tops_bf16, peak_tops_int8=tops_fp8, peak_tops_fp8=tops_fp8,
            mem_bandwidth_gbs=theoretical_bw, mem_capacity_gb=mem_cap,
            mem_bus_width_bits=bus_width, mem_type=mem_type, mem_data_rate_gtps=data_rate,
            compute_efficiency=compute_eff, bandwidth_efficiency=bw_eff,
            tdp_watts=tdp,
        )
    else:
        hw = TIERS[tier]

    st.caption(describe_hw(hw))

    st.markdown("---")
    st.header("Vision workload")

    # Pipeline is picked in two steps: a radio for the narrative track,
    # then a selectbox scoped to that track. Each track has its own
    # selectbox state (via key=f"pipeline__{track_label}") so switching
    # tracks and coming back remembers the last pick for that track;
    # first visit to a track shows that track's canonical default.
    PIPELINE_TRACKS = [
        ("SAM 3 lineage",
         ["sam3_bf16", "essmall_fp8",
          "efficientsam3_es_ev_s_bf16", "efficientsam3p1_es_ev_s_bf16"],
         "sam3_bf16"),
        ("One-model open-vocab",
         ["yoloe26_s_pf_fp16", "yoloe26_s_pf_trt_fp8"],
         "yoloe26_s_pf_trt_fp8"),
        ("Default (Hybrid V2 → TRT)",
         ["hybrid_v2_bf16", "hybrid_v2_torchao_fp8",
          "trt_fp8_every_frame", "trt_fp8_1hz_clip", "yolo_only_fp8"],
         "trt_fp8_1hz_clip"),
        ("YOLOv8n nano",
         ["yolov8n_trt_fp8_every_frame", "yolov8n_trt_fp8_1hz_clip",
          "yolov8n_only_fp8"],
         "yolov8n_trt_fp8_1hz_clip"),
        ("INT8 vendor-comparison",
         ["yolo11s_trt_int8", "yolov8n_trt_int8_coco128",
          "yolov8n_trt_int8_20frame"],
         "yolov8n_trt_int8_coco128"),
    ]
    _TRACK_LABELS = [t[0] for t in PIPELINE_TRACKS]
    _DEFAULT_TRACK_INDEX = 2  # "Default (Hybrid V2 → TRT)"

    # Invariant: PIPELINES ⊆ ⋃(tracks) AND ⋃(tracks) ⊆ PIPELINES.
    # Fails loud on startup/rerun rather than silently dropping a new
    # pipeline from the dropdown or silently keeping a stale key.
    _track_pipelines = {k for _, keys, _ in PIPELINE_TRACKS for k in keys}
    _orphaned = set(PIPELINES.keys()) - _track_pipelines
    _unknown  = _track_pipelines - set(PIPELINES.keys())
    if _orphaned:
        raise RuntimeError(
            f"PIPELINES keys orphaned from PIPELINE_TRACKS (won't appear in "
            f"dropdown): {sorted(_orphaned)}. Add them to a track in "
            f"app.py::PIPELINE_TRACKS."
        )
    if _unknown:
        raise RuntimeError(
            f"PIPELINE_TRACKS references unknown pipeline keys: "
            f"{sorted(_unknown)}. Typo, or removed from PIPELINES?"
        )

    track_label = st.radio(
        "Pipeline track",
        options=_TRACK_LABELS,
        index=_DEFAULT_TRACK_INDEX,
        help="Pick a narrative track, then choose a specific pipeline within it. "
             "Tracks match the deck's optimization journey: where we started (SAM 3), "
             "one-model open-vocab alternatives, the Hybrid V2 → TRT default path, "
             "the yolov8n nano cross-variant, and INT8 vendor-comparison points.",
        key="pipeline_track",
    )
    _, _track_keys, _canonical = next(
        t for t in PIPELINE_TRACKS if t[0] == track_label
    )

    pipeline_key = st.selectbox(
        "Pipeline",
        options=_track_keys,
        format_func=lambda k: PIPELINES[k].label,
        index=_track_keys.index(_canonical),
        key=f"pipeline__{track_label}",
    )
    pipeline = PIPELINES[pipeline_key]
    st.caption(pipeline.description)

    resolution = st.selectbox("Per-stream resolution", ("720p", "1080p", "4K"), index=0,
                               key="resolution")
    n_streams = st.slider("Concurrent streams", 1, 16, 1, 1,
                           help="Each stream processes its own video source. YOLO batching "
                                "kicks in automatically (batch = N_streams).",
                           key="n_streams")

    compiler_quality = st.slider(
        "Edge compiler quality vs TensorRT", 0.50, 1.00, 1.00, 0.05,
        help="5090 measurements came out of NVIDIA TensorRT — a best-in-class compiler. "
             "Vendor edge-NPU compilers (SNPE, NeuroPilot, OpenVINO-NPU, etc.) typically "
             "extract a fraction of the same theoretical peak. **1.00 = parity** (projections "
             "unchanged, optimistic). **0.75 = realistic** (edge compiler 25% slower per kernel). "
             "**0.50 = pessimistic** (half as good — first-gen NPU SDK). Applied as a post-multiplier "
             "on every projected vision latency path.",
        key="compiler_quality",
    )
    if compiler_quality < 1.00:
        st.caption(f"⚠️ Applying {(1 - compiler_quality) * 100:.0f}% compiler-quality haircut "
                    f"to projected vision FPS (LLM tok/s unaffected — those are vendor-measured).")

    with st.expander("ℹ️ CPU preprocessing cost (not in these FPS numbers)"):
        st.markdown(
            "Every YOLO frame needs a **640×640 letterbox resize** before it hits "
            "the TRT engine. That resize runs on the **host CPU** (OpenCV "
            "`cv2.resize` bilinear), not on the GPU/NPU — and it's excluded from "
            "the engine ms/frame numbers here, the same way it's excluded from the "
            "5090 bake-off timings.\n\n"
            "**Measured** (5090 host, i9-14900KF, single-thread, N=500):\n"
            "- 720p → 640×640: **0.17 ms/frame**\n"
            "- 1080p → 640×640: **0.32 ms/frame**\n"
            "- 4K → 640×640: **0.33 ms/frame**\n\n"
            "That's **~0.5–1% of one CPU core at 30 fps**. Flat across source "
            "resolutions because the 640×640 output dominates cost.\n\n"
            "**Edge ARM extrapolation** (Cortex-A55 ≈ 10× slower single-thread): "
            "~**2–3 ms/frame**, ~**6–10% of one edge core at 30 fps**.\n\n"
            "**The caveat that matters:** most edge SoCs move this off-CPU via a "
            "fixed-function ISP, 2D GPU, or video-decoder output scaler "
            "(Qualcomm, MediaTek, NXP i.MX 95, Ambarella, Hailo all ship one). "
            "Pure-NPU boards without such a block (e.g., Google Coral) pay the "
            "full CPU cost."
        )

    st.markdown("---")
    st.header("LLM co-exist")
    llm_enabled = st.toggle("Share the NPU with a generative LLM",
                             value=False,
                             help="Qwen3-30B-A3B MoE (3B active / 30B total).",
                             key="llm_enabled")
    if llm_enabled:
        quant = st.selectbox("Qwen3 quantization",
                              ("Q4_K_M", "Q5_K_M", "Q8_0"), index=0,
                              key="llm_quant")
        llm_workload = st.selectbox(
            "LLM workload pattern",
            options=list(WORKLOAD_CATEGORIES.keys()),
            format_func=lambda k: WORKLOAD_CATEGORIES[k]["label"],
            index=0,
            help="Real-world workload categories measured on Skippy production "
                 "(n=1-5 per category). Decode tok/s spans 3.6-222 across "
                 "categories — pick the one your deployment will actually see.",
            key="llm_workload",
        )
        st.caption(WORKLOAD_CATEGORIES[llm_workload]["description"])

        with st.expander("ℹ️ About these workload patterns"):
            st.markdown(
                "All five patterns were measured in production against "
                "**Qwen3-30B-A3B-Instruct-2507** (Q4_K_M GGUF, llama.cpp) on "
                "an **RTX 5090**. Decode tok/s spans **3.6 → 222 across real "
                "traffic** — a ~60× range that single-number vendor benchmarks "
                "don't capture."
            )
            for key, wc in WORKLOAD_CATEGORIES.items():
                st.markdown(
                    f"**{wc['label']}** &nbsp;·&nbsp; *n={wc['n']}*  \n"
                    f"{wc['description']}  \n"
                    f"5090 reference: **{wc['decode_5090_tok_s_p50']:.1f} tok/s decode (p50)**, "
                    f"TTFT **{wc['ttft_5090_sec_p50']*1000:.0f} ms** (p50)  \n"
                    f"🔸 *{wc['note']}*"
                )
            st.markdown(
                "---\n"
                "**How the sizer scales these:** *plain chat* is the reference (≈ the 1K-prompt "
                "condition under which vendor NPU Q4 benchmarks are published). Each category's "
                "multiplier (measured on 5090) is applied to the target NPU's plain-chat "
                "projection. Both decode tok/s and TTFT are scaled."
            )
        queries_per_min = st.slider("LLM queries per minute", 0.0, 60.0, 2.0, 0.1,
                                     key="llm_queries_per_min")
        answer_kind = st.radio("Typical answer length",
                                ("short", "rag"),
                                index=0,
                                format_func=lambda k: {
                                    "short": "Short (~200 tokens)",
                                    "rag":   "RAG (8K prompt + 2K response)"}[k],
                                horizontal=True,
                                key="llm_answer_kind")
    else:
        quant = "Q4_K_M"
        llm_workload = "plain_chat"
        queries_per_min = 0.0
        answer_kind = "short"

# ───────────────────────── Main area ─────────────────────────

# Compute projections
vision = project_vision(pipeline, hw, resolution, n_streams=n_streams,
                         compiler_quality_vs_trt=compiler_quality)
llm = project_llm(hw, quant, workload=llm_workload) if llm_enabled else None

# ───────────────────────── Front-page summary + pipeline strip ─────────────────────────
# Dynamic "Simulating" line reflecting the current selection.
# HTML form (uses <b> so it renders bold inside an HTML-styled div below).
if llm_enabled:
    wl_label = WORKLOAD_CATEGORIES[llm_workload]["label"]
    llm_summary = (
        f" &middot; LLM <b>on</b> &mdash; Qwen3-30B-A3B <b>{quant}</b>, "
        f"<b>{wl_label}</b> @ <b>{queries_per_min:.1f} q/min</b> "
        f"(<b>{answer_kind}</b> answers)"
    )
else:
    llm_summary = " &middot; LLM <b>off</b>"

# Build the current-config rows (needed by the 'This config' download)
_cur_rows: list[dict] = [
    vision_workload_row(pipeline, hw, resolution, n_streams=n_streams)
]
if llm_enabled:
    _cur_rows.append(llm_workload_row(
        hw, quant, workload=llm_workload,
        queries_per_minute=queries_per_min,
        answer_kind=answer_kind,
    ))
_cur_csv = rows_to_csv_str(_cur_rows)
_hw_slug = hw.name.lower().replace(" ", "_")

# ── Simulating line — full-width summary of the current selection ──
st.markdown(
    "<div style='font-size:17px; line-height:1.55; margin:4px 0 10px 0;'>"
    f"<b>Simulating:</b> {pipeline.label} &nbsp;&middot;&nbsp; "
    f"<b>{n_streams}</b> stream{'s' if n_streams != 1 else ''} "
    f"@ <b>{resolution}</b> &nbsp;&middot;&nbsp; "
    f"<b>{hw.name}</b>{llm_summary}"
    "</div>",
    unsafe_allow_html=True,
)

# ── Projected results: effective-under-LLM math + saturation banner ──
vision_fps_effective = vision["fps_per_stream"]
duty_cycle = 0.0
llm_saturated = False
if llm_enabled:
    qps = queries_per_min / 60
    answer_sec = llm["short_answer_sec"] if answer_kind == "short" else llm["rag_total_sec"]
    duty_cycle = qps * answer_sec
    llm_saturated = duty_cycle >= 1.0
    vision_fps_effective = vision_fps_under_llm_load(
        vision["fps_per_stream"], llm, queries_per_min, answer_kind
    )

if llm_saturated:
    max_qpm = (60 / answer_sec) if answer_sec > 0 else 0
    st.error(
        f"⚠ **NPU oversubscribed by the LLM.** {queries_per_min:.1f} {answer_kind} "
        f"queries/min × {answer_sec:.1f} s/answer = {duty_cycle*100:.0f}% duty cycle. "
        f"Vision is starved to 0 FPS and the LLM queue backs up. "
        f"On {hw.name} at {quant}, the maximum sustainable "
        f"{'short-answer' if answer_kind == 'short' else 'RAG'} rate is ~**{max_qpm:.1f} "
        f"queries/min** (100% NPU duty). Reduce query rate, use a lighter "
        f"answer mode (short vs RAG), upgrade NPU tier, or dedicate a second NPU to the LLM."
    )

# ── Top metric row — the headline numbers, sitting above the fold ──
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    label="Per-camera FPS",
    value=f"{vision_fps_effective:.1f}",
    delta=(f"{vision_fps_effective - vision['fps_per_stream']:+.1f}  under LLM"
            if llm_enabled else f"{n_streams} streams @ {resolution}"),
    delta_color="inverse" if llm_enabled else "off",
    help=(
        "Frame rate **each individual camera stream delivers** — what the "
        "end user sees. Targets: **30 FPS** real-time, **15 FPS** "
        "surveillance-grade."
    ),
)
c2.metric(
    label="Aggregate FPS",
    value=f"{vision_fps_effective * n_streams:.0f}",
    delta=f"{n_streams} streams total",
    delta_color="off",
    help=(
        "Sum of frames/sec **across all cameras** — the NPU's total "
        "throughput capacity. `per-camera FPS × n_streams`."
    ),
)
c3.metric(
    label="Memory fit",
    value="✓ fits" if vision["fits_in_memory"] else "✗ spills",
    delta=f"{vision['vram_mb']:.0f} MB / {hw.mem_capacity_gb*1024:.0f} MB",
    delta_color="off",
)
if llm_enabled:
    c4.metric(
        label=f"LLM decode ({quant})",
        value=f"{llm['decode_tok_s']:.1f} tok/s",
        delta=f"TTFT 1K = {llm['ttft_1k_sec']*1000:.0f} ms",
    )
else:
    c4.metric(
        label="DDR bandwidth ratio vs NPU Mid",
        value=f"{vision['bandwidth_ratio_vs_ref']:.2f}×",
        delta="reference = NPU Mid",
        delta_color="off",
        help=(
            "Ratio of the current hardware's **effective DRAM (LPDDR/GDDR) "
            "bandwidth** to NPU Mid's. Vision pipelines at these model sizes "
            "are bandwidth-bound, so edge FPS scales roughly linearly with "
            "this ratio. Compute is NOT compared here — this is purely the "
            "off-chip memory-bus ratio (bus width × data rate × efficiency)."
        ),
    )

# ── Pipeline flow (collapsible, expanded by default) ──
with st.expander("🔀 Pipeline flow", expanded=True):
    _render_pipeline_strip(
        _stages_for_pipeline(pipeline_key, llm_enabled, llm_workload, quant)
    )
    _legend_html = (
        '<div style="display:flex; flex-wrap:wrap; align-items:center; '
        'gap:20px; margin:4px 0 2px;">'
        '<div style="display:flex; align-items:center; gap:7px;">'
        '<span style="display:inline-block; width:16px; height:16px; '
        'background:#334155; border:1.5px solid #475569; border-radius:3px;"></span>'
        '<span style="font-size:13px;">'
        '<b>Always on</b> &nbsp;— ingest, storage</span></div>'
        '<div style="display:flex; align-items:center; gap:7px;">'
        '<span style="display:inline-block; width:16px; height:16px; '
        'background:#6366F1; border:1.5px solid #6366F1; border-radius:3px;"></span>'
        '<span style="font-size:13px;">'
        '<b>Pipeline stage changes</b> &nbsp;— varies with your choice</span></div>'
        '</div>'
        '<div style="font-size:12px; opacity:0.85; margin-top:4px;">'
        'Every stage is running — the colors just flag where your controls take effect.'
        '</div>'
    )
    st.markdown(_legend_html, unsafe_allow_html=True)

# ── Export data (collapsible, collapsed by default) ──
# Two rows: platform-budget CSVs on top, KPI-spreadsheet preview triggers
# below. KPI preview (if active) renders just below the expander so it's
# full-width and readable.
with st.expander("📥 Export data", expanded=False):
    _btn_cur_col, _btn_mat_col = st.columns(2)
    with _btn_cur_col:
        st.download_button(
            label="💾 This config",
            data=_cur_csv,
            file_name=f"keyhole_sizer_budget_{_hw_slug}_{resolution}_n{n_streams}.csv",
            mime="text/csv",
            help=(
                "Platform-budget CSV row for the *current* config (vision + LLM if "
                "enabled). ss_* columns are additive at the platform level; peak_* "
                "are per-workload ceilings. Read the header `#` comments for caveats."
            ),
            use_container_width=True,
        )
    with _btn_mat_col:
        st.download_button(
            label="📦 All configs",
            data=_full_matrix_csv(),
            file_name="keyhole_sizer_platform_budget_matrix.csv",
            mime="text/csv",
            help=(
                "Every preset HW tier × pipeline × resolution × stream count + every "
                "LLM (quant × workload × answer_kind) combination (~585 rows). "
                "Custom HW is skipped — use 'This config' for custom. Cached hourly."
            ),
            use_container_width=True,
        )
    _kpi_btn_all_col, _kpi_btn_one_col = st.columns(2)
    with _kpi_btn_all_col:
        if st.button(
            "📊 KPI spreadsheet (all models)",
            use_container_width=True,
            key="kpi_btn_all",
            help="Reveal a table of per-pipeline KPIs across all 17 pipelines at "
                 "the current HW / resolution / LLM state, plus a button to "
                 "download the formatted XLSX.",
        ):
            st.session_state.kpi_preview_mode = "all"
    with _kpi_btn_one_col:
        if st.button(
            "📊 KPI spreadsheet (this model)",
            use_container_width=True,
            key="kpi_btn_this",
            help="Reveal the KPI row for just the currently-selected pipeline "
                 "(the one from the sidebar), plus a button to download the XLSX.",
        ):
            st.session_state.kpi_preview_mode = "this"

# ── KPI preview (renders below the expander when a button above is active) ──
_kpi_mode = st.session_state.get("kpi_preview_mode")
if _kpi_mode in ("all", "this"):
    if _kpi_mode == "all":
        _kpi_rows = all_pipeline_kpi_rows(
            hw, resolution=resolution,
            llm_enabled=llm_enabled, llm_quant=quant,
            llm_workload=llm_workload,
            compiler_quality_vs_trt=compiler_quality,
        )
        _kpi_file_slug = "all"
    else:
        _kpi_rows = [pipeline_kpi_row(
            pipeline_key, hw, resolution=resolution,
            llm_enabled=llm_enabled, llm_quant=quant,
            llm_workload=llm_workload,
            compiler_quality_vs_trt=compiler_quality,
        )]
        _kpi_file_slug = pipeline_key

    # Override total_fps so it matches the metric card's Per-camera FPS
    # exactly (same math as the website: project_vision at current hw /
    # resolution / n_streams / compiler_quality, plus vision_fps_under_llm_load
    # when LLM is on). The kpi_breakdown module's own total_fps is vision-only
    # including ingest — honest, but different from what users read in the
    # main metric row, which caused 'wait, why don't these match?' confusion.
    for _row in _kpi_rows:
        _pipe_for_row = PIPELINES[_row["pipeline_key"]]
        _v_for_row = project_vision(
            _pipe_for_row, hw, resolution, n_streams=n_streams,
            compiler_quality_vs_trt=compiler_quality,
        )
        _base_fps = _v_for_row["fps_per_stream"]
        _row["total_fps"] = round(
            vision_fps_under_llm_load(
                _base_fps, llm, queries_per_min, answer_kind
            ) if llm_enabled else _base_fps,
            2,
        )

    _kpi_xlsx_bytes = kpi_rows_to_xlsx(_kpi_rows)
    _kpi_data_col, _kpi_dl_col = st.columns([5, 1])
    with _kpi_dl_col:
        st.download_button(
            "⬇ Download XLSX",
            data=_kpi_xlsx_bytes,
            file_name=f"keyhole_sizer_kpi_{_hw_slug}_{resolution}_{_kpi_file_slug}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"kpi_dl_{_kpi_mode}",
        )
    with _kpi_data_col:
        st.dataframe(
            pd.DataFrame(_kpi_rows),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "`total_fps` matches the **Per-camera FPS** metric card above — "
        "same math (engine ms only, with LLM duty-cycle reduction applied "
        "when LLM is on). The `ingest_ms` column is shown for reference "
        "but not in `total_fps`."
    )

st.markdown("---")

# ───────── Tabs: charts + detail tables ─────────
tab_overview, tab_streams, tab_duty, tab_detail = st.tabs(
    ["📊 Overview", "🎥 Stream scaling", "⚖ Duty-cycle", "🔎 Detail"]
)

with tab_overview:
    st.markdown("### Vision")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Pipeline timing (current config)")
        fig = go.Figure()
        if "yolo_ms" in vision and "clip_ms" in vision:
            fig.add_trace(go.Bar(
                x=["YOLO-seg (batched)", "CLIP component"],
                y=[vision["yolo_ms"], vision["clip_ms"]],
                marker=dict(color=["#6366F1", "#22C55E"]),
                text=[f"{vision['yolo_ms']:.1f} ms", f"{vision['clip_ms']:.1f} ms"],
                textposition="auto",
            ))
        else:
            # Fallback — single-unit pipelines (SAM 3, ES-Small alone) still get a bar
            fig.add_trace(go.Bar(
                x=[f"{pipeline.label} (total)"],
                y=[vision["per_stream_ms"]],
                marker=dict(color=["#F59E0B"]),
                text=[f"{vision['per_stream_ms']:.1f} ms"],
                textposition="auto",
            ))
        fig.update_layout(
            yaxis_title="Edge ms per batch cycle",
            plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
            font=dict(color="#EAEDF4"),
            height=300, margin=dict(l=40, r=20, t=20, b=40),
        )
        _apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(pipeline.note)
        if pipeline.key in {"trt_fp8_1hz_clip", "trt_fp8_every_frame",
                             "hybrid_v2_bf16", "hybrid_v2_torchao_fp8",
                             "yolo_only_fp8",
                             "yolov8n_trt_fp8_1hz_clip", "yolov8n_trt_fp8_every_frame",
                             "yolov8n_only_fp8",
                             "yolo11s_trt_int8",
                             "yolov8n_trt_int8_coco128", "yolov8n_trt_int8_20frame"}:
            st.caption(
                "ℹ️ **Why resolution barely moves the needle:** YOLO runs at a "
                "fixed **640²** input and CLIP at **224²**. Source resolution "
                "only affects FFmpeg decode + resize — a small fraction of the "
                "inference budget. Measured in the Keyhole bake-off: 4K is "
                "only **~21% slower** than 720p, not 9× slower."
            )

    with right:
        st.subheader("Per-stream FPS vs NPU tier")
        tier_rows = []
        for name, t_hw in TIERS.items():
            v = project_vision(pipeline, t_hw, resolution, n_streams=n_streams,
                               compiler_quality_vs_trt=compiler_quality)
            tier_rows.append(dict(tier=name, fps=v["fps_per_stream"]))
        # Add current if custom
        if hw.name not in TIERS:
            tier_rows.append(dict(tier=hw.name, fps=vision["fps_per_stream"]))
        df_tier = pd.DataFrame(tier_rows)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_tier["tier"], y=df_tier["fps"],
            marker=dict(color=["#EF4444", "#22C55E", "#6366F1", "#F59E0B"][:len(df_tier)]),
            text=[f"{f:.1f}" for f in df_tier["fps"]],
            textposition="auto",
        ))
        fig2.add_hline(y=30, line_dash="dot", line_color="#93A1B5",
                        annotation_text="30 FPS real-time", annotation_position="right")
        fig2.add_hline(y=15, line_dash="dot", line_color="#93A1B5",
                        annotation_text="15 FPS surveillance", annotation_position="right")
        fig2.update_layout(
            yaxis_title="FPS per stream", plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
            font=dict(color="#EAEDF4"),
            height=300, margin=dict(l=40, r=60, t=20, b=40),
        )
        _apply_chart_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ───── DRAM bandwidth: saturation approximation vs ncu measurement ─────
    st.markdown("---")
    st.markdown("#### DRAM bandwidth — saturation model vs ncu measurement")

    measured_bytes_per_frame = measured_dram_per_frame(pipeline.key)
    effective_total_fps = vision["fps_per_stream"] * n_streams
    approx_gbs = hw.effective_bandwidth_gbs   # CSV saturation approx at duty=1

    bw_chart_col, bw_text_col = st.columns([1.3, 1])
    with bw_chart_col:
        if measured_bytes_per_frame is None:
            st.info(
                f"No ncu measurement mapped for **{pipeline.label}** yet. "
                "The platform-budget CSV's `ss_ddr_gbs_avg_measured` column "
                "is blank for this pipeline; only the saturation "
                "approximation is available."
            )
        else:
            measured_gbs = measured_bytes_per_frame * effective_total_fps / 1e9
            fig_bw = go.Figure()
            fig_bw.add_trace(go.Bar(
                x=["Saturation model<br>(CSV ss_ddr_gbs_avg)",
                   "Measured (ncu)<br>ss_ddr_gbs_avg_measured"],
                y=[approx_gbs, measured_gbs],
                marker=dict(color=["#EF4444", "#22C55E"]),
                text=[f"{approx_gbs:.1f} GB/s", f"{measured_gbs:.2f} GB/s"],
                textposition="outside",
                textfont=dict(size=14, color="#EAEDF4"),
                cliponaxis=False,
            ))
            fig_bw.add_hline(
                y=hw.effective_bandwidth_gbs,
                line_dash="dot", line_color="#93A1B5",
                annotation_text=f"{hw.name} ceiling ({hw.effective_bandwidth_gbs:.1f} GB/s)",
                annotation_position="top right",
            )
            fig_bw.update_layout(
                yaxis_title="DRAM GB/s consumed by vision pipeline",
                plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
                font=dict(color="#EAEDF4", size=13),
                height=340, margin=dict(l=50, r=40, t=30, b=60),
                showlegend=False,
            )
            _apply_chart_theme(fig_bw)
            st.plotly_chart(fig_bw, use_container_width=True)

    with bw_text_col:
        if measured_bytes_per_frame is None:
            st.caption(
                "**Currently mapped pipelines:** trt_fp8_1hz_clip, "
                "trt_fp8_every_frame, yolo_only_fp8, hybrid_v2_*, "
                "yoloe26_*, efficientsam3_es_ev_s_bf16, essmall_fp8. "
                "**Pending:** sam3_bf16, efficientsam3p1_es_ev_s_bf16 "
                "(kernel-replay ncu sweeps queued)."
            )
        else:
            measured_gbs = measured_bytes_per_frame * effective_total_fps / 1e9
            util_pct = (measured_gbs / approx_gbs * 100) if approx_gbs > 0 else 0
            headroom_gbs = max(0.0, approx_gbs - measured_gbs)
            st.markdown(
                f"**Per-frame DRAM:** {measured_bytes_per_frame / 1e6:.1f} MB  \n"
                f"**Pipeline FPS (all streams):** {effective_total_fps:.1f}  \n"
                f"**Measured usage:** {measured_gbs:.2f} GB/s "
                f"({util_pct:.1f}% of {hw.name} ceiling)  \n"
                f"**Spare bandwidth:** {headroom_gbs:.1f} GB/s"
            )
            comps = measured_components(pipeline.key) or []
            if len(comps) > 1:
                parts = " + ".join(
                    f"`{c['ncu_workload_id']}` × {c['fires_per_frame']:.3g} "
                    f"({c['dram_bytes_per_fire']/1e6:.1f} MB/fire)"
                    for c in comps
                )
                st.caption(f"Composition: {parts}")
            st.caption(
                "The **saturation model** pessimistically assumes the workload "
                "pins the bus at the NPU's effective bandwidth. The **measured** "
                "value comes from ncu-counted DRAM bytes per forward × pipeline "
                "FPS. The gap is real headroom for concurrent work — parallel "
                "LLM, extra streams, other workloads."
            )
            meta = bundle_metadata()
            st.caption(
                f"ncu bundle: `{meta['ncu_bundle_timestamp']}` · "
                f"{meta['ncu_n_workloads']} workloads · host "
                f"*{meta['ncu_measurement_host']}*"
            )

    # ───── LLM timing row (only when LLM is enabled) ─────
    if llm_enabled:
        st.markdown("---")
        st.markdown(f"### LLM — Qwen3-30B-A3B MoE @ {quant}")
        llm_left, llm_right = st.columns([1, 1])

        with llm_left:
            st.subheader("LLM timing (ms) — prefill + decode stacked")
            # Compute per-answer-mode prefill + decode for the current HW+quant
            short_prefill_ms = llm["ttft_1k_sec"] * 1000 * 0.2   # 200 token prompt approx
            # Actually we should use a consistent prefill model. Kyle's LLM bake-off
            # measured ttft_1k_sec explicitly for 1K prompts. For short-answer we
            # assume a ~1K prompt (typical chat), for RAG we use 8K.
            short_prefill_ms = llm["ttft_1k_sec"] * 1000           # 1K prompt
            short_decode_ms = (200 / llm["decode_tok_s"]) * 1000 if llm["decode_tok_s"] > 0 else 0
            rag_prefill_ms = llm["rag_prefill_sec"] * 1000
            rag_decode_ms = llm["rag_decode_sec"] * 1000

            # Format ms values with k/s units so tiny bars' labels stay readable
            def _fmt_ms(ms: float) -> str:
                if ms >= 10_000:
                    return f"{ms/1000:.1f} s"
                if ms >= 1000:
                    return f"{ms/1000:.2f} s"
                return f"{ms:.0f} ms"

            fig_llm = go.Figure()
            fig_llm.add_trace(go.Bar(
                name="Prefill", x=["Short (1K prompt, 200 tok)", "RAG (8K+2K)"],
                y=[short_prefill_ms, rag_prefill_ms],
                marker=dict(color="#F59E0B"),
                text=[_fmt_ms(short_prefill_ms), _fmt_ms(rag_prefill_ms)],
                textposition="outside",
                textfont=dict(size=14, color="#EAEDF4"),
                cliponaxis=False,
            ))
            fig_llm.add_trace(go.Bar(
                name="Decode", x=["Short (1K prompt, 200 tok)", "RAG (8K+2K)"],
                y=[short_decode_ms, rag_decode_ms],
                marker=dict(color="#6366F1"),
                text=[_fmt_ms(short_decode_ms), _fmt_ms(rag_decode_ms)],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=16, color="#FFFFFF"),
            ))
            fig_llm.update_layout(
                barmode="stack",
                yaxis_title="Per-answer latency (ms)",
                plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
                font=dict(color="#EAEDF4", size=13),
                legend=dict(orientation="h", y=-0.18, font=dict(size=13)),
                xaxis=dict(tickfont=dict(size=13, color="#EAEDF4")),
                yaxis=dict(tickfont=dict(size=12, color="#EAEDF4"),
                            title_font=dict(size=13, color="#EAEDF4")),
                height=360, margin=dict(l=60, r=30, t=40, b=60),
            )
            _apply_chart_theme(fig_llm)
            st.plotly_chart(fig_llm, use_container_width=True)
            st.caption(
                f"Current answer mode: **{answer_kind}**  •  "
                f"TTFT 1K = **{llm['ttft_1k_sec']*1000:.0f} ms**  •  "
                f"decode = **{llm['decode_tok_s']:.1f} tok/s**"
            )

        with llm_right:
            st.subheader(f"Decode tok/s vs NPU tier — {WORKLOAD_CATEGORIES[llm_workload]['label']}")
            tier_llm = []
            for name, t_hw in TIERS.items():
                l = project_llm(t_hw, quant, workload=llm_workload)
                tier_llm.append(dict(tier=name, tok_s=l["decode_tok_s"]))
            if hw.name not in TIERS:
                tier_llm.append(dict(tier=hw.name, tok_s=llm["decode_tok_s"]))
            df_llm = pd.DataFrame(tier_llm)

            fig_llm_tier = go.Figure()
            fig_llm_tier.add_trace(go.Bar(
                x=df_llm["tier"], y=df_llm["tok_s"],
                marker=dict(color=["#EF4444", "#22C55E", "#6366F1", "#F59E0B"][:len(df_llm)]),
                text=[f"{t:.1f} tok/s" for t in df_llm["tok_s"]],
                textposition="outside",
                textfont=dict(size=14, color="#EAEDF4"),
                cliponaxis=False,
            ))
            fig_llm_tier.update_layout(
                yaxis_title="Decode tok/s",
                plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
                font=dict(color="#EAEDF4", size=13),
                xaxis=dict(tickfont=dict(size=13, color="#EAEDF4")),
                yaxis=dict(tickfont=dict(size=12, color="#EAEDF4"),
                            title_font=dict(size=13, color="#EAEDF4")),
                height=360, margin=dict(l=60, r=30, t=40, b=60),
            )
            _apply_chart_theme(fig_llm_tier)
            st.plotly_chart(fig_llm_tier, use_container_width=True)
            st.caption(
                f"At {quant}, for this workload category. MoE wins on BW: "
                "only 3B of 30B total are loaded per token."
            )

        # ───── Real-workload distribution row ─────
        st.markdown("#### LLM performance across real-workload mixes")
        st.caption("Hover each bar for the category's description, sample size, and measurement caveat.")
        dist = workload_distribution_on_hw(hw, quant)
        labels = [d["label"] + f"  (n={d['n']})" for d in dist]
        values = [d["decode_tok_s"] for d in dist]
        colors = ["#6366F1" if d["key"] == llm_workload else "#374151" for d in dist]

        # Per-bar custom data for the hover popup
        customdata = [[
            d["description"],
            d["note"],
            d["n"],
            d["ttft_sec"] * 1000,
            d["short_answer_sec"],
        ] for d in dist]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            y=labels, x=values,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f} tok/s" for v in values],
            textposition="outside",
            textfont=dict(size=14, color="#EAEDF4"),
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "<br>"
                "Decode: <b>%{x:.2f} tok/s</b><br>"
                "TTFT 1K prompt: %{customdata[3]:.0f} ms<br>"
                "Short 200-tok answer: %{customdata[4]:.1f} s<br>"
                "<br>"
                "<i>%{customdata[0]}</i><br>"
                "Sample size: n = %{customdata[2]}<br>"
                "<br>"
                "⚠ %{customdata[1]}"
                "<extra></extra>"
            ),
        ))
        fig_dist.update_layout(
            xaxis_title=f"Decode tok/s on {hw.name} @ {quant}",
            plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
            font=dict(color="#EAEDF4", size=13),
            xaxis=dict(tickfont=dict(size=12, color="#EAEDF4"),
                        title_font=dict(size=13, color="#EAEDF4")),
            yaxis=dict(tickfont=dict(size=13, color="#EAEDF4"),
                        automargin=True),
            hoverlabel=dict(
                bgcolor="#1A223B",
                bordercolor="#6366F1",
                font=dict(size=13, color="#EAEDF4", family="system-ui"),
                align="left",
            ),
            height=300, margin=dict(l=20, r=40, t=10, b=40),
            showlegend=False,
        )
        _apply_chart_theme(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)

        mx = max(values); mn = min(v for v in values if v > 0)
        st.caption(
            f"Current selection highlighted. Spread: **{mx/mn:.0f}× worst-case** "
            f"between plain-chat peak ({mx:.0f} tok/s) and cold-start tail "
            f"({mn:.1f} tok/s) on this HW + quant. Reference measurements on "
            f"5090 showed ~60× spread (n=1-5 per category). Edge capacity planning "
            f"should budget for the RAG / tool-use tail, not the plain-chat peak."
        )

with tab_streams:
    st.subheader(f"Per-stream FPS vs concurrent stream count — {pipeline.label}")
    rows = []
    for N in [1, 2, 4, 8, 16]:
        v = project_vision(pipeline, hw, resolution, n_streams=N,
                            compiler_quality_vs_trt=compiler_quality)
        rows.append({
            "N streams": N,
            "Per-stream FPS": round(v["fps_per_stream"], 1),
            "Total system FPS": round(v["total_fps"], 1),
            "Batch cycle ms": round(v["per_stream_ms"], 1),
            "VRAM (MB)": round(v["vram_mb"], 0),
            "Fits": "✓" if v["fits_in_memory"] else "✗",
        })
    df_streams = pd.DataFrame(rows)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_streams["N streams"],
            y=df_streams["Per-stream FPS"],
            mode="lines+markers",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=10, color="#6366F1"),
            name="Per-stream FPS",
        ))
        fig3.add_trace(go.Scatter(
            x=df_streams["N streams"],
            y=df_streams["Total system FPS"],
            mode="lines+markers",
            line=dict(color="#22C55E", width=3, dash="dash"),
            marker=dict(size=10, color="#22C55E"),
            name="Total system FPS",
            yaxis="y2",
        ))
        fig3.add_hline(y=30, line_dash="dot", line_color="#93A1B5",
                        annotation_text="30 FPS real-time")
        fig3.update_layout(
            xaxis_title="Concurrent streams",
            yaxis=dict(title="Per-stream FPS"),
            yaxis2=dict(title="Total system FPS", overlaying="y", side="right"),
            plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
            font=dict(color="#EAEDF4"),
            legend=dict(orientation="h", y=-0.2),
            height=420, margin=dict(l=40, r=40, t=20, b=40),
        )
        _apply_chart_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.dataframe(df_streams, use_container_width=True, hide_index=True)
        st.caption("YOLO batching amortizes kernel overhead — 4 streams at batch=4 "
                    "typically get ~70% of the single-stream FPS, not 25%.")

with tab_duty:
    st.subheader("Vision FPS under concurrent LLM load")
    if not llm_enabled:
        st.info("Enable 'Share the NPU with a generative LLM' in the sidebar to see "
                "the duty-cycle trade-off.")
    else:
        qpm = np.linspace(0, 120, 200)
        qps = qpm / 60
        short_ms = llm["short_answer_sec"] * 1000
        rag_ms = llm["rag_total_sec"] * 1000
        duty_short = qps * short_ms / 1000
        duty_rag = qps * rag_ms / 1000
        fps_short = np.clip(vision["fps_per_stream"] * (1 - duty_short), 0,
                             vision["fps_per_stream"])
        fps_rag = np.clip(vision["fps_per_stream"] * (1 - duty_rag), 0,
                           vision["fps_per_stream"])

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=qpm, y=fps_short, mode="lines",
                                   line=dict(color="#22C55E", width=3),
                                   name=f"Short answer ({short_ms/1000:.1f} s each)"))
        fig4.add_trace(go.Scatter(x=qpm, y=fps_rag, mode="lines",
                                   line=dict(color="#F59E0B", width=3),
                                   name=f"RAG answer ({rag_ms/1000:.0f} s each)"))
        fig4.add_vline(x=queries_per_min, line_dash="dot", line_color="#6366F1",
                        annotation_text=f"current: {queries_per_min}/min",
                        annotation_position="top")
        fig4.add_hline(y=30, line_dash="dot", line_color="#93A1B5")
        fig4.add_hline(y=15, line_dash="dot", line_color="#93A1B5")
        fig4.update_layout(
            xaxis_title="LLM queries per minute",
            yaxis_title="Effective vision FPS",
            plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
            font=dict(color="#EAEDF4"),
            legend=dict(orientation="h", y=-0.2),
            height=420, margin=dict(l=40, r=40, t=20, b=40),
        )
        _apply_chart_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown(f"""
**Current LLM projection — {hw.name} @ {quant}:**

- Decode: **{llm['decode_tok_s']:.1f} tok/s**
- TTFT (1K prompt): **{llm['ttft_1k_sec']*1000:.0f} ms**
- Short 200-token answer: **{llm['short_answer_sec']:.1f} s**
- RAG (8K prompt + 2K response): **{llm['rag_total_sec']:.0f} s**
- GGUF size: **{llm['gguf_size_gb']:.1f} GB** {'✓ fits' if llm['fits_in_memory'] else '✗ spills (needs offload)'}

At **{queries_per_min}/min** of **{answer_kind}** answers, vision drops from
**{vision['fps_per_stream']:.1f}** → **{vision_fps_effective:.1f} FPS** per stream.
""")

with tab_detail:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Vision projection")
        st.json({k: v for k, v in vision.items()
                  if not isinstance(v, (dict, list))})
    with right:
        st.subheader("LLM projection")
        if llm_enabled:
            st.json({k: v for k, v in llm.items()
                      if not isinstance(v, (dict, list))})
        else:
            st.info("LLM not active — toggle in sidebar.")

    st.markdown("---")
    st.subheader("NCU measurement provenance")
    comps = measured_components(pipeline.key)
    if comps is None:
        st.info(
            f"No ncu measurement mapped for `{pipeline.key}`. "
            "The saturation approximation is the only figure available "
            "for this pipeline."
        )
    else:
        df_comps = pd.DataFrame([
            {
                "NVTX workload_id":     c["ncu_workload_id"],
                "Fires per frame":      c["fires_per_frame"],
                "DRAM bytes / fire":    c["dram_bytes_per_fire"],
                "DRAM MB / fire":       round(c["dram_bytes_per_fire"] / 1e6, 2),
                "n_forwards (bakeoff)": c["n_forwards_in_bakeoff"],
            } for c in comps
        ])
        st.dataframe(df_comps, use_container_width=True, hide_index=True)
        total_bytes = sum(
            c["dram_bytes_per_fire"] * c["fires_per_frame"] for c in comps
        )
        st.caption(
            f"Sum (per pipeline frame): **{total_bytes / 1e6:.1f} MB**. "
            "This is the hardware-neutral DRAM figure that transfers across "
            "NPU tiers — scale by pipeline FPS and compare against the NPU's "
            "effective bandwidth ceiling to get consumed GB/s."
        )
        meta = bundle_metadata()
        st.caption(
            f"Bundle exported **{meta['ncu_bundle_timestamp']}** on "
            f"*{meta['ncu_measurement_host']}*. "
            f"Total workloads in bundle: **{meta['ncu_n_workloads']}**. "
            f"Regenerate via `python scripts/export_ncu_for_sizer.py` in the "
            "`keyhole` repo after re-running `scripts/profile_all_ncu.sh`."
        )

    st.markdown("---")
    st.subheader("Hardware config")
    st.json({
        "name": hw.name,
        "bus": f"{hw.mem_bus_width_bits}-bit {hw.mem_type} @ {hw.mem_data_rate_gtps} GT/s",
        "mem_bandwidth_gbs_theoretical": round(hw.mem_bandwidth_gbs, 1),
        "mem_bandwidth_gbs_effective": round(hw.effective_bandwidth_gbs, 1),
        "mem_capacity_gb": hw.mem_capacity_gb,
        "peak_tops_bf16": hw.peak_tops_bf16,
        "peak_tops_fp8": hw.peak_tops_fp8,
        "compute_efficiency": hw.compute_efficiency,
        "bandwidth_efficiency": hw.bandwidth_efficiency,
        "tdp_watts": hw.tdp_watts,
    })

st.markdown("---")
st.caption(
    "keyhole-sizer — derived from the Keyhole bake-off series. "
    "All numbers trace back to measurements on an RTX 5090; edge projections "
    "use vendor-actual NPU tier data where available. "
    "See `github.com/kylefoxaustin/keyhole/REPRODUCE.md` to regenerate the "
    "underlying measurements."
)
