"""keyhole-sizer — HORIZONTAL LAYOUT PROTOTYPE (Step 2, 2026-06-06).

A throwaway layout mockup for Kyle to judge against the live vertical-sidebar
app. Run side-by-side:

    streamlit run app.py                       # current (tall left sidebar)
    streamlit run app_horizontal_prototype.py  # this (top control strip, wide)

Uses the REAL engine (live numbers). Demonstrates: no left sidebar; controls in
a horizontal top strip (tier pills + popovers + workload pills + a ⚙ settings
popover for power-controls); results + charts fill the full page width; KPIs
visible onscreen (not download-only). Detail fidelity is representative — if you
like it, Step 3 is migrating app.py onto this shell.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sizer.npu_model import (
    Hardware, TIERS, NPU_MID, MEMORY_TYPES, PIPELINES,
    project_vision, project_llm, hw_with_memory, hw_with_precision,
    MEMORY_UPGRADE_OPTIONS, describe_hw, bandwidth_ratio, theoretical_bandwidth,
    WORKLOAD_CATEGORIES, workload_distribution_on_hw, capability_level,
)
from sizer.project_vla import project_vla
from sizer.vla_models import VLA_MODELS
from sizer.llm_models import (
    LLM_MODELS, CATEGORY_LABELS, accuracy_delta_pp, PRODUCTION_REFERENCE_KEY,
    perf_scale_factor, METHODOLOGY_VERSION,
)
from sizer.llm_quant_levels import (
    LLM_QUANT_LADDER, W8A8_VS_FP16_CATEGORY_DELTAS,
    QWEN_W8A8_RAG, FP16_REFERENCE, delta_pp_vs_fp16,
)
from sizer.measured import (
    measured_dram_per_frame, measured_components, bundle_metadata,
)

st.set_page_config(page_title="keyhole-sizer · horizontal prototype",
                   layout="wide", initial_sidebar_state="collapsed")

# Highlight the picker popovers (Pipeline / Model / VLA model) in green so the
# "what am I configuring" control pops out from the neutral popovers around it.
# A green border + translucent fill reads correctly in BOTH light and dark
# browser themes (the fill tints whatever's behind it); button text is left
# theme-inherited so contrast is never broken in either mode.
st.markdown("""
<style>
.st-key-pop_pipe button, .st-key-pop_llm button, .st-key-pop_quant button,
.st-key-pop_work button, .st-key-pop_vla button {
    border: 1.5px solid #22A06B !important;
    background-color: rgba(34, 160, 107, 0.14) !important;
}
.st-key-pop_pipe button:hover, .st-key-pop_llm button:hover, .st-key-pop_quant button:hover,
.st-key-pop_work button:hover, .st-key-pop_vla button:hover {
    border-color: #1B7E54 !important;
    background-color: rgba(34, 160, 107, 0.24) !important;
}
</style>
""", unsafe_allow_html=True)

# ── short tier label → TIERS key (the comparison ladder, in order) ──
_TIER_MAP = {
    "i.MX 95": "NPU i.MX 95 (ground truth)",
    "Low-LP5X": "NPU Low-LP5X",
    "Mid": "NPU Mid",
    "High": "NPU High",
    "RTX 5090": "RTX 5090 (reference, measured)",
}
_SHARE_MAP = {"100%": 1.0, "75%": 0.75, "50%": 0.5, "25%": 0.25}
_ACCENT = "#E1483A"   # keyhole red — highlight the selected tier in charts


def _per_tier_bar(values: dict[str, float], y_title: str, selected: str):
    """Horizontal-friendly bar of a metric across the tier ladder; the selected
    tier is accented. `values` keyed by short tier label."""
    fig = go.Figure(go.Bar(
        x=list(values), y=list(values.values()),
        marker_color=[_ACCENT if k == selected else "#9AA7BD" for k in values],
        text=[f"{v:.1f}" for v in values.values()], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white", height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=y_title, showlegend=False,
    )
    return fig


# ── Vision pipeline narrative tracks (mirrors app.py::PIPELINE_TRACKS).
# (label, [pipeline keys], canonical-default-key). The track radio lives
# inside the Pipeline ▾ popover so the flat 23-pipeline list reads as the
# deck's optimization journey instead of an undifferentiated dropdown. ──
PIPELINE_TRACKS = [
    ("SAM 3 lineage",
     ["sam3_bf16", "essmall_fp8",
      "efficientsam3_es_ev_s_bf16", "efficientsam3p1_es_ev_s_bf16"], "sam3_bf16"),
    ("One-model open-vocab",
     ["yoloe26_s_pf_fp16", "yoloe26_s_pf_trt_fp8"], "yoloe26_s_pf_trt_fp8"),
    ("Default (Hybrid V2 → TRT)",
     ["hybrid_v2_bf16", "hybrid_v2_torchao_fp8",
      "trt_fp8_every_frame", "trt_fp8_1hz_clip", "yolo_only_fp8"], "trt_fp8_1hz_clip"),
    ("YOLOv8n nano",
     ["yolov8n_trt_fp8_every_frame", "yolov8n_trt_fp8_1hz_clip",
      "yolov8n_only_fp8"], "yolov8n_trt_fp8_1hz_clip"),
    ("INT8 vendor-comparison",
     ["yolo11s_trt_int8", "yolov8n_trt_int8_coco128", "resnet50v1_int8_224",
      "yolov8n_trt_int4_coco128", "resnet50v1_int4_224"], "yolov8n_trt_int8_coco128"),
    ("ViT alternatives (what-if)",
     ["rtdetr_l_pytorch_fp16", "detr_resnet50_pytorch_fp16",
      "owlv2_base_pytorch_fp16", "grounding_dino_tiny_pytorch_fp32"],
     "owlv2_base_pytorch_fp16"),
]
_DEFAULT_TRACK_INDEX = 2  # "Default (Hybrid V2 → TRT)"


def _render_pipeline_strip(stages: list[tuple[str, bool]]):
    """Horizontal pipeline flow mirroring the Keyhole deck's exec summary.
    Highlighted boxes are indigo; dim boxes are neutral. Ported from app.py."""
    box_highlight = ("background:#6366F1; color:#FFFFFF; border:1.5px solid #6366F1; "
                     "font-weight:600;")
    box_dim = "background:#334155; color:#EAEDF4; border:1.5px solid #475569;"
    arrow = ('<div style="display:flex; align-items:center; color:#6366F1; '
             'font-size:24px; padding:0 4px;">→</div>')
    parts = []
    for i, (label, hl) in enumerate(stages):
        style = box_highlight if hl else box_dim
        parts.append(
            f'<div style="{style} border-radius:10px; padding:10px 14px; '
            f'min-width:115px; flex:1; text-align:center; font-size:12.5px; '
            f'line-height:1.35; white-space:pre-line;">{label}</div>')
        if i < len(stages) - 1:
            parts.append(arrow)
    st.markdown('<div style="display:flex; flex-wrap:nowrap; align-items:stretch; '
                f'gap:2px; margin:12px 0 8px;">{"".join(parts)}</div>',
                unsafe_allow_html=True)


def _stages_for_pipeline(pipeline_key: str, llm_enabled: bool,
                         llm_workload: str, quant: str) -> list[tuple[str, bool]]:
    """Build the 5-stage flow based on current selections. Ported from app.py."""
    mapping = {
        "sam3_bf16":             ("YOLO 11x", False, "SAM 3 BF16", True),
        "essmall_fp8":           ("YOLO 11x", False, "EfficientSAM-Small FP8", True),
        "efficientsam3_es_ev_s_bf16": ("YOLO 11x", False, "EfficientSAM3 ES-EV-S\nBF16", True),
        "efficientsam3p1_es_ev_s_bf16": ("(text-prompt)", False, "EfficientSAM3.1 ES-EV-S\nBF16 (n=1 concept)", True),
        "yoloe26_s_pf_fp16":     ("YOLOE-26S-PF FP16\n(one model)", True, "(open-vocab built-in)", False),
        "yoloe26_s_pf_trt_fp8":  ("YOLOE-26S-PF TRT FP8\n(optimized ceiling)", True, "(open-vocab built-in)", False),
        "hybrid_v2_bf16":        ("YOLO-seg BF16", True, "CLIP BF16", True),
        "hybrid_v2_torchao_fp8": ("YOLO-seg BF16", True, "CLIP FP8 (torchao)", True),
        "trt_fp8_every_frame":   ("YOLO-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\nevery frame", True),
        "trt_fp8_1hz_clip":      ("YOLO-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\n@ 1 Hz", True),
        "yolo_only_fp8":         ("YOLO-seg FP8 (TRT)", True, "(no CLIP)", False),
        "yolov8n_trt_fp8_every_frame": ("yolov8n-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\nevery frame", True),
        "yolov8n_trt_fp8_1hz_clip":    ("yolov8n-seg FP8 (TRT)", True, "CLIP FP8 (TRT)\n@ 1 Hz", True),
        "yolov8n_only_fp8":            ("yolov8n-seg FP8 (TRT)", True, "(no CLIP)", False),
        "yolo11s_trt_int8":            ("yolo11s-seg INT8 (TRT)\n20-frame PTQ", True, "(no CLIP)", False),
        "yolov8n_trt_int8_coco128":    ("yolov8n-seg INT8 (TRT)\ncoco128-seg PTQ", True, "(no CLIP)", False),
        "yolov8n_trt_int4_coco128":    ("yolov8n-seg INT4-w (TRT)\ncoco128-seg PTQ", True, "(no CLIP)", False),
        "resnet50v1_int4_224":         ("ResNet-50 INT4-w (TRT)\nImageNet 224×224", True, "(no CLIP)", False),
    }
    det_label, det_hl, enr_label, enr_hl = mapping.get(
        pipeline_key, ("?", False, "?", False))
    stages: list[tuple[str, bool]] = [
        ("FFmpeg\ningest", False), (det_label, det_hl),
        (enr_label, enr_hl), ("SQLite\n+ FTS5", False),
    ]
    if llm_enabled:
        wl_label = WORKLOAD_CATEGORIES[llm_workload]["label"]
        stages.append((f"Qwen3-30B-A3B {quant}\n{wl_label}", True))
    else:
        stages.append(("NLQ / LLM\n(off)", False))
    return stages


# ───────────────────────── TOP CONTROL STRIP ─────────────────────────
st.markdown("### 🎯 keyhole-sizer  ·  &nbsp;_horizontal-layout prototype_", unsafe_allow_html=True)

with st.container(border=True):
    # ── Row 1: the frequently-touched controls. All three are LABELED widgets,
    # so their label-tops line up on one baseline (NPU tier ↔ Workloads ↔
    # Cameras). Workloads sits immediately right of the tier and carries an emoji
    # so it reads as the primary "what am I sizing" control; Cameras hugs it. ──
    r1 = st.columns([2.4, 1.3, 1.9, 3.4])
    with r1[0]:
        tier_label = st.segmented_control(
            "NPU tier", options=list(_TIER_MAP), default="High", key="p_tier",
            help="Silicon target — horizontal pills, not a sidebar dropdown.",
        ) or "High"
    base_tier = tier_label
    hw = TIERS[_TIER_MAP[tier_label]]

    with r1[1]:
        workloads = st.pills(
            "🧩 Workloads", options=["Vision", "LLM", "VLA"],
            selection_mode="multi", default=["Vision", "VLA"], key="p_workloads",
        ) or []
    with r1[2]:
        n_cameras = st.number_input(
            "📷 Choose # of cameras", 1, 8, 1, key="p_ncam", width=160,
            help="Number of camera feeds the NPU drives in parallel.")

    # ── Row 2: the tuning knobs. All three are POPOVERS, so their buttons share
    # one baseline (this is what fixes the Settings-vs-Cameras misalignment — the
    # old layout mixed a label-less popover into a row of labeled inputs). ──
    r2 = st.columns([1.1, 1.1, 1.1, 5.7])
    with r2[0]:
        if tier_label in ("Mid", "High"):
            with st.popover("Memory ▾", use_container_width=True):
                opts = ["Stock"] + [o[0] for o in MEMORY_UPGRADE_OPTIONS]
                mc = st.radio("Memory upgrade", opts, index=0, key="p_mem")
                if mc != "Stock":
                    o = next(o for o in MEMORY_UPGRADE_OPTIONS if o[0] == mc)
                    hw = hw_with_memory(hw, o[1], o[2], name_suffix=f"{o[1]}-{o[2]:.0f}")
        else:
            st.popover("Memory ▾", use_container_width=True, disabled=True,
                       help="Memory upgrades apply to Mid / High only.")

    with r2[1]:
        with st.popover("BW share ▾", use_container_width=True):
            share_label = st.segmented_control(
                "NPU BW share", options=list(_SHARE_MAP), default="75%", key="p_share",
            ) or "75%"
    npu_share = _SHARE_MAP[share_label]

    # ⚙ Settings popover — the "rarely touched power controls" home (Kyle's call:
    # a settings button instead of a sidebar). Custom NPU builder + global knobs.
    with r2[2]:
        with st.popover("⚙ Settings", use_container_width=True):
            use_custom = st.toggle("Use a custom NPU (override tier)", key="p_custom_on")
            if use_custom:
                c_bus = st.select_slider("Bus width (bits)",
                                         [64, 96, 128, 192, 256, 384, 512], 128, key="p_c_bus")
                c_rate = st.slider("Data rate (GT/s)", 2.0, 32.0, 8.4, 0.1, key="p_c_rate")
                c_tops = st.slider("INT8 TOPS", 50, 1000, 200, 10, key="p_c_tops")
                c_cap = st.slider("DRAM (GB)", 2, 64, 16, 1, key="p_c_cap")
                c_tdp = st.slider("TDP (W)", 2, 150, 25, 1, key="p_c_tdp")
                c_bw = theoretical_bandwidth(c_bus, c_rate)
                hw = Hardware(
                    name="Custom NPU", peak_tops_bf16=c_tops, peak_tops_int8=c_tops,
                    peak_tops_fp8=c_tops, mem_bandwidth_gbs=c_bw, mem_capacity_gb=c_cap,
                    mem_bus_width_bits=c_bus, mem_type="LPDDR5X", mem_data_rate_gtps=c_rate,
                    compute_efficiency=0.65, bandwidth_efficiency=0.70, tdp_watts=c_tdp,
                )
                base_tier = None
            st.divider()
            compiler_quality = st.slider("Vision compiler quality", 0.5, 1.0, 1.0, 0.01,
                                         key="p_cq", help="Haircut vs TRT-ideal kernels.")
            show_kpis = st.toggle("Show KPI table onscreen", value=True, key="p_kpi_on")

st.caption(describe_hw(hw))
st.divider()

if not workloads:
    st.info("Pick one or more workloads in the strip above.")

# ───────────────────────── VISION ─────────────────────────
if "Vision" in workloads:
    head, picker, _sp = st.columns([1.4, 0.9, 7.7])  # picker hugs the section name
    head.markdown("#### 📹 Vision pipeline")
    with picker:
        with st.popover("Pipeline ▾", use_container_width=True, key="pop_pipe"):
            # Two-step pick: narrative track radio → track-scoped selectbox
            # (each track remembers its own pick via key=f"p_pipe__{track}").
            track_label = st.radio("Pipeline track", [t[0] for t in PIPELINE_TRACKS],
                                   index=_DEFAULT_TRACK_INDEX, key="p_track")
            _, _tkeys, _canon = next(t for t in PIPELINE_TRACKS if t[0] == track_label)
            st.selectbox("Pipeline", _tkeys, index=_tkeys.index(_canon),
                         format_func=lambda k: PIPELINES[k].label,
                         key=f"p_pipe__{track_label}")
            res = st.segmented_control("Resolution", ["720p", "1080p", "4K"],
                                       default="1080p", key="p_res") or "1080p"
    track_label = st.session_state.get("p_track", PIPELINE_TRACKS[_DEFAULT_TRACK_INDEX][0])
    _, _tkeys, _canon = next(t for t in PIPELINE_TRACKS if t[0] == track_label)
    pk = st.session_state.get(f"p_pipe__{track_label}", _canon)
    res = st.session_state.get("p_res", "1080p")
    vr = project_vision(PIPELINES[pk], hw, resolution=res, n_streams=n_cameras,
                        compiler_quality_vs_trt=compiler_quality, npu_share=npu_share)

    st.caption(f"📹 **{PIPELINES[pk].label}** — {PIPELINES[pk].description}")

    m = st.columns([1.1, 1.1, 1.1, 1.1, 3.6])  # cluster the 4 metrics left; spacer eats the rest
    m[0].metric("Per-camera FPS", f"{vr.get('fps_per_stream', vr.get('total_fps', 0.0)):.1f}")
    m[1].metric("Aggregate FPS", f"{vr.get('total_fps', 0.0):.0f}",
                delta=f"× {n_cameras} cam", delta_color="off")
    m[2].metric("Memory fit", "✓ fits" if vr.get("fits_in_memory") else "✗ spills")
    m[3].metric("DDR bandwidth ratio",
                f"{vr.get('bandwidth_ratio_vs_ref', bandwidth_ratio(hw)):.2f}×",
                delta="vs NPU Mid", delta_color="off")

    # 2-up: per-tier FPS comparison + a compact timing/regime readout.
    g1, g2 = st.columns([3, 2])
    with g1:
        per_tier = {}
        for lbl, key in _TIER_MAP.items():
            tr = project_vision(PIPELINES[pk], TIERS[key], resolution=res,
                                n_streams=1, compiler_quality_vs_trt=compiler_quality,
                                npu_share=npu_share)
            per_tier[lbl] = tr.get("fps_per_stream", tr.get("total_fps", 0.0))
        st.plotly_chart(_per_tier_bar(per_tier, "FPS / camera", base_tier),
                        use_container_width=True, key="v_tier")
    with g2:
        st.caption("**Timing**")
        st.metric("Per-frame latency",
                  f"{vr.get('per_stream_ms', 0.0):.2f} ms", delta=vr.get("regime", "?"),
                  delta_color="off")
        st.caption(f"source: `{vr.get('edge_ms_source', '?')}` · BW floor "
                   f"{vr.get('bw_floor_ms', 0):.2f} ms · compute floor "
                   f"{vr.get('compute_floor_ms', 0):.2f} ms · "
                   f"DRAM {vr.get('vram_mb', 0):.0f} MB/frame")

    # ── scoped depth tabs (Stream scaling / DRAM bandwidth / Pipeline flow) ──
    vt_stream, vt_bw, vt_flow = st.tabs(
        ["Stream scaling", "DRAM bandwidth", "Pipeline flow"])

    with vt_stream:
        srows = []
        for N in [1, 2, 4, 8, 16]:
            v = project_vision(PIPELINES[pk], hw, resolution=res, n_streams=N,
                               compiler_quality_vs_trt=compiler_quality, npu_share=npu_share)
            srows.append({"N streams": N,
                          "Per-stream FPS": round(v["fps_per_stream"], 1),
                          "Total system FPS": round(v["total_fps"], 1),
                          "Batch cycle ms": round(v["per_stream_ms"], 1),
                          "VRAM (MB)": round(v["vram_mb"], 0),
                          "Fits": "✓" if v["fits_in_memory"] else "✗"})
        df_s = pd.DataFrame(srows)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=df_s["N streams"], y=df_s["Per-stream FPS"],
                                       mode="lines+markers", line=dict(color="#6366F1", width=3),
                                       marker=dict(size=9), name="Per-stream FPS"))
            fig_s.add_trace(go.Scatter(x=df_s["N streams"], y=df_s["Total system FPS"],
                                       mode="lines+markers",
                                       line=dict(color="#22C55E", width=3, dash="dash"),
                                       marker=dict(size=9), name="Total system FPS", yaxis="y2"))
            fig_s.add_hline(y=30, line_dash="dot", line_color="#93A1B5",
                            annotation_text="30 FPS real-time")
            fig_s.update_layout(template="plotly_white", height=340,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Concurrent streams",
                                yaxis=dict(title="Per-stream FPS"),
                                yaxis2=dict(title="Total system FPS", overlaying="y", side="right"),
                                legend=dict(orientation="h", y=-0.28))
            st.plotly_chart(fig_s, use_container_width=True, key="v_streams")
        with c2:
            st.dataframe(df_s, width="stretch", hide_index=True)
            st.caption("YOLO batching amortizes kernel overhead — 4 streams at batch=4 "
                       "typically get ~70% of single-stream FPS, not 25%.")

    with vt_bw:
        mbpf = measured_dram_per_frame(pk)
        eff_total_fps = vr["fps_per_stream"] * n_cameras
        approx = hw.effective_bandwidth_gbs
        if mbpf is None:
            st.info(f"No ncu measurement mapped for **{PIPELINES[pk].label}** yet — "
                    "only the saturation approximation is available for this pipeline.")
        else:
            meas = mbpf * eff_total_fps / 1e9
            fig_bw = go.Figure(go.Bar(
                x=["Saturation model<br>(CSV ss_ddr_gbs_avg)",
                   "Measured (ncu)<br>ss_ddr_gbs_avg_measured"],
                y=[approx, meas], marker_color=["#EF4444", "#22C55E"],
                text=[f"{approx:.1f} GB/s", f"{meas:.2f} GB/s"], textposition="outside"))
            fig_bw.add_hline(y=approx, line_dash="dot", line_color="#93A1B5",
                             annotation_text=f"{hw.name} ceiling ({approx:.1f} GB/s)",
                             annotation_position="top right")
            fig_bw.update_layout(template="plotly_white", height=320,
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 yaxis_title="DRAM GB/s consumed by vision pipeline",
                                 showlegend=False)
            st.plotly_chart(fig_bw, use_container_width=True, key="v_bw")
            util = (meas / approx * 100) if approx > 0 else 0
            st.markdown(f"**Per-frame DRAM:** {mbpf/1e6:.1f} MB · **Pipeline FPS:** "
                        f"{eff_total_fps:.1f} · **Measured usage:** {meas:.2f} GB/s "
                        f"({util:.1f}% of ceiling) · **Spare:** {max(0.0, approx-meas):.1f} GB/s")
            comps = measured_components(pk) or []
            if len(comps) > 1:
                parts = " + ".join(f"`{c['ncu_workload_id']}` × {c['fires_per_frame']:.3g} "
                                   f"({c['dram_bytes_per_fire']/1e6:.1f} MB/fire)" for c in comps)
                st.caption(f"Composition: {parts}")
            meta = bundle_metadata()
            st.caption("Saturation = pessimistic bus-pin assumption; measured = ncu DRAM "
                       "bytes/forward × FPS. The gap is real headroom for concurrent LLM / "
                       f"extra streams. ncu bundle `{meta['ncu_bundle_timestamp']}` · "
                       f"{meta['ncu_n_workloads']} workloads · host *{meta['ncu_measurement_host']}*.")

    with vt_flow:
        _llm_on = "LLM" in workloads
        _wl = st.session_state.get("p_work", "plain_chat")
        _q = st.session_state.get("p_quant", "Q4_K_M")
        _render_pipeline_strip(_stages_for_pipeline(pk, _llm_on, _wl, _q))
        st.caption("Indigo = stage driven by your pipeline / LLM choice; slate = always-on "
                   "infrastructure (ingest, storage). Every stage runs — the colour just "
                   "flags where your controls take effect.")
    st.divider()

# ───────────────────────── LLM ─────────────────────────
if "LLM" in workloads:
    # Header strip: the three GREEN pickers (Model / Quant / Workload — the
    # "what am I configuring" controls) + a neutral Duty popover for the
    # duty-cycle inputs (queries/min + answer length), which feed the
    # cross-workload Duty-cycle view rather than the LLM headline.
    head, pm, pq, pw, pdu, _sp = st.columns([0.6, 0.9, 1.0, 1.2, 0.9, 5.4])
    head.markdown("#### 🤖 LLM")
    with pm:
        with st.popover("Model ▾", use_container_width=True, key="pop_llm"):
            lkeys = list(LLM_MODELS)
            st.selectbox("LLM model", lkeys, index=lkeys.index("skippy_finetune"),
                         format_func=lambda k: LLM_MODELS[k].label.split(" (")[0],
                         key="p_llm")
    with pq:
        with st.popover("Quant ▾", use_container_width=True, key="pop_quant"):
            st.selectbox("Quantization", ("Q4_K_M", "Q5_K_M", "Q8_0"), index=0,
                         key="p_quant")
    with pw:
        with st.popover("Workload ▾", use_container_width=True, key="pop_work"):
            st.selectbox("Workload pattern", list(WORKLOAD_CATEGORIES), index=0,
                         format_func=lambda k: WORKLOAD_CATEGORIES[k]["label"],
                         key="p_work")
            wl_now = st.session_state.get("p_work", "plain_chat")
            st.caption(WORKLOAD_CATEGORIES[wl_now]["description"])
    with pdu:
        with st.popover("Duty ▾", use_container_width=True):
            st.slider("Queries / min", 0.0, 60.0, 2.0, 0.1, key="p_qpm")
            st.radio("Answer length", ("short", "rag"), index=0,
                     format_func=lambda k: {"short": "Short (~200 tok)",
                                            "rag": "RAG (8K + 2K)"}[k], key="p_ans")

    lk = st.session_state.get("p_llm", "skippy_finetune")
    quant = st.session_state.get("p_quant", "Q4_K_M")
    llm_workload = st.session_state.get("p_work", "plain_chat")
    alias = LLM_MODELS[lk].measurement_alias or lk
    _model = LLM_MODELS[lk]
    _prod = LLM_MODELS[PRODUCTION_REFERENCE_KEY]
    _is_production = (lk == PRODUCTION_REFERENCE_KEY)
    lr = project_llm(hw, quant, workload=llm_workload, npu_share=npu_share,
                     model_key=alias)

    st.caption(f"🤖 **{_model.label.split(' (')[0]}** · {quant} · workload "
               f"**{WORKLOAD_CATEGORIES[llm_workload]['label']}**")

    m = st.columns([1.1, 1.1, 1.1, 1.1, 3.6])  # cluster the 4 metrics left; spacer eats the rest
    m[0].metric("Decode", f"{lr['decode_tok_s']:.1f} tok/s")
    m[1].metric("TTFT (1K)", f"{lr['ttft_1k_sec']*1000:.0f} ms")
    m[2].metric("Memory fit", "✓ fits" if lr["fits_in_memory"] else "✗ spills",
                delta=f"{lr['gguf_size_gb']:.1f} GB", delta_color="off")
    m[3].metric("DDR bandwidth ratio", f"{bandwidth_ratio(hw):.2f}×",
                delta="vs NPU Mid", delta_color="off")

    g1, g2 = st.columns([2, 3])
    with g1:
        per_tier = {}
        for lbl, key in _TIER_MAP.items():
            tr = project_llm(TIERS[key], quant, workload=llm_workload,
                             npu_share=npu_share, model_key=alias)
            per_tier[lbl] = tr["decode_tok_s"]
        st.plotly_chart(_per_tier_bar(per_tier, "decode tok/s", base_tier),
                        use_container_width=True, key="l_tier")
    with g2:
        # The precision what-if compare (Mid/High) — the validated feature, in
        # its full-width home now instead of buried in a sidebar.
        if base_tier in ("Mid", "High"):
            st.caption("**🎛️ Precision what-if — if this NPU added FP8 / FP4**")
            mat = st.radio("FP4 runtime", ["Immature (edge)", "Mature (vLLM/TRT)"],
                           index=0, horizontal=True, key="p_fp4mat")
            mat = "immature" if mat.startswith("Immature") else "mature"
            base_hw_p = TIERS[_TIER_MAP[base_tier]]
            rc = st.columns(3)
            base_ttft = None
            for col, (lab, ps) in zip(rc, [("INT-only", "int8"),
                                           ("+FP8", "int8_fp8"),
                                           ("+FP8+FP4", "int8_fp8_fp4")]):
                _mt = mat if ps == "int8_fp8_fp4" else "mature"
                pr = project_llm(hw_with_precision(base_hw_p, ps), quant,
                                 workload=llm_workload, npu_share=npu_share,
                                 model_key=alias, fp4_runtime_maturity=_mt)
                tt = pr['ttft_1k_sec'] * 1000
                if ps == "int8":
                    base_ttft = tt
                sp = (f" · {base_ttft/tt:.1f}× vs INT8"
                      if base_ttft and ps != "int8" and tt < base_ttft else "")
                col.metric(lab, f"{tt:.0f} ms", delta="prefill", delta_color="off")
                col.caption(f"decode {pr['decode_tok_s']:.0f} tok/s{sp}")
        else:
            st.caption("_Precision what-if available on Mid / High (FP-capable memory class)._")

    # ── scoped depth tabs (Accuracy / Precision / Performance / Timing) ──
    t_acc, t_prec, t_perf, t_tim = st.tabs(
        ["Accuracy", "Precision", "Performance", "Timing"])

    with t_acc:
        if _model.pass_rate is None:
            st.info(f"**{_model.label}** — perf-reference variant (same weights, "
                    "alternate compute_dtype to reach an anchor cell). No standalone "
                    "eval; pick a production / fine-tune / baseline row for accuracy.")
        else:
            st.markdown(
                f"**{_model.label}** — {_model.pass_rate*100:.1f}% pass "
                f"({_model.pass_n_passes}/{_model.pass_n_total})  ·  base "
                f"{_model.base}  ·  {_model.total_params_b:.0f}B / "
                f"{_model.active_params_b:.0f}B active")
            cat_rows = []
            for _k, _mm in LLM_MODELS.items():
                d = accuracy_delta_pp(_mm, _prod)
                ds = ("— (ref)" if _k == PRODUCTION_REFERENCE_KEY
                      else "perf ref" if d is None
                      else f"{'+' if d >= 0 else ''}{d:.1f}pp")
                cat_rows.append({
                    "Model": (("➤ " if _k == lk else "") + _mm.label.split(' (')[0]),
                    "Pass": f"{_mm.pass_rate*100:.1f}%" if _mm.pass_rate is not None else "—",
                    "Δ vs prod": ds,
                    "Arch": f"{_mm.total_params_b:.0f}B/{_mm.active_params_b:.0f}B",
                })
            st.dataframe(pd.DataFrame(cat_rows), width="stretch", hide_index=True)
            if _model.category_deltas:
                _pc = _prod.category_deltas or {}
                st.markdown("**Per-category** (production reference):" if _is_production
                            else "**Per-category** (Δ vs production, + = this model wins):")
                for cat, data in _model.category_deltas.items():
                    lab = CATEGORY_LABELS.get(cat, cat)
                    p, n, rate = data.get("pass", 0), data.get("n", 0), data.get("rate", 0.0)
                    pdat = _pc.get(cat)
                    if pdat and not _is_production:
                        dl = p - pdat.get("pass", 0)
                        st.markdown(f"- {lab}: **{p}/{n}** ({rate:.0%}) — Δ {'+' if dl >= 0 else ''}{dl}")
                    else:
                        st.markdown(f"- {lab}: **{p}/{n}** ({rate:.0%})")
            with st.expander("📐 Eval methodology — Finding 4 (Qwen-family format bias)"):
                st.markdown(
                    "Headline uses **semantic grading** (GPT-4o binary, 132-sample "
                    "v2-RAG, temp=0). The production model's substring lift eroded "
                    "across five successive cross-checks:")
                st.markdown(
                    "| # | Cross-check | Result |\n|---|---|---|\n"
                    "| 1 | Substring (original) | **+3.1pp** |\n"
                    "| 2 | LLM-judge (Sonnet 4.6) | −0.35 |\n"
                    "| 3 | Temp=0.3 substring | −29.3pp |\n"
                    "| 4 | Cross-judge (GPT-4o) | −0.69 |\n"
                    "| 5 | **Semantic regrade** | **−4.6pp** (sign reversal) |")
                st.caption(
                    "Production decision unaffected — Skippy 7B v4 ships on the "
                    "three-gate framework (capability + voice + safety); substring "
                    "was never load-bearing. Full deck narrative carries from app.py. "
                    f"Methodology `{METHODOLOGY_VERSION}`.")

    with t_prec:
        _blocked = (hw.capability_levels is not None
                    and capability_level(hw, "int8") == "tensor_compat")
        st.markdown(
            f"**Quality cost of the quant recipe** (Qwen2.5-14B + RAG, v2 prompt set, "
            f"132 samples). fp16 reference: **{FP16_REFERENCE.pass_rate*100:.1f}%** "
            f"({FP16_REFERENCE.pass_n_passes}/{FP16_REFERENCE.pass_n_total}).")
        prec_rows = []
        for cfg in LLM_QUANT_LADDER:
            d = delta_pp_vs_fp16(cfg)
            ds = "—" if cfg.key == FP16_REFERENCE.key else f"{d:+.1f}pp"
            lab = cfg.label
            if cfg.key == QWEN_W8A8_RAG.key and _blocked:
                lab = f"⚠ {lab} — n/a on {hw.name}"
            prec_rows.append({"Configuration": lab,
                              "Pass": f"{cfg.pass_rate*100:.1f}%",
                              "Δ vs fp16": ds,
                              "n": f"{cfg.pass_n_passes}/{cfg.pass_n_total}",
                              "Host": cfg.measurement_host})
        st.dataframe(pd.DataFrame(prec_rows), width="stretch", hide_index=True)
        if _blocked:
            st.warning(
                "**W8A8 INT8 is ecosystem-blocked on this tier.** Consumer Blackwell "
                "SM120 throws `RuntimeError: Int8 not supported on SM120` — the W8A8 "
                "row is an H100 measurement kept for the deck story, not achievable here.")
        st.markdown("**Where the W8A8 −3.8pp regression lives** (vs fp16 base):")
        for cat, dp in W8A8_VS_FP16_CATEGORY_DELTAS.items():
            lab = CATEGORY_LABELS.get(cat, cat)
            st.markdown(f"- {lab}: **±0** (no drift)" if dp == 0
                        else f"- {lab}: **{'+' if dp > 0 else ''}{dp} passes**")
        st.caption(
            "Coding + reasoning byte-identical fp16↔W8A8 (structured output untouched "
            "by INT8); regression localizes in retrieval-grounded wording. Deck framing: "
            "W8A8 ~−3.8pp vs fp16; base→FT adds ~5pp, so a fine-tuned W8A8 lands near "
            "the fp16 base — lifecycle cost dominates, not the ~4pp hit.")

    with t_perf:
        st.markdown(
            "All five patterns measured on **Qwen3-30B-A3B-Instruct-2507** (Q4_K_M, "
            "llama.cpp) on an **RTX 5090** — decode spans **3.6 → 222 tok/s** across "
            "real traffic (~60×), which single-number vendor benchmarks miss.")
        dist = workload_distribution_on_hw(hw, quant)
        f = perf_scale_factor(_model)
        if f != 1.0:
            dist = [{**d, "decode_tok_s": d["decode_tok_s"] * f} for d in dist]
        d_labels = [f"{d['label']}  (n={d['n']})" for d in dist]
        d_values = [d["decode_tok_s"] for d in dist]
        d_colors = ["#22A06B" if d["key"] == llm_workload else "#9AA7BD" for d in dist]
        fig_dist = go.Figure(go.Bar(
            y=d_labels, x=d_values, orientation="h", marker_color=d_colors,
            text=[f"{v:.1f}" for v in d_values], textposition="outside"))
        fig_dist.update_layout(template="plotly_white", height=260,
                               margin=dict(l=10, r=30, t=10, b=10),
                               xaxis_title=f"decode tok/s on {hw.name} @ {quant}")
        st.plotly_chart(fig_dist, use_container_width=True, key="l_dist")
        mx = max(d_values); mn = min(v for v in d_values if v > 0)
        st.caption(
            f"Selected workload highlighted (green). Spread **{mx/mn:.0f}× worst-case** "
            f"({mx:.0f} → {mn:.1f} tok/s) on this HW+quant. Edge capacity planning "
            f"should budget for the RAG / tool-use tail, not the plain-chat peak.")

    with t_tim:
        sp_ms = lr["ttft_1k_sec"] * 1000
        sd_ms = (200 / lr["decode_tok_s"]) * 1000 if lr["decode_tok_s"] > 0 else 0
        rp_ms = lr["rag_prefill_sec"] * 1000
        rd_ms = lr["rag_decode_sec"] * 1000
        fig_tim = go.Figure()
        fig_tim.add_trace(go.Bar(name="Prefill", x=["Short (1K, 200 tok)", "RAG (8K+2K)"],
                                 y=[sp_ms, rp_ms], marker_color="#F59E0B"))
        fig_tim.add_trace(go.Bar(name="Decode", x=["Short (1K, 200 tok)", "RAG (8K+2K)"],
                                 y=[sd_ms, rd_ms], marker_color="#6366F1"))
        fig_tim.update_layout(barmode="stack", template="plotly_white", height=300,
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="Per-answer latency (ms)",
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_tim, use_container_width=True, key="l_tim")
        st.caption(f"TTFT 1K = **{lr['ttft_1k_sec']*1000:.0f} ms** · "
                   f"decode = **{lr['decode_tok_s']:.1f} tok/s** · answer mode from Duty ▾")
    st.divider()

# ───────────────────────── VLA ─────────────────────────
if "VLA" in workloads:
    head, picker, _sp = st.columns([0.6, 0.9, 8.5])  # picker hugs the section name
    head.markdown("#### 🦾 VLA")
    with picker:
        with st.popover("VLA model ▾", use_container_width=True, key="pop_vla"):
            vkeys = list(VLA_MODELS)
            vk = st.selectbox("VLA model", vkeys, index=vkeys.index("nora_3b"),
                              format_func=lambda k: VLA_MODELS[k].display_name, key="p_vla")
    vk = st.session_state.get("p_vla", "nora_3b")
    r = project_vla(VLA_MODELS[vk], hw, npu_share=npu_share, n_cameras=n_cameras)

    if r.get("deferred"):
        st.info(f"**{VLA_MODELS[vk].display_name}** — projection deferred: {r['reason']}")
    elif not r.get("runs"):
        st.error(f"**{VLA_MODELS[vk].display_name} won't run on {hw.name}** — {r['reason']}")
    else:
        m = st.columns([1.1, 1.1, 1.1, 1.1, 3.6])  # cluster the 4 metrics left; spacer eats the rest
        m[0].metric("Per-camera FPS", f"{r['camera_fps']:.1f}")
        m[1].metric("Aggregate FPS", f"{r['aggregate_camera_fps']:.0f}",
                    delta=f"× {r['n_cameras']} cam", delta_color="off")
        m[2].metric("Memory fit", "✓ fits" if r["fits_in_memory"] else "✗ spills",
                    delta=f"{r['dram_gb']:.1f} GB", delta_color="off")
        m[3].metric("DDR bandwidth ratio",
                    f"{hw.effective_bandwidth_gbs/NPU_MID.effective_bandwidth_gbs:.2f}×",
                    delta="vs NPU Mid", delta_color="off")
        st.caption(f"🤖 `{r['architecture']}` · control loop **{r['action_hz']:.1f} Hz** "
                   f"({r['ms_per_action']:.0f} ms/action)")

        g1, g2 = st.columns([3, 2])
        with g1:
            per_tier = {}
            for lbl, key in _TIER_MAP.items():
                tr = project_vla(VLA_MODELS[vk], TIERS[key], npu_share=npu_share,
                                 n_cameras=n_cameras)
                per_tier[lbl] = tr.get("camera_fps", 0.0) if tr.get("runs") else 0.0
            st.plotly_chart(_per_tier_bar(per_tier, "per-camera FPS", base_tier),
                            use_container_width=True, key="w_tier")
        with g2:
            with st.expander("ℹ️ Control-loop detail", expanded=True):
                st.markdown(f"`{r['regime']}` · **{r['action_hz']:.1f} Hz** "
                            f"({r['ms_per_action']:.0f} ms/action)")
                _bw = r.get("ddr_bw_demand_gbs")
                if _bw is not None:
                    st.markdown(f"🚌 ~**{_bw:.0f} GB/s** avg DDR demand "
                                f"({_bw/r.get('ddr_bw_available_gbs',1)*100:.0f}% of available)")
    st.divider()

# ───────────────────────── KPIs ONSCREEN ─────────────────────────
if workloads and st.session_state.get("p_kpi_on", True):
    st.markdown("#### 📊 KPIs — current configuration")
    rows = []
    if "Vision" in workloads:
        rows.append({"Workload": f"Vision · {pk}", "Per-cam FPS": round(vr.get('fps_per_stream', 0.0), 1),
                     "Aggregate FPS": round(vr.get('total_fps', 0.0), 0),
                     "Fits": "✓" if vr.get("fits_in_memory") else "✗",
                     "BW ratio vs Mid": f"{vr.get('bandwidth_ratio_vs_ref', 1.0):.2f}×"})
    if "LLM" in workloads:
        rows.append({"Workload": f"LLM · {LLM_MODELS[lk].label.split(' (')[0]}",
                     "Per-cam FPS": None, "Aggregate FPS": None,
                     "Fits": "✓" if lr["fits_in_memory"] else "✗",
                     "BW ratio vs Mid": f"{bandwidth_ratio(hw):.2f}×",
                     "Decode tok/s": round(lr["decode_tok_s"], 1),
                     "TTFT ms": round(lr["ttft_1k_sec"] * 1000, 0)})
    if "VLA" in workloads and r.get("runs"):
        rows.append({"Workload": f"VLA · {VLA_MODELS[vk].display_name}",
                     "Per-cam FPS": round(r["camera_fps"], 1),
                     "Aggregate FPS": round(r["aggregate_camera_fps"], 0),
                     "Fits": "✓" if r["fits_in_memory"] else "✗",
                     "BW ratio vs Mid": f"{hw.effective_bandwidth_gbs/NPU_MID.effective_bandwidth_gbs:.2f}×",
                     "Control Hz": round(r["action_hz"], 1)})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Export KPIs (CSV)", df.to_csv(index=False),
                           "keyhole_kpis.csv", "text/csv", key="p_kpi_dl")

st.divider()
st.caption("⬑ **Prototype** — control strip + full-width results/charts + onscreen KPIs, "
           "no sidebar. Tier across all silicon highlighted in red. If the layout works, "
           "Step 3 ports the live app.py onto this shell.")
