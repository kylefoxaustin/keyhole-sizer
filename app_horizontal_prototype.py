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
)
from sizer.project_vla import project_vla
from sizer.vla_models import VLA_MODELS
from sizer.llm_models import LLM_MODELS

st.set_page_config(page_title="keyhole-sizer · horizontal prototype",
                   layout="wide", initial_sidebar_state="collapsed")

# Highlight the picker popovers (Pipeline / Model / VLA model) in green so the
# "what am I configuring" control pops out from the neutral popovers around it.
# A green border + translucent fill reads correctly in BOTH light and dark
# browser themes (the fill tints whatever's behind it); button text is left
# theme-inherited so contrast is never broken in either mode.
st.markdown("""
<style>
.st-key-pop_pipe button, .st-key-pop_llm button, .st-key-pop_vla button {
    border: 1.5px solid #22A06B !important;
    background-color: rgba(34, 160, 107, 0.14) !important;
}
.st-key-pop_pipe button:hover, .st-key-pop_llm button:hover, .st-key-pop_vla button:hover {
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
            pk = st.selectbox("Pipeline", list(PIPELINES), key="p_pipe")
            res = st.segmented_control("Resolution", ["720p", "1080p", "4K"],
                                       default="1080p", key="p_res") or "1080p"
    pk = st.session_state.get("p_pipe", list(PIPELINES)[0])
    res = st.session_state.get("p_res", "1080p")
    vr = project_vision(PIPELINES[pk], hw, resolution=res, n_streams=n_cameras,
                        compiler_quality_vs_trt=compiler_quality, npu_share=npu_share)

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
    st.divider()

# ───────────────────────── LLM ─────────────────────────
if "LLM" in workloads:
    head, picker, _sp = st.columns([0.6, 0.9, 8.5])  # picker hugs the section name
    head.markdown("#### 🤖 LLM")
    with picker:
        with st.popover("Model ▾", use_container_width=True, key="pop_llm"):
            lkeys = list(LLM_MODELS)
            lk = st.selectbox("LLM model", lkeys,
                              index=lkeys.index("skippy_finetune"),
                              format_func=lambda k: LLM_MODELS[k].label.split(" (")[0],
                              key="p_llm")
    lk = st.session_state.get("p_llm", "skippy_finetune")
    alias = LLM_MODELS[lk].measurement_alias or lk
    lr = project_llm(hw, "Q4_K_M", workload="plain_chat", npu_share=npu_share, model_key=alias)

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
            tr = project_llm(TIERS[key], "Q4_K_M", workload="plain_chat",
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
            for col, (lab, ps) in zip(rc, [("INT-only", "int8"),
                                           ("+FP8", "int8_fp8"),
                                           ("+FP8+FP4", "int8_fp8_fp4")]):
                _m = mat if ps == "int8_fp8_fp4" else "mature"
                pr = project_llm(hw_with_precision(base_hw_p, ps), "Q4_K_M",
                                 workload="plain_chat", npu_share=npu_share,
                                 model_key=alias, fp4_runtime_maturity=_m)
                col.metric(lab, f"{pr['ttft_1k_sec']*1000:.0f} ms", delta="prefill", delta_color="off")
        else:
            st.caption("_Precision what-if available on Mid / High (FP-capable memory class)._")
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
