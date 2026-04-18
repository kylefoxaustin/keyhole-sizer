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
st.markdown(
    "Interactive sandbox for the Keyhole bake-off findings. Tweak the NPU spec, "
    "pipeline, concurrency, and LLM load — see live FPS / tok/s / duty-cycle "
    "projections. All numbers trace back to measured bake-offs ("
    "`github.com/kylefoxaustin/keyhole`, see `REPRODUCE.md`)."
)


# ───────────────────────── Helper: pipeline strip renderer ─────────────────────────

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
        options=("NPU Low", "NPU Mid", "NPU High", "Custom"),
        index=1,
        help="Low = 64-bit LPDDR4 baseline, Mid = 128-bit LPDDR5X (Keyhole shipping target), "
             "High = vendor high-bin, Custom = roll your own.",
    )

    if tier == "Custom":
        st.markdown("---")
        st.caption("Custom NPU")
        bus_width = st.select_slider(
            "Memory bus width (bits)",
            options=[64, 96, 128, 192, 256, 384, 512],
            value=128,
        )
        mem_type = st.selectbox("Memory type", options=MEMORY_TYPES, index=2)
        data_rate = st.slider("Data rate (GT/s)", min_value=2.0, max_value=32.0,
                               value=8.4, step=0.1)
        theoretical_bw = theoretical_bandwidth(bus_width, data_rate)
        st.caption(f"→ theoretical BW: **{theoretical_bw:.1f} GB/s**")
        bw_eff = st.slider("Bandwidth efficiency", 0.50, 0.95, 0.80, 0.01,
                            help="Fraction of theoretical BW realized on real workloads. "
                                 "0.80 is typical for modern NPUs.")
        # Both sliders share the same 50-1000 range so equal numeric values
        # sit at equal horizontal positions. FP8 on Blackwell-class silicon
        # is typically 2× BF16, but some NPUs are 1:1 or even missing FP8 —
        # decouple the sliders so the user can model either.
        tops_bf16 = st.slider("Peak BF16 TOPS", 50, 1000, 200, 10)
        tops_fp8 = st.slider("Peak FP8 TOPS", 0, 1000, min(1000, int(tops_bf16 * 2)), 10,
                              help="Set to 0 if silicon doesn't support FP8 natively. "
                                   "Blackwell-class is typically 2× BF16.")
        compute_eff = st.slider("Compute efficiency", 0.40, 0.85, 0.65, 0.01)
        mem_cap = st.slider("DRAM capacity (GB)", 2, 64, 8, 1)
        tdp = st.slider("TDP (W)", 2, 150, 25, 1)

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
    pipeline_key = st.selectbox(
        "Pipeline",
        options=list(PIPELINES.keys()),
        format_func=lambda k: PIPELINES[k].label,
        index=list(PIPELINES.keys()).index("trt_fp8_1hz_clip"),
    )
    pipeline = PIPELINES[pipeline_key]
    st.caption(pipeline.description)

    resolution = st.selectbox("Per-stream resolution", ("720p", "1080p", "4K"), index=0)
    n_streams = st.slider("Concurrent streams", 1, 16, 1, 1,
                           help="Each stream processes its own video source. YOLO batching "
                                "kicks in automatically (batch = N_streams).")

    st.markdown("---")
    st.header("LLM co-exist")
    llm_enabled = st.toggle("Share the NPU with a generative LLM",
                             value=False,
                             help="Qwen3-30B-A3B MoE (3B active / 30B total).")
    if llm_enabled:
        quant = st.selectbox("Qwen3 quantization",
                              ("Q4_K_M", "Q5_K_M", "Q8_0"), index=0)
        llm_workload = st.selectbox(
            "LLM workload pattern",
            options=list(WORKLOAD_CATEGORIES.keys()),
            format_func=lambda k: WORKLOAD_CATEGORIES[k]["label"],
            index=0,
            help="Real-world workload categories measured on Skippy production "
                 "(n=1-5 per category). Decode tok/s spans 3.6-222 across "
                 "categories — pick the one your deployment will actually see.",
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
        queries_per_min = st.slider("LLM queries per minute", 0.0, 60.0, 2.0, 0.1)
        answer_kind = st.radio("Typical answer length",
                                ("short", "rag"),
                                index=0,
                                format_func=lambda k: {
                                    "short": "Short (~200 tokens)",
                                    "rag":   "RAG (8K prompt + 2K response)"}[k],
                                horizontal=True)
    else:
        quant = "Q4_K_M"
        llm_workload = "plain_chat"
        queries_per_min = 0.0
        answer_kind = "short"

# ───────────────────────── Main area ─────────────────────────

# Compute projections
vision = project_vision(pipeline, hw, resolution, n_streams=n_streams)
llm = project_llm(hw, quant, workload=llm_workload) if llm_enabled else None

# ───────────────────────── Front-page summary + pipeline strip ─────────────────────────
# Dynamic "Simulating" line reflecting the current selection
if llm_enabled:
    wl_label = WORKLOAD_CATEGORIES[llm_workload]["label"]
    llm_summary = (
        f" · LLM **on** — Qwen3-30B-A3B **{quant}**, **{wl_label}** @ "
        f"**{queries_per_min:.1f} q/min** (**{answer_kind}** answers)"
    )
else:
    llm_summary = " · LLM **off**"

# ── Current-config header + export buttons ──
_cfg_col, _btn_cur_col, _btn_mat_col = st.columns([3.2, 1.4, 1.4])
with _cfg_col:
    st.markdown("##### 🔧 Configuration")

# Build the current-config rows (for the download button)
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
        label="📦 Full matrix",
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

st.markdown(
    f"**Simulating:** {pipeline.label} &nbsp;·&nbsp; "
    f"**{n_streams}** stream{'s' if n_streams != 1 else ''} "
    f"@ **{resolution}** &nbsp;·&nbsp; "
    f"**{hw.name}**{llm_summary}"
)

# Visual pipeline flow — reflects current pipeline + LLM selection
_render_pipeline_strip(
    _stages_for_pipeline(pipeline_key, llm_enabled, llm_workload, quant)
)
_legend_html = (
    '<div style="display:flex; flex-wrap:wrap; align-items:center; '
    'gap:20px; margin:4px 0 2px;">'
    '<div style="display:flex; align-items:center; gap:7px;">'
    '<span style="display:inline-block; width:16px; height:16px; '
    'background:#334155; border:1.5px solid #475569; border-radius:3px;"></span>'
    '<span style="font-size:13px; color:#C8D0E1;">'
    '<b>Always on</b> &nbsp;— ingest, storage</span></div>'
    '<div style="display:flex; align-items:center; gap:7px;">'
    '<span style="display:inline-block; width:16px; height:16px; '
    'background:#6366F1; border:1.5px solid #6366F1; border-radius:3px;"></span>'
    '<span style="font-size:13px; color:#C8D0E1;">'
    '<b>Pipeline stage changes</b> &nbsp;— varies with your choice</span></div>'
    '</div>'
    '<div style="font-size:12px; color:#93A1B5; margin-top:4px;">'
    'Every stage is running — the colors just flag where your controls take effect.'
    '</div>'
)
st.markdown(_legend_html, unsafe_allow_html=True)

# Visual break so the metric cards below read as "outputs of the whole
# pipeline", not as "one metric per block above". The horizontal rule +
# section label explicitly re-anchor the reader.
st.markdown(
    "<hr style='border:none; border-top:1px solid #334155; margin:20px 0 4px;'>",
    unsafe_allow_html=True,
)
st.markdown("##### 📊 Projected results — the whole pipeline combined")

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

# Saturation banner — the LLM request rate alone exceeds what one NPU can deliver.
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

# ───────── Top metric row ─────────
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(pipeline.note)
        if pipeline.key in {"trt_fp8_1hz_clip", "trt_fp8_every_frame",
                             "hybrid_v2_bf16", "hybrid_v2_torchao_fp8",
                             "yolo_only_fp8"}:
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
            v = project_vision(pipeline, t_hw, resolution, n_streams=n_streams)
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
        st.plotly_chart(fig2, use_container_width=True)

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
        v = project_vision(pipeline, hw, resolution, n_streams=N)
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
