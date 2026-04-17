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
    describe_hw, project_vision, project_llm,
    theoretical_bandwidth, vision_fps_under_llm_load,
)

st.set_page_config(
    page_title="keyhole-sizer",
    page_icon="🎯",
    layout="wide",
)

# ───────────────────────── Header ─────────────────────────

st.title("🎯 keyhole-sizer")
st.markdown(
    "Interactive sandbox for the Keyhole bake-off findings. Tweak the NPU spec, "
    "pipeline, concurrency, and LLM load — see live FPS / tok/s / duty-cycle "
    "projections. All numbers trace back to measured bake-offs ("
    "`github.com/kylefoxaustin/keyhole`, see `REPRODUCE.md`)."
)

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
        queries_per_min = 0.0
        answer_kind = "short"

# ───────────────────────── Main area ─────────────────────────

# Compute projections
vision = project_vision(pipeline, hw, resolution, n_streams=n_streams)
llm = project_llm(hw, quant) if llm_enabled else None

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
    label=f"Vision FPS / stream ({n_streams}-stream {resolution})",
    value=f"{vision_fps_effective:.1f}",
    delta=(f"{vision_fps_effective - vision['fps_per_stream']:+.1f}  under LLM"
            if llm_enabled else f"{vision['total_fps']:.1f} total"),
    delta_color="inverse" if llm_enabled else "normal",
)
c2.metric(
    label="Total system FPS",
    value=f"{vision_fps_effective * n_streams:.0f}",
    delta=f"{n_streams} streams",
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
        label="Bandwidth ratio vs NPU Mid",
        value=f"{vision['bandwidth_ratio_vs_ref']:.2f}×",
        delta="reference = NPU Mid",
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
            st.subheader("LLM decode tok/s vs NPU tier")
            tier_llm = []
            for name, t_hw in TIERS.items():
                l = project_llm(t_hw, quant)
                tier_llm.append(dict(tier=name, tok_s=l["decode_tok_s"]))
            if hw.name not in TIERS:
                tier_llm.append(dict(tier=hw.name, tok_s=llm["decode_tok_s"]))
            df_llm = pd.DataFrame(tier_llm)

            fig_llm_tier = go.Figure()
            fig_llm_tier.add_trace(go.Bar(
                x=df_llm["tier"], y=df_llm["tok_s"],
                marker=dict(color=["#EF4444", "#22C55E", "#6366F1", "#F59E0B"][:len(df_llm)]),
                text=[f"{t:.1f} tok/s" for t in df_llm["tok_s"]],
                textposition="auto",
            ))
            fig_llm_tier.update_layout(
                yaxis_title="Decode tok/s",
                plot_bgcolor="#0F192E", paper_bgcolor="#0F192E",
                font=dict(color="#EAEDF4"),
                height=320, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_llm_tier, use_container_width=True)
            st.caption(
                "Decode is bandwidth-bound on active params × bytes/param. "
                "MoE wins: only 3B of the 30B total are loaded per token. "
                "Q4_K_M / Q5_K_M / Q8_0 scale inversely with bytes/param."
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
