"""
Platform-budget CSV exporter — turn sizer projections into additive,
platform-level SoC budget rows that slot into a spreadsheet.

Each emitted row = ONE workload slot on the NPU (vision, LLM, ...). Platform
totals = sum across rows for the ss_* (steady-state) columns. peak_* columns
are NOT additive (two workloads don't peak at the same instant in steady state).

Used by:
  - app.py           : UI "Download platform budget CSV (current config)" button
  - scripts/export_platform_budget.py        : CLI for one specific config
  - scripts/export_platform_matrix.py        : full preset-HW × pipeline matrix
"""
from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

from .npu_model import (
    Hardware, VisionPipeline,
    project_vision, project_llm,
    BYTES_PER_PARAM, ACTIVE_PARAMS, GGUF_SIZE_GB,
    WORKLOAD_CATEGORIES,
)
from .measured import (
    measured_dram_per_frame,
    measured_components,
    bundle_metadata,
)


# ───────────────────────── Schema ─────────────────────────

CSV_COLUMNS = [
    # Identity — tells you which workload this row is and what config produced it
    "workload_id",
    "workload_category",     # 'vision' | 'llm'
    "config_summary",
    # Hardware — duplicated on every row so each row is self-contained
    "hw_name",
    "hw_bus_width_bits",
    "hw_mem_type",
    "hw_data_rate_gtps",
    "hw_ddr_gbs_theoretical",
    "hw_ddr_gbs_effective",
    "hw_peak_tops_bf16",
    "hw_peak_tops_fp8",
    "hw_dram_capacity_gb",
    "hw_tdp_watts",
    # Steady-state (1-second averages, ADDITIVE across rows)
    "ss_duty_cycle_frac",    # 0..1, fraction of NPU wall-clock this workload consumes
    "ss_ddr_gbs_avg",        # avg GB/s this workload pulls from DRAM (APPROXIMATED — saturation × duty)
    "ss_ddr_gbs_avg_measured", # avg GB/s from actual ncu DRAM bytes × fps (None if no ncu data)
    "ss_dram_per_forward_mb",  # per-frame DRAM footprint from ncu (None if no ncu data)
    "ss_tops_avg",            # avg effective TOPS consumed
    "ss_dram_resident_mb",    # MB resident — add across rows for total memory fit check
    "ss_power_avg_watts",     # TDP × duty approximation (NOT measured)
    "ss_throughput",          # FPS for vision, tok/s for LLM
    "ss_throughput_unit",     # 'fps_total' | 'tok_per_sec'
    # Per-frame peak — useful for sanity-checking no single workload exceeds HW ceiling
    "peak_per_frame_ms",      # per-frame latency (vision) or per-token (LLM)
    "peak_ddr_gbs",           # instantaneous GB/s during a single forward
    "peak_tops",              # instantaneous TOPS during a single forward
    # NCU provenance — for measured rows, lets the spreadsheet cite the sweep
    "ncu_source_workloads",   # comma-sep list of NVTX labels feeding this row
    "ncu_n_forwards",         # bake-off frame counts that produced the measurement
    "ncu_bundle_timestamp",   # when the sweep was exported
    # Provenance — lets the spreadsheet trace a row back to the sizer revision
    "sizer_commit_sha",
    "export_timestamp_iso",
]


HEADER_COMMENTS = [
    "# Keyhole sizer — platform-budget export",
    "# Each row = one workload slot on the NPU (vision or LLM).",
    "# Platform totals = sum across rows for the ss_* columns.",
    "#",
    "# CAVEATS (please read before using these numbers for procurement):",
    "#  - ss_power_avg_watts is a TDP × duty-cycle approximation, NOT measured per-workload.",
    "#  - NPU tier numbers (Low-LP4/Low-LP5X/Mid/High) are bandwidth-scaled from RTX 5090 measurements,",
    "#    NOT measured on actual NPU silicon. Real vendor numbers may differ by \u00b130%.",
    "#  - ss_* columns are additive across rows; peak_* columns are NOT",
    "#    (two workloads don't peak at the same instant in steady state).",
    "#  - Assumes workloads are bandwidth-bound (true for edge NPU at these model sizes);",
    "#    ss_ddr_gbs_avg = effective_DDR_BW \u00d7 duty_cycle under this assumption.",
    "#",
    "#  - ss_ddr_gbs_avg_measured (and ss_dram_per_forward_mb) come from MEASURED",
    "#    Nsight Compute counters via sizer/sizer_bundle.json (see sizer.measured).",
    "#    Blank for pipelines without a vendored ncu measurement.",
    "#    Prefer the _measured column over the saturation approximation when present.",
    "#",
    "#  - To consume in pandas:  pd.read_csv(path, comment='#')",
    "#",
]


# ───────────────────────── Helpers ─────────────────────────

def _current_commit_sha() -> str:
    """Short git SHA of the sizer repo, for provenance. Falls back to 'unknown'."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hw_columns(hw: Hardware) -> dict:
    return {
        "hw_name":               hw.name,
        "hw_bus_width_bits":     hw.mem_bus_width_bits,
        "hw_mem_type":           hw.mem_type,
        "hw_data_rate_gtps":     hw.mem_data_rate_gtps,
        "hw_ddr_gbs_theoretical": round(hw.mem_bandwidth_gbs, 2),
        "hw_ddr_gbs_effective":  round(hw.effective_bandwidth_gbs, 2),
        "hw_peak_tops_bf16":     hw.peak_tops_bf16,
        "hw_peak_tops_fp8":      hw.peak_tops_fp8,
        "hw_dram_capacity_gb":   hw.mem_capacity_gb,
        "hw_tdp_watts":          hw.tdp_watts,
    }


# ───────────────────────── Row builders ─────────────────────────

def vision_workload_row(
    pipeline: VisionPipeline,
    hw: Hardware,
    resolution: str,
    n_streams: int = 1,
    duty_cycle_frac: float = 1.0,
) -> dict:
    """One vision workload row at full (or budgeted) throughput.

    duty_cycle_frac = 1.0  →  NPU fully dedicated to this workload (max FPS)
    duty_cycle_frac = 0.5  →  only half the NPU time available, throughput halves
    """
    v = project_vision(pipeline, hw, resolution, n_streams=n_streams)

    # Throughput scales with duty cycle
    effective_fps_per_stream = v["fps_per_stream"] * duty_cycle_frac
    effective_total_fps = effective_fps_per_stream * n_streams

    # Approximated under BW-bound assumption: consumed DDR BW ≈ effective BW × duty cycle.
    ss_ddr_gbs_avg = hw.effective_bandwidth_gbs * duty_cycle_frac
    ss_tops_avg = (hw.peak_tops_fp8 * hw.compute_efficiency) * duty_cycle_frac
    ss_power_avg = hw.tdp_watts * duty_cycle_frac

    # Measured DRAM (from ncu) — preferred over the saturation approximation
    # for any pipeline mapped in measured.PIPELINE_TO_NCU. Falls back to None
    # otherwise; CSV rows show blank in the _measured columns.
    measured_dram_bytes_per_frame = measured_dram_per_frame(pipeline.key)
    if measured_dram_bytes_per_frame is not None:
        ss_ddr_gbs_avg_measured = (
            measured_dram_bytes_per_frame * effective_total_fps / 1e9
        )
        ss_dram_per_forward_mb = measured_dram_bytes_per_frame / 1e6
        comps = measured_components(pipeline.key) or []
        ncu_source_workloads = ", ".join(
            f"{c['ncu_workload_id']}×{c['fires_per_frame']:.4f}" for c in comps
        )
        ncu_n_forwards = ", ".join(str(c["n_forwards_in_bakeoff"]) for c in comps)
    else:
        ss_ddr_gbs_avg_measured = None
        ss_dram_per_forward_mb = None
        ncu_source_workloads = ""
        ncu_n_forwards = ""

    bundle_meta = bundle_metadata()

    config = (
        f"{pipeline.label} \u00b7 {n_streams} stream(s) @ {resolution} \u00b7 {hw.name}"
        + (f" \u00b7 duty {duty_cycle_frac:.2f}" if duty_cycle_frac < 1.0 else "")
    )

    return {
        "workload_id":        f"vision_{pipeline.key}_{resolution}_n{n_streams}",
        "workload_category":  "vision",
        "config_summary":     config,
        **_hw_columns(hw),
        "ss_duty_cycle_frac": round(duty_cycle_frac, 3),
        "ss_ddr_gbs_avg":     round(ss_ddr_gbs_avg, 2),
        "ss_ddr_gbs_avg_measured": (
            round(ss_ddr_gbs_avg_measured, 2)
            if ss_ddr_gbs_avg_measured is not None else ""
        ),
        "ss_dram_per_forward_mb": (
            round(ss_dram_per_forward_mb, 2)
            if ss_dram_per_forward_mb is not None else ""
        ),
        "ss_tops_avg":        round(ss_tops_avg, 1),
        "ss_dram_resident_mb": round(v["vram_mb"], 0),
        "ss_power_avg_watts": round(ss_power_avg, 1),
        "ss_throughput":      round(effective_total_fps, 2),
        "ss_throughput_unit": "fps_total",
        "peak_per_frame_ms":  round(v["per_stream_ms"], 2),
        "peak_ddr_gbs":       round(hw.effective_bandwidth_gbs, 2),
        "peak_tops":          round(hw.peak_tops_fp8 * hw.compute_efficiency, 1),
        "ncu_source_workloads":  ncu_source_workloads,
        "ncu_n_forwards":        ncu_n_forwards,
        "ncu_bundle_timestamp":  bundle_meta["ncu_bundle_timestamp"]
                                 if ncu_source_workloads else "",
        "sizer_commit_sha":   _current_commit_sha(),
        "export_timestamp_iso": _iso_now(),
    }


def llm_workload_row(
    hw: Hardware,
    quant: str,
    workload: str = "plain_chat",
    queries_per_minute: float = 2.0,
    answer_kind: str = "short",   # "short" (~200 tok) or "rag" (8K+2K)
) -> dict:
    """One LLM workload row with a specific query-rate and answer shape.

    duty_cycle is derived: qps × answer_sec. Saturates at 1.0 if the LLM
    demand exceeds NPU capacity (flagged via ss_throughput which stays at
    the ceiling while duty_cycle caps).
    """
    llm = project_llm(hw, quant, workload=workload)
    answer_sec = llm["short_answer_sec"] if answer_kind == "short" else llm["rag_total_sec"]
    qps = queries_per_minute / 60.0
    duty_raw = qps * answer_sec
    duty = min(1.0, duty_raw)

    # LLM decode BW: active_params × bytes/param × decode_tok/s (bytes/s) × duty
    bpp = BYTES_PER_PARAM[quant]
    active_gbs_at_full_decode = (ACTIVE_PARAMS * bpp * llm["decode_tok_s"]) / 1e9
    ss_ddr_gbs_avg = active_gbs_at_full_decode * duty

    ss_tops_avg = (hw.peak_tops_fp8 * hw.compute_efficiency) * duty
    ss_power_avg = hw.tdp_watts * duty

    config = (
        f"Qwen3-30B-A3B {quant} \u00b7 {WORKLOAD_CATEGORIES[workload]['label']} \u00b7 "
        f"{queries_per_minute:.1f} q/min \u00b7 {answer_kind} answer \u00b7 {hw.name}"
    )

    return {
        "workload_id":        f"llm_{quant}_{workload}_{answer_kind}",
        "workload_category":  "llm",
        "config_summary":     config,
        **_hw_columns(hw),
        "ss_duty_cycle_frac": round(duty, 3),
        "ss_ddr_gbs_avg":     round(ss_ddr_gbs_avg, 2),
        "ss_ddr_gbs_avg_measured": "",      # LLM not in current ncu sweep (B-prime skip)
        "ss_dram_per_forward_mb": "",
        "ss_tops_avg":        round(ss_tops_avg, 1),
        "ss_dram_resident_mb": round(llm["gguf_size_gb"] * 1024, 0),
        "ss_power_avg_watts": round(ss_power_avg, 1),
        "ss_throughput":      round(llm["decode_tok_s"], 2),
        "ss_throughput_unit": "tok_per_sec",
        "peak_per_frame_ms":  round(1000.0 / max(llm["decode_tok_s"], 1e-9), 2),
        "peak_ddr_gbs":       round(hw.effective_bandwidth_gbs, 2),
        "peak_tops":          round(hw.peak_tops_fp8 * hw.compute_efficiency, 1),
        "ncu_source_workloads":  "",
        "ncu_n_forwards":        "",
        "ncu_bundle_timestamp":  "",
        "sizer_commit_sha":   _current_commit_sha(),
        "export_timestamp_iso": _iso_now(),
    }


# ───────────────────────── CSV serialization ─────────────────────────

def rows_to_csv_str(rows: list[dict], include_header_comments: bool = True) -> str:
    """Serialize a list of dict rows to a CSV string.

    Header comments (prefixed with `#`) appear above the column header line.
    Most CSV consumers (pandas `comment='#'`, R `read.csv(... comment.char='#')`)
    will skip them. Excel users may see them as rows with `#` in column A —
    delete those 12 rows or use Text-to-Columns import.
    """
    out = StringIO()
    if include_header_comments:
        out.write("\n".join(HEADER_COMMENTS) + "\n")
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return out.getvalue()


def write_csv(rows: list[dict], out_path: Path, include_header_comments: bool = True) -> None:
    """Write rows to a CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rows_to_csv_str(rows, include_header_comments))
