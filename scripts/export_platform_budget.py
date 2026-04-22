"""
CLI — emit one platform-budget CSV for a specific config.

Examples:
    # Vision only (shipping stack @ 720p, 8 streams on NPU Mid):
    python scripts/export_platform_budget.py \
        --pipeline trt_fp8_1hz_clip --hw npu_mid --res 720p --streams 8 \
        --out budget.csv

    # Vision + LLM co-existing:
    python scripts/export_platform_budget.py \
        --pipeline trt_fp8_1hz_clip --hw npu_mid --res 720p --streams 4 \
        --llm-quant Q4_K_M --llm-workload tool_use --llm-qpm 2 --answer-kind short \
        --out budget.csv

    # Dump to stdout instead of a file:
    python scripts/export_platform_budget.py --pipeline sam3_bf16 --hw npu_low --res 720p

Available pipeline keys, HW tiers, quants, and workloads are printed with
`--list`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sizer.npu_model import PIPELINES, TIERS, BYTES_PER_PARAM, WORKLOAD_CATEGORIES
from sizer.platform_budget import (
    vision_workload_row, llm_workload_row, write_csv, rows_to_csv_str,
)

HW_SLUGS = {
    "npu_low_lp5":  "NPU Low-LP5",
    "npu_low_lp5x": "NPU Low-LP5X",
    "npu_mid":      "NPU Mid",
    "npu_mid_int8": "NPU Mid-INT8",
    "npu_high":     "NPU High",
    # Backwards-compat aliases: `npu_low` and `npu_low_lp4` keep resolving
    # to the entry-tier NPU (now spec'd as LPDDR5 @ 6.4 GT/s after the
    # 2026-04-22 correction — previously mis-spec'd as LPDDR4 @ 4.0 GT/s).
    "npu_low":      "NPU Low-LP5",
    "npu_low_lp4":  "NPU Low-LP5",
}


def main():
    ap = argparse.ArgumentParser(
        description="Export a platform-budget CSV row for a specific Keyhole-sizer config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--list", action="store_true", help="List valid pipeline/HW/quant/workload keys and exit")
    ap.add_argument("--pipeline", help="Pipeline key (see --list)")
    ap.add_argument("--hw", choices=list(HW_SLUGS.keys()), help="Hardware tier")
    ap.add_argument("--res", choices=["720p", "1080p", "4K"], help="Per-stream resolution")
    ap.add_argument("--streams", type=int, default=1, help="Concurrent streams (default 1)")
    ap.add_argument("--duty", type=float, default=1.0, help="Vision duty-cycle fraction 0..1 (default 1.0 = saturated)")
    ap.add_argument("--llm-quant", choices=list(BYTES_PER_PARAM.keys()),
                    help="If set, also emit an LLM row at this quant")
    ap.add_argument("--llm-workload", default="plain_chat",
                    choices=list(WORKLOAD_CATEGORIES.keys()),
                    help="LLM workload category (default plain_chat)")
    ap.add_argument("--llm-qpm", type=float, default=2.0,
                    help="LLM queries per minute (default 2.0)")
    ap.add_argument("--answer-kind", choices=["short", "rag"], default="short",
                    help="LLM answer shape (default short = ~200 tok)")
    ap.add_argument("--out", type=Path, help="Output CSV path. Omit to print to stdout.")
    ap.add_argument("--no-header-comments", action="store_true",
                    help="Omit the leading # comment block (useful for appending to another CSV)")
    args = ap.parse_args()

    if args.list:
        print("Pipelines:")
        for k, p in PIPELINES.items():
            print(f"  {k:40s}  {p.label}")
        print("\nHardware tiers:")
        for slug, name in HW_SLUGS.items():
            print(f"  {slug:10s} -> {name}")
        print("\nLLM quants:")
        for q in BYTES_PER_PARAM:
            print(f"  {q}")
        print("\nLLM workloads:")
        for k, w in WORKLOAD_CATEGORIES.items():
            print(f"  {k:25s}  {w['label']}")
        return

    # Validate required args for actual export
    required = ["pipeline", "hw", "res"]
    missing = [a for a in required if getattr(args, a) is None]
    if missing:
        ap.error(f"Missing required arg(s): {', '.join('--' + a for a in missing)}. Run with --list to see valid values.")

    if args.pipeline not in PIPELINES:
        ap.error(f"Unknown --pipeline {args.pipeline!r}. Run with --list to see valid keys.")

    pipeline = PIPELINES[args.pipeline]
    hw = TIERS[HW_SLUGS[args.hw]]

    rows = [vision_workload_row(
        pipeline, hw, args.res,
        n_streams=args.streams,
        duty_cycle_frac=args.duty,
    )]
    if args.llm_quant:
        rows.append(llm_workload_row(
            hw, args.llm_quant,
            workload=args.llm_workload,
            queries_per_minute=args.llm_qpm,
            answer_kind=args.answer_kind,
        ))

    if args.out:
        write_csv(rows, args.out, include_header_comments=not args.no_header_comments)
        print(f"Wrote {len(rows)} row(s) → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(rows_to_csv_str(rows, include_header_comments=not args.no_header_comments))


if __name__ == "__main__":
    main()
