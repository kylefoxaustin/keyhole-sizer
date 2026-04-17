"""
Full-matrix platform-budget export.

Iterates the Cartesian product of preset HW tiers, pipelines, resolutions,
stream counts, and (separately) LLM quant × workload combinations. Custom HW
is NOT included — it's user-defined so has no fixed set of points to
enumerate. For custom HW, use the UI download button or the CLI.

Output: data/platform_budget_matrix.csv

Rows: ~540 (vision + LLM)

Usage:
    python scripts/export_platform_matrix.py
    python scripts/export_platform_matrix.py --out /tmp/budget.csv --no-llm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sizer.npu_model import PIPELINES, TIERS, BYTES_PER_PARAM, WORKLOAD_CATEGORIES
from sizer.platform_budget import (
    vision_workload_row, llm_workload_row, write_csv,
)

RESOLUTIONS = ["720p", "1080p", "4K"]
STREAM_COUNTS = [1, 2, 4, 8, 16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "platform_budget_matrix.csv",
                    help="Output CSV path")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM rows (vision matrix only)")
    ap.add_argument("--no-vision", action="store_true",
                    help="Skip vision rows (LLM matrix only)")
    ap.add_argument("--qpm", type=float, default=2.0,
                    help="LLM queries/min for every LLM row (default 2.0)")
    args = ap.parse_args()

    rows: list[dict] = []

    if not args.no_vision:
        print("Enumerating vision workloads...", file=sys.stderr)
        for hw in TIERS.values():
            for pipeline in PIPELINES.values():
                for res in RESOLUTIONS:
                    for n in STREAM_COUNTS:
                        try:
                            rows.append(vision_workload_row(pipeline, hw, res, n_streams=n))
                        except Exception as e:
                            print(f"  SKIP {hw.name} × {pipeline.key} × {res} × n={n}: {e}",
                                  file=sys.stderr)

    if not args.no_llm:
        print("Enumerating LLM workloads...", file=sys.stderr)
        for hw in TIERS.values():
            for quant in BYTES_PER_PARAM:
                for workload in WORKLOAD_CATEGORIES:
                    for answer_kind in ("short", "rag"):
                        try:
                            rows.append(llm_workload_row(
                                hw, quant, workload=workload,
                                queries_per_minute=args.qpm,
                                answer_kind=answer_kind,
                            ))
                        except Exception as e:
                            print(f"  SKIP {hw.name} × {quant} × {workload} × {answer_kind}: {e}",
                                  file=sys.stderr)

    write_csv(rows, args.out)
    print(f"\nWrote {len(rows)} rows → {args.out}", file=sys.stderr)
    print(f"\nBreakdown:", file=sys.stderr)
    for cat in ("vision", "llm"):
        n = sum(1 for r in rows if r["workload_category"] == cat)
        print(f"  {cat:<8s} {n:>4d} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
