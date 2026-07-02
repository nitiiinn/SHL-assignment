"""CLI entrypoint for the SHL evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation import evaluate_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the public SHL traces, score retrieval/recommendation metrics, "
            "and run behavior probes."
        )
    )
    parser.add_argument(
        "--traces-dir",
        default=str(Path(__file__).resolve().parents[1] / "example traces"),
        help="Directory containing the public markdown traces.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Directory containing processed assessments and retrieval indexes.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the full JSON report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = evaluate_project(
        traces_dir=args.traces_dir,
        data_dir=args.data_dir,
    )
    payload = report.to_dict()
    output_path = Path(args.output) if args.output else Path(__file__).with_name("eval-report.json")

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"Full report written to {output_path}")


if __name__ == "__main__":
    main()
