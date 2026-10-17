#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation entrypoint (template)"
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to a run directory."
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # TODO: load checkpoint, run evaluation, save metrics
    print(f"Evaluating run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
