#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minionerec_utils.metrics import write_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--info", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--stop-on-invalid", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    metrics = write_metrics(Path(args.predictions), Path(args.info), Path(args.output), tuple(args.cutoffs))
    print(json.dumps(metrics, indent=2))
    if args.stop_on_invalid and not metrics["valid_experiment"]:
        raise SystemExit("ERROR: CC != 0; evaluation is invalid and the pipeline is stopped.")
