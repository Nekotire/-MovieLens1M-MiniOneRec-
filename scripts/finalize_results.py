#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", required=True)
parser.add_argument("--dataset", default="MovieLens1M")
parser.add_argument("--report", action="store_true")
args = parser.parse_args()
root = Path(args.results_dir)
sft = json.loads((root / "sft_metrics.json").read_text())
rl_path = root / "rl_metrics.json"
rl = json.loads(rl_path.read_text()) if rl_path.exists() else None
comparison = {"dataset": args.dataset, "sft": sft, "rl": rl}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

if args.report:
    lines = [
        f"# {args.dataset} MiniOneRec 实验报告",
        "",
        "以下指标由 pipeline 真实生成，不包含占位结果。",
        "",
    ]
    for name, values in (("SFT", sft), ("RL", rl)):
        lines.extend([f"## {name}", ""])
        if values is None:
            lines.append("未执行")
        else:
            lines.extend(f"- {key}: {value}" for key, value in values.items())
        lines.append("")
    (root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
