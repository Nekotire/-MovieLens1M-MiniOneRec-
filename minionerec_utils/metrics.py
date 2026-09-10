from __future__ import annotations

import json
import math
from pathlib import Path


def recommendation_metrics(records: list[dict], valid_items: set[str], cutoffs=(1, 3, 5, 10, 20, 50)) -> dict:
    hits = {k: 0 for k in cutoffs}
    ndcg = {k: 0.0 for k in cutoffs}
    invalid = 0
    for row in records:
        predictions = [str(value).strip() for value in row.get("predict", [])]
        invalid += sum(value not in valid_items for value in predictions)
        target = str(row.get("output", row.get("target", row.get("item_sid", "")))).strip('"\n ')
        for k in cutoffs:
            if target in predictions[:k]:
                rank = predictions.index(target) + 1
                hits[k] += 1
                ndcg[k] += 1.0 / math.log2(rank + 1)
    total = len(records)
    result = {f"HR@{k}": hits[k] / total if total else 0.0 for k in cutoffs}
    result.update({f"NDCG@{k}": ndcg[k] / total if total else 0.0 for k in cutoffs})
    result.update({"invalid_item_count": invalid, "CC": invalid, "valid_experiment": invalid == 0,
                   "num_examples": total})
    return result


def write_metrics(records_path: Path, info_path: Path, output_path: Path, cutoffs=(1, 3, 5, 10, 20, 50)) -> dict:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    valid = {line.split("\t", 1)[0].strip() for line in info_path.read_text(encoding="utf-8").splitlines() if line}
    metrics = recommendation_metrics(records, valid, cutoffs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
