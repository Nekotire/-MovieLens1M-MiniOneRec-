#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .datasets import EmbDataset
    from .models.rqvae import RQVAE
except ImportError:
    from datasets import EmbDataset
    from models.rqvae import RQVAE


def generate(data_path: Path, checkpoint: Path, output_file: Path, device: str = "cuda:0",
             batch_size: int = 64) -> dict[str, float | int]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    data = EmbDataset(str(data_path))
    model = RQVAE(in_dim=data.dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
                  layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn,
                  loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
                  beta=getattr(args, "beta", .25), kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters, sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters)
    model.load_state_dict(ckpt["state_dict"])
    target = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    model.to(target).eval()
    rows = []
    with torch.no_grad():
        for values in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
            rows.extend(model.get_indices(values.to(target), use_sk=False).view(values.shape[0], -1).cpu().tolist())
    names = ["a", "b", "c", "d", "e"]
    encoded = [[f"<{names[level]}_{int(value)}>" for level, value in enumerate(row)] for row in rows]
    if len(encoded) != len(data):
        raise RuntimeError("SID count does not match embedding count")
    counts = collections.Counter("".join(row) for row in encoded)
    stats = {"total_items": len(encoded), "unique_sid": len(counts),
             "collision_count": len(encoded) - len(counts),
             "collision_rate": (len(encoded) - len(counts)) / len(encoded) if encoded else 0.0,
             "max_collision_size": max(counts.values(), default=0)}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({str(i): value for i, value in enumerate(encoded)}, indent=2), encoding="utf-8")
    output_file.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    generate(Path(args.data_path), Path(args.checkpoint), Path(args.output_file), args.device, args.batch_size)


if __name__ == "__main__":
    main()
