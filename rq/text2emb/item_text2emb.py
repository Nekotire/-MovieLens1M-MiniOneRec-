#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ordered_item_texts(item_file: Path) -> list[str]:
    items = json.loads(item_file.read_text(encoding="utf-8"))
    expected = [str(i) for i in range(len(items))]
    if sorted(items, key=int) != expected:
        raise ValueError("Item IDs must be exactly 0..num_items-1")
    return [f'{items[key].get("title", "")} {items[key].get("description", "")}'.strip() for key in expected]


def validate_embeddings(embeddings: np.ndarray, num_items: int, expected_dim: int | None = None) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != num_items:
        raise ValueError(f"Expected {num_items} embedding rows, got {embeddings.shape}")
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {embeddings.shape[1]}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf")


def encode(item_file: Path, output_file: Path, model_name: str, batch_size: int = 64,
           max_length: int = 512, expected_dim: int | None = None, dtype: str = "float16") -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = ordered_item_texts(item_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported embedding dtype: {dtype}")
    model_dtype = dtype_map[dtype] if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=model_dtype)
    model.to(device).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(batches, axis=0)
    validate_embeddings(embeddings, len(texts), expected_dim)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--expected-dim", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    encode(Path(args.item_file), Path(args.output_file), args.model, args.batch_size,
           args.max_length, args.expected_dim, args.dtype)


if __name__ == "__main__":
    main()
