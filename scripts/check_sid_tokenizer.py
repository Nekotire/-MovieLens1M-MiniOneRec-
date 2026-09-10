#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minionerec_utils.constrained_decoding import build_sid_prefix_tree, encode_without_special_tokens


def validate_sid_tokenizer(
    tokenizer, indices: dict[str, list[str]], seed: int, sample_size: int, require_existing: bool = False
) -> None:
    ordered = [indices[key] for key in sorted(indices, key=int)]
    if not ordered or any(len(sid) != 3 for sid in ordered):
        raise ValueError("Semantic IDs must be non-empty and contain exactly three layers")
    new_tokens = sorted({token for sid in ordered for token in sid})
    if require_existing:
        vocabulary = tokenizer.get_vocab()
        missing = [token for token in new_tokens if token not in vocabulary]
        if missing:
            raise ValueError(f"Saved tokenizer is missing {len(missing)} SID tokens; first missing token: {missing[0]}")
    else:
        tokenizer.add_tokens(new_tokens)
    token_ids: dict[str, int] = {}
    for token in new_tokens:
        encoded = encode_without_special_tokens(tokenizer, token)
        expected = tokenizer.convert_tokens_to_ids(token)
        if len(encoded) != 1 or encoded[0] != expected:
            raise ValueError(f"SID token is not atomic after tokenizer extension: {token} -> {encoded}")
        token_ids[token] = expected

    rng = random.Random(seed)
    sample = ordered if len(ordered) <= sample_size else rng.sample(ordered, sample_size)
    semantic_ids = ["".join(sid) for sid in sample]
    tree = build_sid_prefix_tree(tokenizer, semantic_ids)
    for sid, text in zip(sample, semantic_ids):
        encoded = encode_without_special_tokens(tokenizer, text)
        expected = [token_ids[token] for token in sid]
        if encoded != expected:
            raise ValueError(f"SID does not encode to the expected three tokens: {text} -> {encoded}")
        prefix: list[int] = []
        for token_id in encoded:
            if token_id not in tree.allowed(prefix):
                raise ValueError(f"Prefix tree rejected legal SID: {text}")
            prefix.append(token_id)
        if tokenizer.eos_token_id not in tree.allowed(prefix):
            raise ValueError(f"Prefix tree cannot terminate legal SID: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen tokenizer and SID constrained decoding.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--require-existing", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    indices = json.loads(Path(args.index).read_text(encoding="utf-8"))
    validate_sid_tokenizer(tokenizer, indices, args.seed, args.sample_size, args.require_existing)
    print(f"SID tokenizer preflight passed: model={args.model}, sampled={min(len(indices), args.sample_size)}")


if __name__ == "__main__":
    main()
