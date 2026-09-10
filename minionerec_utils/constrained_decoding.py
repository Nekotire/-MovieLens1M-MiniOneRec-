from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class PrefixNode:
    children: dict[int, "PrefixNode"] = field(default_factory=dict)
    terminal: bool = False


class TokenPrefixTree:
    def __init__(self, paths: Iterable[Sequence[int]], eos_token_id: int):
        self.root = PrefixNode()
        self.eos_token_id = eos_token_id
        for path in paths:
            node = self.root
            for token in path:
                node = node.children.setdefault(int(token), PrefixNode())
            node.terminal = True

    def allowed(self, prefix: Sequence[int]) -> list[int]:
        node = self.root
        for token in prefix:
            node = node.children.get(int(token))
            if node is None:
                return []
        allowed = list(node.children)
        if node.terminal:
            allowed.append(self.eos_token_id)
        return sorted(set(allowed))


def encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids)


def generation_prefix_length(tokenizer, response_prefix: str = "### Response:\n") -> int:
    return len(encode_without_special_tokens(tokenizer, response_prefix))


def build_sid_prefix_tree(tokenizer, semantic_ids: Iterable[str], eos_token_id: int | None = None) -> TokenPrefixTree:
    eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    return TokenPrefixTree((encode_without_special_tokens(tokenizer, sid) for sid in semantic_ids), eos)
