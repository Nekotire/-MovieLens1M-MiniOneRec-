import re

from minionerec_utils.constrained_decoding import build_sid_prefix_tree, generation_prefix_length


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def test_prefix_tree_accepts_every_legal_sid_and_eos():
    tokenizer = CharacterTokenizer()
    sids = ["<a_1><b_2><c_3>", "<a_1><b_4><c_5>"]
    tree = build_sid_prefix_tree(tokenizer, sids)
    for sid in sids:
        tokens = tokenizer(sid)["input_ids"]
        prefix = []
        for token in tokens:
            assert token in tree.allowed(prefix)
            prefix.append(token)
        assert tokenizer.eos_token_id in tree.allowed(prefix)


def test_prefix_tree_rejects_illegal_path_and_prefix_length_is_dynamic():
    tokenizer = CharacterTokenizer()
    tree = build_sid_prefix_tree(tokenizer, ["<a_1><b_2><c_3>"])
    assert tree.allowed([999999]) == []
    assert generation_prefix_length(tokenizer) == len("### Response:\n")


def test_semantic_id_token_names_are_three_legal_layers():
    sid = ["<a_1>", "<b_2>", "<c_3>"]
    assert [re.fullmatch(fr"<{layer}_\d+>", token) is not None for layer, token in zip("abc", sid)] == [True] * 3


def test_registered_sid_tokens_are_atomic():
    class AddedTokenTokenizer(CharacterTokenizer):
        def __init__(self):
            self.added = set()

        def add_tokens(self, tokens):
            self.added.update(tokens)

        def __call__(self, text, add_special_tokens=False):
            if text in self.added:
                return {"input_ids": [hash(text) % 100000 + 1]}
            return super().__call__(text, add_special_tokens)

    tokenizer = AddedTokenTokenizer()
    tokens = ["<a_1>", "<b_2>", "<c_3>"]
    tokenizer.add_tokens(tokens)
    assert all(len(tokenizer(token, add_special_tokens=False)["input_ids"]) == 1 for token in tokens)
