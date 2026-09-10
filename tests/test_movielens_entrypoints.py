from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_RUNTIME_TERMS = (
    "data/Amazon", "Industrial_and_Scientific", "Office_Products",
    "amazon18", "amazon23", "reviewerID", "asin",
)


def test_formal_entrypoint_is_movielens_only():
    config = yaml.safe_load((REPO / "config/movielens1m.yaml").read_text())
    assert config["dataset"] == "MovieLens1M"
    assert config["download"]["url"] == "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    pipeline = (REPO / "scripts/pipeline.py").read_text()
    labels = [
        "Environment Check", "MovieLens Download", "Data Preprocessing", "Item Embedding",
        "RQ-VAE", "Semantic ID", "Dataset Conversion", "SFT", "SFT Evaluation",
        "Recommendation RL", "RL Evaluation", "Comparison", "Final Report",
    ]
    assert all(label in pipeline for label in labels)
    assert "config/movielens1m.yaml" in (REPO / "run.sh").read_text() or "scripts/pipeline.py" in (REPO / "run.sh").read_text()


def test_no_legacy_dataset_or_runtime_reference_remains():
    assert not (REPO / "data/Amazon").exists()
    assert not any((REPO / "data").glob("amazon*"))
    runtime_files = [
        *REPO.glob("*.py"), *REPO.glob("*.sh"), *REPO.glob("config/*.yaml"),
        *REPO.glob("data/*.py"), *REPO.glob("rq/*.py"), *REPO.glob("rq/*.sh"),
        *REPO.glob("rq/text2emb/*.py"), *REPO.glob("scripts/*.py"), *REPO.glob("scripts/*.sh"),
        *REPO.glob("minionerec_utils/*.py"),
    ]
    for path in runtime_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in FORBIDDEN_RUNTIME_TERMS:
            assert term not in text, f"legacy dataset reference {term!r} remains in {path}"


def test_optional_branches_expose_movielens_paths_and_shells_parse():
    expected = {
        "rq/rqkmeans_faiss.py": ("MovieLens1M", "--data_path"),
        "rq/rqkmeans_constrained.py": ("MovieLens1M", "--data_path"),
        "rq/rqkmeans_plus.py": ("MovieLens1M", "--data_path"),
        "sasrec.py": ("MovieLens1M", "--data_root"),
        "convert_dataset_gpr.py": ("MovieLens1M", "--data-dir"),
    }
    for relative, required in expected.items():
        source = (REPO / relative).read_text()
        assert all(value in source for value in required), relative
        compile(source, relative, "exec")
    shells = [REPO / name for name in (
        "run.sh", "setup.sh", "ts_rec_sft.sh", "rq/rqvae.sh",
        "rq/rqkmeans_constrained.sh", "rq/rqkmeans_plus.sh", "rq/generate_indices_plus.sh",
    )]
    subprocess.run(["bash", "-n", *map(str, shells)], check=True)
