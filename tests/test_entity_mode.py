"""`entity` retrieval: question -> nearest entity descriptions -> their chunks.

Reuses the synthetic two-pack dataset of test_scope.py and adds entity vectors
(embeddings of each entity's description), as a dataset declares them with
`embeddings.entity_vectors_glob`.

    uv run --python 3.12 --with pytest --with pytest-asyncio --with numpy \
        --with "mcp>=1.6,<2" pytest tests/test_entity_mode.py
"""
import json

import pytest

from ontorag_mcp.store import Dataset, LocalSource, _embed_hashed
from tests.test_scope import DIM, ENTITIES, NS, dataset_dir  # noqa: F401  (fixture)

pytestmark = pytest.mark.asyncio

DESCRIPTIONS = {
    "Magic": "the general art of working wonders",
    "Fireball": "a blazing sphere of flame hurled at enemies that burns them",
    "Gift": "the inborn talent that lets someone work Hermetic magic",
    "Wyvern": "a winged reptile beast with a venomous tail that lives in mountains",
}


@pytest.fixture
def entity_dataset(dataset_dir):  # noqa: F811
    m = json.loads((dataset_dir / "manifest.json").read_text())
    m["embeddings"]["entity_vectors_glob"] = "embeddings/entities/*.jsonl"
    (dataset_dir / "manifest.json").write_text(json.dumps(m))
    (dataset_dir / "embeddings" / "entities").mkdir()
    (dataset_dir / "embeddings" / "entities" / "0.jsonl").write_text("".join(
        json.dumps({"id": NS + name, "vector": _embed_hashed(text, DIM).tolist()}) + "\n"
        for name, text in DESCRIPTIONS.items()))
    return dataset_dir


async def _load(path, retrieval="entity"):
    return await Dataset.from_source(LocalSource(str(path)), retrieval=retrieval)


async def test_a_described_thing_finds_its_chunks(entity_dataset):
    ds = await _load(entity_dataset)
    q = _embed_hashed("which winged reptile beast has a venomous tail", DIM)
    hits = ds.search("which winged reptile beast has a venomous tail", k=3, qvec=q)
    assert hits and all(h["id"].startswith("supp::") for h in hits)   # the Wyvern's chunks


async def test_returns_k_chunks_when_available(entity_dataset):
    ds = await _load(entity_dataset)
    q = _embed_hashed("a blazing sphere of flame", DIM)
    assert len(ds.search("a blazing sphere of flame", k=6, qvec=q)) == 6


async def test_scope_applies_to_entities_and_chunks(entity_dataset):
    ds = await _load(entity_dataset)
    q = _embed_hashed("which winged reptile beast has a venomous tail", DIM)
    hits = ds.search("which winged reptile beast has a venomous tail", k=6, qvec=q, scope=["core"])
    # the Wyvern is not attested in core, so nothing of the supplement leaks in
    assert hits and all(h["id"].startswith("core::") for h in hits)


async def test_entity_mode_requires_entity_vectors(dataset_dir):  # noqa: F811
    with pytest.raises(ValueError):
        await _load(dataset_dir)


async def test_answer_names_the_matched_entities(entity_dataset):
    ds = await _load(entity_dataset)
    ds.embed = lambda text: _embed_hashed(text, DIM)
    r = ds.answer("which winged reptile beast has a venomous tail", k=2, expand=1)
    assert "Wyvern" in r["matched_entities"] and r["passages"]
