"""Pack scope semantics (https://ontorag.org/provenance/#packs), for the in-memory
store in every retrieval mode and for the Redis store.

    uv run --python 3.12 --with pytest --with pytest-asyncio --with fakeredis --with numpy \
        --with redis --with "mcp>=1.6,<2" pytest tests/test_scope.py
"""
import json

import pytest

from ontorag_mcp.store import (Dataset, LocalSource, _embed_hashed, chunk_in_scope,
                               entity_in_scope, resolve_scope)

NS = "https://example.org/ds/"
DIM = 64


def _ent(name, attested, defined):
    return {"iri": NS + name, "types": [NS + "Thing"], "label": name, "aliases": [],
            "summary": "about " + name, "attestedIn": attested, "definedIn": defined}


ENTITIES = [
    _ent("Magic", [], []),                                  # spine
    _ent("Fireball", ["core"], ["core"]),                   # core only
    _ent("Gift", ["core", "supp"], ["core"]),               # defined in core, also in supp
    _ent("Wyvern", ["supp"], ["supp"]),                     # supplement only
]


def _chunks():
    out = []
    for i in range(5):
        out.append({"id": "core::%04d" % i, "doc": "core", "seq": i, "heading_path": ["Core"],
                    "text": "Fireball and the Gift are Magic spells in the core rules, part %d." % i,
                    "entities": [NS + "Fireball", NS + "Gift", NS + "Magic"]})
    for i in range(5):
        out.append({"id": "supp::%04d" % i, "doc": "supp", "seq": i, "heading_path": ["Supp"],
                    "text": "The Wyvern resists the Gift and Magic in the supplement, part %d." % i,
                    "entities": [NS + "Wyvern", NS + "Gift", NS + "Magic"]})
    return out


@pytest.fixture
def dataset_dir(tmp_path):
    (tmp_path / "ontology").mkdir()
    (tmp_path / "content" / "chunks").mkdir(parents=True)
    (tmp_path / "embeddings" / "vectors").mkdir(parents=True)
    manifest = {
        "ontorag": "0.1",
        "dataset": {"id": "t", "name": "test", "version": "1"},
        "ontology": {"graph": "ontology/world.ttl", "entity_index": "ontology/entities.jsonl",
                     "base_iri": NS},
        "content": {"chunks_glob": "content/chunks/*.jsonl"},
        "embeddings": {"config": "embeddings/config.json",
                       "vectors_glob": "embeddings/vectors/*.jsonl", "dim": DIM, "metric": "cosine"},
        "composition": {"model": "packs", "registry": "content/books.json"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "content" / "books.json").write_text(json.dumps({
        "core": {"title": "Core", "requires": []},
        "supp": {"title": "Supplement", "requires": ["core"]}}))
    (tmp_path / "ontology" / "entities.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in ENTITIES))
    (tmp_path / "embeddings" / "config.json").write_text(
        json.dumps({"provider": "hashed", "model": "hashed", "dim": DIM}))
    for pack in ("core", "supp"):
        chunks = [c for c in _chunks() if c["doc"] == pack]
        (tmp_path / "content" / "chunks" / (pack + ".jsonl")).write_text(
            "".join(json.dumps(c) + "\n" for c in chunks))
        (tmp_path / "embeddings" / "vectors" / (pack + ".jsonl")).write_text("".join(
            json.dumps({"id": c["id"], "vector": _embed_hashed(c["text"], DIM).tolist()}) + "\n"
            for c in chunks))
    return tmp_path


# --------------------------------------------------------------------------- #
#  pure semantics
# --------------------------------------------------------------------------- #

REGISTRY = {"core": {"requires": []}, "supp": {"requires": ["core"]}}


def test_resolve_scope_access_mode_never_expands():
    assert resolve_scope(None, REGISTRY) is None
    assert resolve_scope(["supp"], REGISTRY) == {"supp"}
    assert resolve_scope(["supp"], REGISTRY, close_over_requires=True) == {"supp", "core"}
    assert resolve_scope([], REGISTRY) == frozenset()


def test_entity_and_chunk_membership():
    ents = {e["label"]: e for e in ENTITIES}
    supp = frozenset({"supp"})
    assert entity_in_scope(ents["Magic"], supp)          # spine is always visible
    assert entity_in_scope(ents["Gift"], supp)           # attested in supp
    assert entity_in_scope(ents["Wyvern"], supp)
    assert not entity_in_scope(ents["Fireball"], supp)   # core only
    assert not entity_in_scope(ents["Fireball"], frozenset())
    assert chunk_in_scope({"doc": "supp"}, supp) and not chunk_in_scope({"doc": "core"}, supp)


# --------------------------------------------------------------------------- #
#  in-memory store, every retrieval mode
# --------------------------------------------------------------------------- #

async def _load(dataset_dir, mode):
    return await Dataset.from_source(LocalSource(str(dataset_dir)), retrieval=mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["vector", "ontology", "hybrid", "auto"])
async def test_search_respects_scope_and_keeps_k(dataset_dir, mode):
    ds = await _load(dataset_dir, mode)
    # the query favours core chunks; a supplement-only consumer must still get k supp hits
    q = "Fireball Gift core rules"
    unscoped = ds.search(q, k=3)
    assert len(unscoped) == 3
    hits = ds.search(q, k=3, scope=["supp"])
    assert len(hits) == 3 and {h["doc"] for h in hits} == {"supp"}
    closed = ds.search(q, k=10, scope=["supp"], close_over_requires=True)
    assert {h["doc"] for h in closed} == {"core", "supp"}
    assert ds.search(q, k=3, scope=[]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["vector", "ontology", "hybrid"])
async def test_answer_hides_out_of_scope_facts_and_passages(dataset_dir, mode):
    ds = await _load(dataset_dir, mode)
    out = ds.answer("Fireball and the Gift", k=3, expand=5, scope=["supp"])
    assert out["passages"] and all(p["doc"] == "supp" for p in out["passages"])
    labels = {f["label"] for f in out["ontology_facts"]}
    assert "Fireball" not in labels and "Gift" in labels
    assert "Fireball" not in out.get("matched_entities", [])


@pytest.mark.asyncio
async def test_entity_tools_respect_scope(dataset_dir):
    ds = await _load(dataset_dir, "ontology")
    assert ds.get_entity("Fireball", scope=["supp"]) is None
    assert ds.get_entity("Fireball", scope=["supp"], close_over_requires=True)["label"] == "Fireball"
    assert ds.get_entity("Magic", scope=["supp"])["label"] == "Magic"
    gift = ds.get_entity("Gift", scope=["supp"])
    assert gift["linked_chunks"] == 5                     # only the supplement's chunks
    assert ds.get_entity("Gift")["linked_chunks"] == 10
    chunks = ds.entity_chunks("Gift", k=8, scope=["supp"])
    assert len(chunks) == 5 and {c["doc"] for c in chunks} == {"supp"}
    assert ds.entity_chunks("Fireball", scope=["supp"]) == []
    found = {e["label"] for e in ds.search_entities("", limit=10, scope=["supp"])}
    assert found == {"Magic", "Gift", "Wyvern"}


# --------------------------------------------------------------------------- #
#  Redis store
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ontology", "hybrid"])
async def test_redis_scope(dataset_dir, fake_redis, mode):
    from ontorag_mcp.redis_store import RedisDataset
    ds = RedisDataset(fake_redis, "t", LocalSource(str(dataset_dir)), mode=mode)
    q = "Fireball Gift core rules"
    hits = await ds.search(q, k=3, scope=["supp"])
    assert len(hits) == 3 and {h["doc"] for h in hits} == {"supp"}
    closed = await ds.search(q, k=10, scope=["supp"], close_over_requires=True)
    assert "core" in {h["doc"] for h in closed}
    out = await ds.answer("Fireball and the Gift", k=3, expand=5, scope=["supp"])
    assert all(p["doc"] == "supp" for p in out["passages"])
    assert "Fireball" not in {f["label"] for f in out["ontology_facts"]}
    assert await ds.get_entity("Fireball", scope=["supp"]) is None
    assert (await ds.get_entity("Gift", scope=["supp"]))["linked_chunks"] == 5
    chunks = await ds.entity_chunks("Gift", k=8, scope=["supp"])
    assert len(chunks) == 5 and {c["doc"] for c in chunks} == {"supp"}
    ents = await ds.search_entities("Fireball Gift Wyvern", scope=["supp"])
    assert {e["label"] for e in ents} == {"Gift", "Wyvern"}
    assert {e["label"]: e["linked_chunks"] for e in ents}["Gift"] == 5
