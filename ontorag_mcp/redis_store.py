"""
Redis-backed, embedding-free OntoRAG retrieval for serverless / stateless deploys.

Instead of loading a dataset into process memory, the ontology + lexical index is
**materialized into Redis on a cache miss with a 24 h TTL**. Each request probes
only the keys it needs (query n-gram aliases -> matched entities -> their chunk
ids -> just those chunks), so no per-instance load and the index is shared across
instances. When the keys expire (24 h), the next query repopulates from source —
which also refreshes the dataset.

Key space (prefix `ontorag:<spec>:`):
  meta                 JSON dataset summary
  ready:v2             marker; expires ~10 min before data keys (bumped when the
                       key layout changes, so older caches repopulate)
  registry             JSON pack registry (pack -> {requires, ...})
  cdoc     (hash)      chunk_id -> pack (the chunk's doc), for scope filtering
  ent:<iri>            JSON entity record
  alias:<norm>         JSON [iri,...]            (query entity matching)
  echunks:<iri>        JSON [chunk_id,...]       (entity -> chunks)
  edf      (hash)      iri -> document frequency (for entity idf)
  chunk:<id>           JSON {text,doc,heading_path,entities}
  df       (hash)      token -> document frequency
  doclen   (hash)      chunk_id -> token count
  tok:<token>          JSON [[chunk_id,tf],...]  (capped lexical postings)
  N, avgdl             corpus stats
"""
import asyncio
import base64
import json
import math
import re
from collections import Counter

import numpy as np
import redis.asyncio as aioredis

from .store import (Dataset, _embed_hashed, _embed_ollama, _norm_name, _tokenize,
                    entity_in_scope, resolve_scope)

TTL = 24 * 3600
READY_EARLY = 600          # ready marker expires this many seconds before data
NGRAM = 4                  # longest entity-name phrase to probe
POSTINGS_CAP = 400         # max chunks stored per lexical token
CAND_CAP = 800             # max candidate chunks scored per query
READY = "ready:v2"
_pop_lock = asyncio.Lock()


def client(url):
    return aioredis.from_url(url, decode_responses=True)


# --------------------------------------------------------------------------- #
#  Population (cache miss): load via source, write the index with TTL
# --------------------------------------------------------------------------- #

async def populate_redis(source, r, spec, ttl=TTL, mode="ontology", ollama_url="http://localhost:11434"):
    ds = await Dataset.from_source(source, ollama_url=ollama_url,
                                   retrieval=("hybrid" if mode == "hybrid" else "ontology"))
    P = "ontorag:%s:" % spec

    pipe = r.pipeline(transaction=False)
    n = 0

    async def flush(force=False):
        nonlocal pipe, n
        if n and (force or n >= 4000):
            await pipe.execute()
            pipe = r.pipeline(transaction=False)
            n = 0

    def setk(key, val):
        nonlocal n
        pipe.set(P + key, val, ex=ttl)
        n += 1

    setk("meta", json.dumps(ds.info()))
    setk("registry", json.dumps(ds.registry))
    setk("N", str(ds._N))
    setk("avgdl", str(ds._avgdl))

    # entities
    for iri, e in ds.entities.items():
        setk("ent:" + iri, json.dumps(e))
        await flush()

    # alias map + entity->chunks (linked entities only) + entity df hash
    alias_map = {}
    edf = {}
    for iri, cids in ds._ent_chunks.items():
        edf[iri] = len(cids)
        setk("echunks:" + iri, json.dumps(cids))
        e = ds.entities.get(iri)
        if e:
            for a in [e.get("label", "")] + e.get("aliases", []):
                key = _norm_name(a)
                if len(key) >= 3:
                    alias_map.setdefault(key, set()).add(iri)
        await flush()
    if edf:
        pipe.hset(P + "edf", mapping={k: str(v) for k, v in edf.items()})
        pipe.expire(P + "edf", ttl)
        n += 2
    for key, iris in alias_map.items():
        setk("alias:" + key, json.dumps(sorted(iris)))
        await flush()

    # chunks + doclen + lexical postings
    doclen = {}
    postings = {}
    for cid, c in ds.chunks.items():
        setk("chunk:" + cid, json.dumps({"text": c["text"], "doc": c["doc"],
                                         "heading_path": c.get("heading_path", []),
                                         "entities": c.get("entities", [])}))
        toks = _tokenize(c["text"])
        doclen[cid] = len(toks)
        for t, f in Counter(toks).items():
            postings.setdefault(t, []).append((cid, f))
        await flush()

    cdoc = {cid: c["doc"] for cid, c in ds.chunks.items()}
    if cdoc:
        items = list(cdoc.items())
        for i in range(0, len(items), 4000):
            pipe.hset(P + "cdoc", mapping=dict(items[i:i + 4000]))
            n += 1
            await flush()
        pipe.expire(P + "cdoc", ttl); n += 1

    if doclen:
        # doclen hash can be large; write in field-batches
        items = list(doclen.items())
        for i in range(0, len(items), 4000):
            pipe.hset(P + "doclen", mapping={k: str(v) for k, v in items[i:i + 4000]})
            n += 1
            await flush()
        pipe.expire(P + "doclen", ttl); n += 1

    df = {t: len(lst) for t, lst in postings.items()}
    if df:
        items = list(df.items())
        for i in range(0, len(items), 4000):
            pipe.hset(P + "df", mapping={k: str(v) for k, v in items[i:i + 4000]})
            n += 1
            await flush()
        pipe.expire(P + "df", ttl); n += 1

    half = ds._N * 0.5
    for t, lst in postings.items():
        if df[t] > half:           # skip near-ubiquitous tokens (cheap stopword filter)
            continue
        lst.sort(key=lambda x: -x[1])
        setk("tok:" + t, json.dumps(lst[:POSTINGS_CAP]))
        await flush()

    # hybrid mode: also store per-chunk vectors (base64 float32) + embedding config
    if mode == "hybrid" and ds.has_vectors:
        setk("emb", json.dumps({"provider": ds.provider, "model": ds.model, "dim": ds.dim}))
        mat = ds.mat.astype(np.float32)
        for i, cid in enumerate(ds.ids):
            setk("vec:" + cid, base64.b64encode(mat[i].tobytes()).decode("ascii"))
            await flush()

    await flush(force=True)
    # ready marker expires slightly earlier so refresh happens before data keys drop
    await r.set(P + READY, "1", ex=max(60, ttl - READY_EARLY))
    return ds.info()


# --------------------------------------------------------------------------- #
#  Query (per request): probe only what's needed
# --------------------------------------------------------------------------- #

class RedisDataset:
    def __init__(self, r, spec, source, ttl=TTL, mode="ontology", ollama_url="http://localhost:11434"):
        self.r = r
        self.spec = spec
        self.source = source
        self.ttl = ttl
        self.mode = mode               # "ontology" (sparse) or "hybrid" (sparse->dense re-rank)
        self.ollama_url = ollama_url
        self.P = "ontorag:%s:" % spec
        self._emb = None

    async def ensure(self):
        if await self.r.exists(self.P + READY):
            return
        async with _pop_lock:
            if not await self.r.exists(self.P + READY):
                await populate_redis(self.source, self.r, self.spec, self.ttl,
                                     mode=self.mode, ollama_url=self.ollama_url)

    async def _embcfg(self):
        if self._emb is None:
            raw = await self.r.get(self.P + "emb")
            self._emb = json.loads(raw) if raw else {}
        return self._emb

    def _embed(self, cfg, query):
        if cfg.get("provider") == "ollama":
            return _embed_ollama(query, cfg["model"], self.ollama_url)
        if cfg.get("provider") == "hashed":
            return _embed_hashed(query, int(cfg["dim"]))
        raise RuntimeError("hybrid mode needs an embedding config in Redis ('emb' key)")

    async def info(self):
        await self.ensure()
        meta = await self.r.get(self.P + "meta")
        return json.loads(meta) if meta else {}

    # ---- scope (see store.resolve_scope) ----
    async def scope(self, scope=None, close_over_requires=False):
        if scope is None:
            return None
        registry = {}
        if close_over_requires:
            raw = await self.r.get(self.P + "registry")
            registry = json.loads(raw) if raw else {}
        return resolve_scope(scope, registry, close_over_requires)

    async def _in_scope_cids(self, cids, packs):
        """Keep only chunk ids whose pack is in scope (before any capping)."""
        cids = list(cids)
        if packs is None or not cids:
            return cids
        docs = await self.r.hmget(self.P + "cdoc", cids)
        return [c for c, d in zip(cids, docs) if d in packs]

    async def _in_scope_iris(self, iris, packs):
        if packs is None or not iris:
            return set(iris)
        ents = await self._mget("ent:", list(iris))
        return {i for i, e in ents.items() if entity_in_scope(e, packs)}

    async def _mget(self, prefix, ids):
        if not ids:
            return {}
        vals = await self.r.mget([self.P + prefix + i for i in ids])
        return {i: json.loads(v) for i, v in zip(ids, vals) if v}

    async def _match_entities(self, query):
        words = re.findall(r"[A-Za-z0-9']+", query)
        grams = set()
        for nlen in range(1, NGRAM + 1):
            for i in range(len(words) - nlen + 1):
                g = _norm_name(" ".join(words[i:i + nlen]))
                if len(g) >= 3:
                    grams.add(g)
        amap = await self._mget("alias:", list(grams))
        iris = set()
        for v in amap.values():
            iris.update(v)
        return iris

    async def _entity_labels(self, iris):
        ents = await self._mget("ent:", list(iris))
        return [ents[i]["label"] for i in iris if i in ents]

    async def _rank(self, query, k, packs=None):
        await self.ensure()
        qe = await self._in_scope_iris(await self._match_entities(query), packs)
        qterms = set(_tokenize(query))
        cand = set()
        if qe:
            for cids in (await self._mget("echunks:", list(qe))).values():
                cand.update(cids)
        else:
            for lst in (await self._mget("tok:", list(qterms))).values():
                cand.update(cid for cid, _tf in lst)
        cand = (await self._in_scope_cids(cand, packs))[:CAND_CAP]
        if not cand:
            return [], qe, {}

        chunks = await self._mget("chunk:", cand)
        N = int(await self.r.get(self.P + "N") or 1)
        avgdl = float(await self.r.get(self.P + "avgdl") or 1.0)
        edf = {iri: int(v) for iri, v in zip(qe, await self.r.hmget(self.P + "edf", list(qe))) if v} if qe else {}
        dft = {t: int(v) for t, v in zip(qterms, await self.r.hmget(self.P + "df", list(qterms))) if v}
        dl = {cid: int(v) for cid, v in zip(cand, await self.r.hmget(self.P + "doclen", cand)) if v}

        rows = []
        for cid, c in chunks.items():
            e_score = sum(math.log(1 + N / edf[e]) for e in c.get("entities", []) if e in edf)
            tf = Counter(_tokenize(c["text"]))
            L = dl.get(cid, 1) or 1
            l_score = 0.0
            for t in qterms:
                f = tf.get(t, 0)
                if not f:
                    continue
                df = dft.get(t, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                l_score += idf * (f * 2.2) / (f + 1.2 * (1 - 0.75 + 0.75 * L / avgdl))
            rows.append([cid, e_score, l_score])
        emax = max((r[1] for r in rows), default=0) or 1.0
        lmax = max((r[2] for r in rows), default=0) or 1.0
        we = 0.6 if qe else 0.0
        for r in rows:
            r.append(we * (r[1] / emax) + (1 - we) * (r[2] / lmax))
        rows.sort(key=lambda r: -r[3])
        return rows[:k], qe, chunks

    def _hit(self, cid, score, chunks, labels):
        c = chunks[cid]
        return {"id": cid, "doc": c["doc"], "score": round(float(score), 4),
                "heading_path": c.get("heading_path", []),
                "entities": labels, "text": c["text"]}

    async def search(self, query, k=6, scope=None, close_over_requires=False):
        packs = await self.scope(scope, close_over_requires)
        if self.mode == "hybrid":
            return await self.search_hybrid(query, k=k, packs=packs)
        rows, _qe, chunks = await self._rank(query, k, packs)
        labelmap = await self._labels_for(chunks, [r[0] for r in rows])
        return [self._hit(cid, s, chunks, labelmap[cid]) for cid, _e, _l, s in rows]

    # ---- hybrid: sparse candidates -> fetch only their vectors -> cosine re-rank ----
    async def _hybrid_rank(self, query, k, packs=None):
        await self.ensure()
        qe = await self._in_scope_iris(await self._match_entities(query), packs)
        qterms = set(_tokenize(query))
        cand = set()
        if qe:
            for cids in (await self._mget("echunks:", list(qe))).values():
                cand.update(cids)
        else:
            for lst in (await self._mget("tok:", list(qterms))).values():
                cand.update(cid for cid, _tf in lst)
        cand = (await self._in_scope_cids(cand, packs))[:CAND_CAP]
        if not cand:
            return [], qe
        vraw = await self.r.mget([self.P + "vec:" + c for c in cand])
        vecs, cids = [], []
        for c, b in zip(cand, vraw):
            if b:
                vecs.append(np.frombuffer(base64.b64decode(b), dtype=np.float32))
                cids.append(c)
        if not cids:
            return [], qe
        qv = self._embed(await self._embcfg(), query)
        sims = np.vstack(vecs) @ qv
        order = list(np.argsort(-sims)[:k])
        return [(cids[i], float(sims[i])) for i in order], qe

    async def search_hybrid(self, query, k=6, packs=None):
        ranked, _qe = await self._hybrid_rank(query, k, packs)
        cids = [c for c, _ in ranked]
        chunks = await self._mget("chunk:", cids)
        labelmap = await self._labels_for(chunks, [c for c in cids if c in chunks])
        return [self._hit(cid, s, chunks, labelmap[cid]) for cid, s in ranked if cid in chunks]

    async def answer_hybrid(self, query, k=6, expand=3, packs=None):
        ranked, qe = await self._hybrid_rank(query, k, packs)
        chosen = [c for c, _ in ranked]
        chunks = await self._mget("chunk:", chosen)
        seed = set(qe)
        for cid in chosen:
            seed.update(chunks.get(cid, {}).get("entities", []))
        expanded = []
        if expand and seed:
            extra = await self._mget("echunks:", list(seed))
            chosen_set, seen, pool = set(chosen), set(), []
            for cids in extra.values():
                for cid in cids:
                    if cid in chosen_set or cid in seen:
                        continue
                    seen.add(cid)
                    pool.append(cid)
            more = await self._mget("chunk:", (await self._in_scope_cids(pool, packs))[:CAND_CAP])
            scored = sorted(((len(set(more[cid].get("entities", [])) & seed), cid) for cid in more),
                            reverse=True)
            expanded = [cid for _, cid in scored[:expand]]
            chunks.update(more)
        used = [c for c in chosen + expanded if c in chunks]
        labelmap = await self._labels_for(chunks, used)
        all_iris = sorted({i for cid in used for i in chunks[cid].get("entities", [])} | set(qe))
        ents = await self._mget("ent:", all_iris)
        facts = [{"label": ents[i]["label"], "tags": ents[i].get("tags", []),
                  "summary": ents[i].get("summary", "")} for i in all_iris
                 if i in ents and entity_in_scope(ents[i], packs)]
        passages = [{"cite": cid, "doc": chunks[cid]["doc"],
                     "heading_path": chunks[cid].get("heading_path", []),
                     "entities": labelmap.get(cid, []), "text": chunks[cid]["text"]} for cid in used]
        return {"query": query, "matched_entities": await self._entity_labels(qe),
                "ontology_facts": facts, "passages": passages,
                "instruction": "Answer the query using ONLY these passages; cite by [cite]. "
                               "Use ontology_facts for grounding. Say so if insufficient."}

    async def _labels_for(self, chunks, cids):
        iris = set()
        for cid in cids:
            iris.update(chunks[cid].get("entities", []))
        ents = await self._mget("ent:", list(iris))
        return {cid: [ents[i]["label"] for i in chunks[cid].get("entities", []) if i in ents]
                for cid in cids}

    async def answer(self, query, k=6, expand=3, scope=None, close_over_requires=False):
        packs = await self.scope(scope, close_over_requires)
        if self.mode == "hybrid":
            return await self.answer_hybrid(query, k=k, expand=expand, packs=packs)
        rows, qe, chunks = await self._rank(query, k, packs)
        chosen = [r[0] for r in rows]
        seed = set(qe)
        for cid in chosen:
            seed.update(chunks[cid].get("entities", []))
        expanded = []
        if expand and seed:
            extra = await self._mget("echunks:", list(seed))
            chosen_set, seen, ranked = set(chosen), set(), []
            for cids in extra.values():
                for cid in cids:
                    if cid in chosen_set or cid in seen:
                        continue
                    seen.add(cid)
                    ranked.append(cid)
            more = await self._mget("chunk:", (await self._in_scope_cids(ranked, packs))[:CAND_CAP])
            scored = sorted(((len(set(more[cid].get("entities", [])) & seed), cid)
                             for cid in more), reverse=True)
            expanded = [cid for _, cid in scored[:expand]]
            chunks.update(more)
        used = chosen + expanded
        labelmap = await self._labels_for(chunks, used)
        all_iris = sorted({i for cid in used for i in chunks[cid].get("entities", [])} | set(qe))
        ents = await self._mget("ent:", all_iris)
        facts = [{"label": ents[i]["label"], "tags": ents[i].get("tags", []),
                  "summary": ents[i].get("summary", "")} for i in all_iris
                 if i in ents and entity_in_scope(ents[i], packs)]
        passages = [{"cite": cid, "doc": chunks[cid]["doc"],
                     "heading_path": chunks[cid].get("heading_path", []),
                     "entities": labelmap[cid], "text": chunks[cid]["text"]} for cid in used]
        return {"query": query, "matched_entities": await self._entity_labels(qe),
                "ontology_facts": facts, "passages": passages,
                "instruction": "Answer the query using ONLY these passages; cite by [cite]. "
                               "Use ontology_facts for grounding. Say so if insufficient."}

    async def get_entity(self, key, scope=None, close_over_requires=False):
        packs = await self.scope(scope, close_over_requires)
        await self.ensure()
        iri = key
        if not (await self.r.exists(self.P + "ent:" + key)):
            amap = await self._mget("alias:", [_norm_name(key)])
            iri = (amap.get(_norm_name(key)) or [None])[0]
        if not iri:
            return None
        raw = await self.r.get(self.P + "ent:" + iri)
        if not raw:
            return None
        e = json.loads(raw)
        if not entity_in_scope(e, packs):
            return None
        if packs is None:
            df = await self.r.hmget(self.P + "edf", [iri])
            e["linked_chunks"] = int(df[0]) if df and df[0] else 0
        else:
            cidsraw = await self.r.get(self.P + "echunks:" + iri)
            e["linked_chunks"] = len(await self._in_scope_cids(json.loads(cidsraw) if cidsraw else [], packs))
        return e

    async def search_entities(self, query, limit=20, scope=None, close_over_requires=False):
        packs = await self.scope(scope, close_over_requires)
        await self.ensure()
        iris = await self._in_scope_iris(await self._match_entities(query), packs)
        ents = await self._mget("ent:", list(iris))
        if packs is None:
            dfs = await self.r.hmget(self.P + "edf", list(iris)) if iris else []
            dfmap = {i: int(v) for i, v in zip(iris, dfs) if v}
        else:
            echunks = await self._mget("echunks:", list(iris))
            dfmap = {i: len(await self._in_scope_cids(c, packs)) for i, c in echunks.items()}
        out = [{"iri": e["iri"], "label": e["label"], "types": e.get("types", []),
                "tags": e.get("tags", []), "summary": e.get("summary", ""),
                "linked_chunks": dfmap.get(e["iri"], 0)} for e in ents.values()]
        out.sort(key=lambda x: -x["linked_chunks"])
        return out[:limit]

    async def entity_chunks(self, key, k=8, scope=None, close_over_requires=False):
        e = await self.get_entity(key, scope=scope, close_over_requires=close_over_requires)
        if not e:
            return []
        packs = await self.scope(scope, close_over_requires)
        cidsraw = await self.r.get(self.P + "echunks:" + e["iri"])
        cids = (await self._in_scope_cids(json.loads(cidsraw) if cidsraw else [], packs))[:k]
        chunks = await self._mget("chunk:", cids)
        labelmap = await self._labels_for(chunks, [c for c in cids if c in chunks])
        return [self._hit(cid, 1.0, chunks, labelmap[cid]) for cid in cids if cid in chunks]
