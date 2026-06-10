"""
Dataset loading + retrieval for the OntoRAG MCP server.

A *dataset* is any repo in the OntoRAG "GitHub-as-storage" layout (root
manifest.json + ontology/ + content/chunks + embeddings/vectors). Files are read
through a **Source**:

  * MirageGitHubSource — mounts the repo via mirage-ai's GitHub resource and reads
    files as if from a local disk (no clone; GitHub data surfaced as a VFS).
  * LocalSource       — a directory on disk (local dev / a Mirage FUSE mount).

Retrieval is in-memory after load; query embeddings use the SAME provider/model
the dataset declares in embeddings/config.json.
"""
import glob as _glob
import hashlib
import json
import math
import os
import re

import numpy as np


def _norm_name(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", re.sub(r"^the\s+", "", s.lower()))).strip()


def _shq(s):
    return "'" + s.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------- #
#  Sources (async file access)
# --------------------------------------------------------------------------- #

class LocalSource:
    """Read dataset files from a local directory."""
    def __init__(self, root):
        self.root = root

    async def read_text(self, rel):
        with open(os.path.join(self.root, rel), encoding="utf-8") as f:
            return f.read()

    async def glob(self, rel_glob):
        return sorted(os.path.relpath(p, self.root)
                      for p in _glob.glob(os.path.join(self.root, rel_glob)))

    def describe(self):
        return self.root


class MirageGitHubSource:
    """Read dataset files from a GitHub repo via mirage-ai's GitHub resource,
    surfaced as a read-only virtual disk mounted at /d."""
    def __init__(self, owner, repo, ref=None, token=None):
        from mirage import MountMode, Workspace
        from mirage.resource.github import GitHubConfig, GitHubResource
        self.owner, self.repo, self.ref = owner, repo, ref or "main"
        res = GitHubResource(config=GitHubConfig(token=token or os.environ.get("GITHUB_TOKEN")),
                             owner=owner, repo=repo, ref=self.ref)
        self.ws = Workspace({"/d": res}, mode=MountMode.READ)

    async def _sh(self, cmd):
        r = await self.ws.execute(cmd)
        return await r.stdout_str()

    async def read_text(self, rel, win=1900):
        # Mirage's shell caps stdout at ~2000 lines, so page large files with
        # tail+head windows (the file is intact server-side) and reassemble.
        full = _shq("/d/" + rel)
        parts, start = [], 1
        while True:
            chunk = await self._sh("tail -n +%d %s | head -n %d" % (start, full, win))
            if not chunk:
                break
            parts.append(chunk)
            if chunk.count("\n") < win:
                break
            start += win
        return "".join(parts)

    async def glob(self, rel_glob):
        d, pat = os.path.split(rel_glob)
        out = await self._sh("find %s -name %s -type f" % (_shq("/d/" + d), _shq(pat)))
        rels = [p[len("/d/"):] for p in out.split() if p.startswith("/d/")]
        return sorted(rels)

    def describe(self):
        return "github:%s/%s@%s" % (self.owner, self.repo, self.ref)


def resolve_source(spec, ref=None, token=None):
    """Local path with a manifest -> LocalSource; '<org>/<repo>' -> MirageGitHubSource."""
    spec = spec.strip()
    if os.path.isdir(spec) and os.path.exists(os.path.join(spec, "manifest.json")):
        return LocalSource(spec)
    slug = spec.strip("/")
    if slug.count("/") != 1:
        raise ValueError("dataset must be '<org>/<repo>' or a local path, got %r" % spec)
    owner, repo = slug.split("/", 1)
    return MirageGitHubSource(owner, repo, ref=ref, token=token)


# --------------------------------------------------------------------------- #
#  Query embedding (mirrors the dataset's declared provider)
# --------------------------------------------------------------------------- #

def _l2(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _embed_ollama(text, model, url):
    import urllib.request
    payload = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/embeddings", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    return _l2(np.asarray(data["embedding"], dtype=np.float32))


def _embed_hashed(text, dim):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    grams = toks + [toks[i] + "_" + toks[i + 1] for i in range(len(toks) - 1)]
    counts = {}
    for t in grams:
        counts[t] = counts.get(t, 0) + 1
    vec = np.zeros(dim, dtype=np.float32)
    for t, c in counts.items():
        h = hashlib.md5(t.encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign * (1.0 + math.log(c))
    return _l2(vec)


# --------------------------------------------------------------------------- #
#  Embedding-free retrieval helpers (ontology + lexical)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def build_alias_matcher(entities, linkable_iris):
    """Two combined alternation regexes for detecting entity mentions in a query.
    Multi-word aliases match case-insensitively; single-word aliases exact-case."""
    ci, cs = {}, {}
    for iri in linkable_iris:
        e = entities.get(iri)
        if not e:
            continue
        for a in [e.get("label", "")] + e.get("aliases", []):
            a = (a or "").strip()
            if len(a) < 3:
                continue
            (ci if " " in a else cs).setdefault(a.lower() if " " in a else a, set()).add(iri)

    def rx(keys, flags):
        if not keys:
            return None
        body = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
        return re.compile(r"\b(?:" + body + r")\b", flags)

    return (rx(list(ci), re.IGNORECASE), ci, rx(list(cs), 0), cs)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class Dataset:
    def __init__(self, manifest, emb_cfg, entities, chunks, vid_to_vec, ollama_url, origin,
                 retrieval="vector"):
        self.manifest = manifest
        self.emb_cfg = emb_cfg or {}
        self.origin = origin
        self.ollama_url = ollama_url
        self.provider = self.emb_cfg.get("provider")
        self.model = self.emb_cfg.get("model")
        self.dim = int(self.emb_cfg.get("dim") or 0)

        self.entities = entities
        self._name_idx = {}
        for iri, e in entities.items():
            for surf in [e.get("label", "")] + e.get("aliases", []):
                self._name_idx.setdefault(_norm_name(surf), iri)

        self.chunks = chunks
        self._ent_chunks = {}
        for cid, c in chunks.items():
            for iri in c.get("entities", []):
                self._ent_chunks.setdefault(iri, []).append(cid)

        # vector index (only when embeddings are used)
        self.ids = [cid for cid in chunks if cid in vid_to_vec]
        self.has_vectors = bool(vid_to_vec)
        self.mat = (np.asarray([vid_to_vec[cid] for cid in self.ids], dtype=np.float32)
                    if self.has_vectors else None)

        self.retrieval = retrieval if retrieval != "auto" else (
            "vector" if self.has_vectors else "ontology")
        if self.retrieval in ("ontology", "hybrid"):
            self._build_ontology_index()

    def _build_ontology_index(self):
        # query-time entity matcher over entities actually linked to chunks
        self._qmatch = build_alias_matcher(self.entities, set(self._ent_chunks))
        # lexical statistics for a BM25-lite fallback (df only — light on memory)
        self._df, self._doc_len, total = {}, {}, 0
        for cid, c in self.chunks.items():
            toks = _tokenize(c.get("text", ""))
            self._doc_len[cid] = len(toks)
            total += len(toks)
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1
        self._N = max(1, len(self.chunks))
        self._avgdl = (total / self._N) or 1.0
        self._idf_ent = {e: math.log(1 + self._N / len(cs))
                         for e, cs in self._ent_chunks.items()}

    @classmethod
    async def from_source(cls, source, ollama_url="http://localhost:11434", retrieval="vector"):
        async def jsonl(rel):
            txt = await source.read_text(rel)
            return [json.loads(l) for l in txt.splitlines() if l.strip()]

        manifest = json.loads(await source.read_text("manifest.json"))

        emb_cfg = None
        if retrieval != "ontology" and "embeddings" in manifest:
            try:
                emb_cfg = json.loads(await source.read_text(manifest["embeddings"]["config"]))
            except Exception:
                emb_cfg = None

        entities = {e["iri"]: e for e in await jsonl(manifest["ontology"]["entity_index"])}

        chunks = {}
        for rel in await source.glob(manifest["content"]["chunks_glob"]):
            for c in await jsonl(rel):
                chunks[c["id"]] = c

        vid_to_vec = {}
        if retrieval != "ontology" and emb_cfg is not None:
            for rel in await source.glob(manifest["embeddings"]["vectors_glob"]):
                for r in await jsonl(rel):
                    vid_to_vec[r["id"]] = r["vector"]

        return cls(manifest, emb_cfg, entities, chunks, vid_to_vec, ollama_url,
                   source.describe(), retrieval)

    # ---- info ----
    def info(self):
        m = self.manifest
        return {
            "dataset": m["dataset"], "origin": self.origin, "retrieval": self.retrieval,
            "embedding": ({"provider": self.provider, "model": self.model, "dim": self.dim,
                           "metric": self.emb_cfg.get("metric", "cosine")}
                          if self.has_vectors else None),
            "counts": {"entities": len(self.entities), "chunks": len(self.chunks),
                       "vectors": len(self.ids), "linked_entities": len(self._ent_chunks)},
            "ontology_types": m.get("ontology", {}).get("counts", {}).get("by_type", {}),
        }

    # ---- embedding ----
    def embed(self, text):
        if self.provider == "ollama":
            return _embed_ollama(text, self.model, self.ollama_url)
        if self.provider == "hashed":
            return _embed_hashed(text, self.dim)
        raise RuntimeError("unsupported embedding provider: %s" % self.provider)

    # ---- retrieval ----
    def _entity_labels(self, iris):
        return [self.entities[i]["label"] for i in iris if i in self.entities]

    def _hit(self, cid, score):
        c = self.chunks[cid]
        return {"id": cid, "doc": c["doc"], "score": round(float(score), 4),
                "heading_path": c.get("heading_path", []),
                "entities": self._entity_labels(c.get("entities", [])), "text": c["text"]}

    def search(self, query, k=6):
        if self.retrieval == "ontology":
            return self.search_ontology(query, k=k)
        if self.retrieval == "hybrid":
            return self.search_hybrid(query, k=k)
        if not self.ids:
            return []
        q = self.embed(query)
        scores = self.mat @ q
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self._hit(self.ids[i], scores[i]) for i in top]

    # ---- embedding-free (ontology + lexical) retrieval ----
    def match_query_entities(self, query):
        rx_ci, ci, rx_cs, cs = self._qmatch
        hits = set()
        if rx_ci:
            for m in rx_ci.finditer(query):
                hits.update(ci.get(m.group(0).lower(), ()))
        if rx_cs:
            for m in rx_cs.finditer(query):
                hits.update(cs.get(m.group(0), ()))
        return hits

    def _lexical_score(self, cid, qterms, k1=1.2, b=0.75):
        from collections import Counter
        tf = Counter(_tokenize(self.chunks[cid].get("text", "")))
        L = self._doc_len.get(cid, 1) or 1
        s = 0.0
        for t in qterms:
            f = tf.get(t, 0)
            if not f:
                continue
            df = self._df.get(t, 0)
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * L / self._avgdl))
        return s

    def _rank(self, query, k):
        qe = self.match_query_entities(query)
        qterms = set(_tokenize(query))
        if qe:
            cand = set()
            for e in qe:
                cand.update(self._ent_chunks.get(e, []))
        else:
            # no entity in the query -> lexical scan over the whole corpus
            cand = set(self.chunks)
        rows = []
        for cid in cand:
            ce = self.chunks[cid].get("entities", [])
            e_score = sum(self._idf_ent.get(e, 0.0) for e in ce if e in qe)
            l_score = self._lexical_score(cid, qterms)
            rows.append([cid, e_score, l_score])
        if not rows:
            return [], qe
        emax = max(r[1] for r in rows) or 1.0
        lmax = max(r[2] for r in rows) or 1.0
        for r in rows:
            we = 0.6 if qe else 0.0
            r.append(we * (r[1] / emax) + (1 - we) * (r[2] / lmax))
        rows.sort(key=lambda r: -r[3])
        return rows[:k], qe

    def search_ontology(self, query, k=6):
        rows, _ = self._rank(query, k)
        return [self._hit(cid, score) for cid, _e, _l, score in rows]

    # ---- hybrid: sparse candidates, dense (cosine) re-rank ----
    def _candidate_cids(self, query):
        qe = self.match_query_entities(query)
        cand = set()
        for e in qe:
            cand.update(self._ent_chunks.get(e, []))
        return qe, cand

    def _hybrid_rank(self, query, k):
        qe, cand = self._candidate_cids(query)
        row = {cid: r for r, cid in enumerate(self.ids)}
        cids = [c for c in cand if c in row] or list(self.ids)  # fall back to full dense
        qv = self.embed(query)
        idx = np.fromiter((row[c] for c in cids), dtype=np.int64, count=len(cids))
        sims = self.mat[idx] @ qv
        order = np.argsort(-sims)[:k]
        return [(cids[i], float(sims[i])) for i in order], qe

    def search_hybrid(self, query, k=6):
        ranked, _qe = self._hybrid_rank(query, k)
        return [self._hit(cid, s) for cid, s in ranked]

    def answer_hybrid(self, query, k=6, expand=3):
        ranked, qe = self._hybrid_rank(query, k)
        chosen = [cid for cid, _s in ranked]
        seed = set(qe)
        for cid in chosen:
            seed.update(self.chunks[cid].get("entities", []))
        expanded = []
        if expand and seed:
            chosen_set, seen, cands = set(chosen), set(), []
            for iri in seed:
                for cid in self._ent_chunks.get(iri, []):
                    if cid in chosen_set or cid in seen:
                        continue
                    seen.add(cid)
                    cands.append((len(set(self.chunks[cid].get("entities", [])) & seed), cid))
            cands.sort(reverse=True)
            expanded = [cid for _, cid in cands[:expand]]
        used = chosen + expanded
        facts = []
        for iri in sorted({i for cid in used for i in self.chunks[cid].get("entities", [])} | set(qe)):
            e = self.entities.get(iri)
            if e:
                facts.append({"label": e["label"], "tags": e.get("tags", []),
                              "summary": e.get("summary", "")})
        passages = [{"cite": cid, "doc": self.chunks[cid]["doc"],
                     "heading_path": self.chunks[cid].get("heading_path", []),
                     "entities": self._entity_labels(self.chunks[cid].get("entities", [])),
                     "text": self.chunks[cid]["text"]} for cid in used]
        return {"query": query, "matched_entities": self._entity_labels(sorted(qe)),
                "ontology_facts": facts, "passages": passages,
                "instruction": "Answer the query using ONLY these passages; cite by [cite]. "
                               "Use ontology_facts for grounding. Say so if insufficient."}

    def answer(self, query, k=6, expand=3):
        if self.retrieval == "ontology":
            return self.answer_ontology(query, k=k, expand=expand)
        if self.retrieval == "hybrid":
            return self.answer_hybrid(query, k=k, expand=expand)
        q = self.embed(query)
        scores = self.mat @ q
        kk = min(k, len(self.ids))
        top = np.argpartition(-scores, kk - 1)[:kk]
        top = list(top[np.argsort(-scores[top])])
        chosen = [self.ids[i] for i in top]

        seed_ents = set()
        for cid in chosen:
            seed_ents.update(self.chunks[cid].get("entities", []))
        expanded = []
        if expand and seed_ents:
            chosen_set = set(chosen)
            id_to_row = {cid: r for r, cid in enumerate(self.ids)}
            seen, ranked = set(), []
            for iri in seed_ents:
                for cid in self._ent_chunks.get(iri, []):
                    if cid in chosen_set or cid in seen:
                        continue
                    seen.add(cid)
                    overlap = len(set(self.chunks[cid].get("entities", [])) & seed_ents)
                    ranked.append((overlap, float(scores[id_to_row[cid]]), cid))
            ranked.sort(reverse=True)
            expanded = [cid for _, _, cid in ranked[:expand]]

        used = chosen + expanded
        facts = []
        for iri in sorted({i for cid in used for i in self.chunks[cid].get("entities", [])}):
            e = self.entities.get(iri)
            if e:
                facts.append({"label": e["label"], "tags": e.get("tags", []),
                              "summary": e.get("summary", "")})
        passages = [{"cite": cid, "doc": self.chunks[cid]["doc"],
                     "heading_path": self.chunks[cid].get("heading_path", []),
                     "entities": self._entity_labels(self.chunks[cid].get("entities", [])),
                     "text": self.chunks[cid]["text"]} for cid in used]
        return {"query": query, "ontology_facts": facts, "passages": passages,
                "instruction": "Answer the query using ONLY these passages; cite by [cite]. "
                               "Use ontology_facts for grounding. Say so if insufficient."}

    def answer_ontology(self, query, k=6, expand=3):
        rows, qe = self._rank(query, k)
        chosen = [cid for cid, _e, _l, _s in rows]
        seed_ents = set(qe)
        for cid in chosen:
            seed_ents.update(self.chunks[cid].get("entities", []))
        expanded = []
        if expand and seed_ents:
            chosen_set, seen, ranked = set(chosen), set(), []
            for iri in seed_ents:
                for cid in self._ent_chunks.get(iri, []):
                    if cid in chosen_set or cid in seen:
                        continue
                    seen.add(cid)
                    overlap = len(set(self.chunks[cid].get("entities", [])) & seed_ents)
                    ranked.append((overlap, cid))
            ranked.sort(reverse=True)
            expanded = [cid for _, cid in ranked[:expand]]
        used = chosen + expanded
        facts = []
        for iri in sorted({i for cid in used for i in self.chunks[cid].get("entities", [])} | set(qe)):
            e = self.entities.get(iri)
            if e:
                facts.append({"label": e["label"], "tags": e.get("tags", []),
                              "summary": e.get("summary", "")})
        passages = [{"cite": cid, "doc": self.chunks[cid]["doc"],
                     "heading_path": self.chunks[cid].get("heading_path", []),
                     "entities": self._entity_labels(self.chunks[cid].get("entities", [])),
                     "text": self.chunks[cid]["text"]} for cid in used]
        return {"query": query, "matched_entities": self._entity_labels(sorted(qe)),
                "ontology_facts": facts, "passages": passages,
                "instruction": "Answer the query using ONLY these passages; cite by [cite]. "
                               "Use ontology_facts for grounding. Say so if insufficient."}

    # ---- ontology ----
    def _resolve_entity(self, key):
        if key in self.entities:
            return self.entities[key]
        iri = self._name_idx.get(_norm_name(key))
        return self.entities.get(iri) if iri else None

    def get_entity(self, key):
        e = self._resolve_entity(key)
        if not e:
            return None
        out = dict(e)
        out["linked_chunks"] = len(self._ent_chunks.get(e["iri"], []))
        return out

    def search_entities(self, query, limit=20):
        ql = query.lower()
        hits = []
        for e in self.entities.values():
            hay = " ".join([e.get("label", "")] + e.get("aliases", []) + [e.get("summary", "")]).lower()
            if ql in hay:
                hits.append((len(self._ent_chunks.get(e["iri"], [])), e))
        hits.sort(key=lambda x: -x[0])
        return [{"iri": e["iri"], "label": e["label"], "types": e.get("types", []),
                 "tags": e.get("tags", []), "summary": e.get("summary", ""),
                 "linked_chunks": n} for n, e in hits[:limit]]

    def entity_chunks(self, key, k=8):
        e = self._resolve_entity(key)
        if not e:
            return []
        return [self._hit(cid, 1.0) for cid in self._ent_chunks.get(e["iri"], [])[:k]]
