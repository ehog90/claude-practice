"""Retriever: a RAG keresőrésze önállóan, újrahasználható osztályként.

A retriever dolga: egy kérdésre visszaadni a tudásbázis legrelevánsabb darabjait (a válaszadás nem az ő dolga).
Három keresési módot tud, amiket egymás mellett össze is lehet hasonlítani:

    bm25       kulcsszavak alapján (pontos szavak, nevek, kódok)
    embedding  jelentés alapján (szinonimák, átfogalmazás)
    hybrid     a kettő RRF-fel összefésülve (reciprocal rank fusion): a pontszám a helyezések reciprokának összege

Mit tud még:
  - darabolás (add_text): hosszú szöveget bekezdések szerint, átfedéssel vág darabokra,
  - metaadat és szűrés (where=): pl. csak egy adott forrásfájl darabjai között keres,
  - embedding gyorsítótár fájlban (a dokumentumokat nem kell minden indításnál újraszámolni),
  - kiértékelés (evaluate): recall@k mérése címkézett kérdésekkel, hogy ne találgatni kelljen, melyik mód a jobb.

Használat:
    r = Retriever(embed_fn=embed_voyage, cache_path="cache.json")
    r.add_documents([{"text": "...", "metadata": {"source": "gyik"}}])
    hits = r.search("Mennyi idő alatt kapom vissza a pénzem?", mode="hybrid", k=3)

Indítás:
    python retriever.py             # interaktív: a három mód találatai egymás mellett
    python retriever.py --eval      # recall@1 és recall@3 mérése a beépített kérdéssorral
    (mindkettőhöz: --offline = kulcs nélküli, szóegyezéses embedding helyettesítő)
"""

import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# BM25 (tiszta Python)
# ---------------------------------------------------------------------------

STOPWORDS = {"hogy", "egy", "van", "meg", "mint", "vagy", "csak", "nem", "igen", "ezt", "azt", "ami", "aki", "kell", "lehet"}


def tokenize(text: str) -> list[str]:
    """Kisbetűsít, eltávolítja az ékezeteket, és az első 5 betűre vágja a szavakat (nyers 'szótövezés').

    A magyar ragozott: a 'visszaküldés' és a 'visszaküldési' a BM25 szemében két külön szó lenne; az 5 betűs
    csonkolással mindkettőből 'vissz' lesz. Durva módszer: a 'pénzem' és a 'pénzt' így sem egyezik, a rövid
    szótövek pedig összekeveredhetnek. Éles rendszerben valódi magyar szótövező való ide.
    """
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return [w[:5] for w in re.findall(r"[a-z0-9]{3,}", plain) if w not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b  # k1: a szógyakoriság telítődése, b: a hossznormalizálás ereje
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) or 1.0
        n = len(documents)
        df = Counter(word for c in self.docs for word in c)  # hány dokumentumban szerepel a szó
        self.idf = {w: math.log((n - f + 0.5) / (f + 0.5) + 1) for w, f in df.items()}  # a ritka szó többet ér

    def scores(self, query: str) -> list[float]:
        words = tokenize(query)
        result = []
        for counts, length in zip(self.docs, self.lengths):
            score = 0.0
            for w in words:
                tf = counts.get(w, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[w] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result


# ---------------------------------------------------------------------------
# Segédek: koszinusz-hasonlóság, rangsor, darabolás
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rank(scores: dict[int, float], skip_zero: bool = False) -> dict[int, int]:
    """Dokumentumindex -> helyezés (1 = legjobb). skip_zero: a nulla pontú dokumentum nem kerül a listára."""
    order = sorted(scores, key=lambda i: scores[i], reverse=True)
    return {i: place for place, i in enumerate((i for i in order if not skip_zero or scores[i] > 0), start=1)}


def chunk(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Bekezdések szerint darabol, max_chars-ig; a darabok eleje az előző darab végével átfed.

    Egy max_chars-nál hosszabb bekezdést nem vág tovább (azt mondatonkénti vágással lehetne kiegészíteni).
    """
    chunks, current = [], ""
    for paragraph in (p.strip() for p in text.split("\n\n") if p.strip()):
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = f"{current[-overlap:]}\n\n{paragraph}" if overlap else paragraph  # átfedés az előző darab végéből
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# A Retriever
# ---------------------------------------------------------------------------

MODES = ("bm25", "embedding", "hybrid")


class Retriever:
    def __init__(self, embed_fn, cache_path: str | Path | None = None, batch_size: int = 64):
        """embed_fn(texts, input_type) -> vektorok; input_type: 'document' vagy 'query'."""
        self.embed_fn = embed_fn
        self.cache_path = Path(cache_path) if cache_path else None
        self.batch_size = batch_size
        self.items: list[dict] = []  # {"text": ..., "metadata": {...}}
        self.vectors: list[list[float]] = []
        self.bm25: BM25 | None = None

    # --- indexelés ---------------------------------------------------------

    def add_documents(self, documents: list[dict]):
        """Dokumentumok hozzáadása: [{"text": ..., "metadata": {...}}]. Egyszerűség kedvéért az egész indexet újraépíti."""
        self.items.extend({"text": d["text"], "metadata": d.get("metadata", {})} for d in documents)
        texts = [item["text"] for item in self.items]
        self.vectors = self._embed_documents(texts)
        self.bm25 = BM25(texts)

    def add_text(self, text: str, max_chars: int = 1000, overlap: int = 150, **metadata):
        """Hosszú szöveg darabolása és hozzáadása; a metaadat minden darabra rákerül (+ a darab sorszáma)."""
        self.add_documents(
            [{"text": c, "metadata": {**metadata, "chunk": n}} for n, c in enumerate(chunk(text, max_chars, overlap))]
        )

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        key = hashlib.sha256(json.dumps([self.embed_fn.__name__, texts], ensure_ascii=False).encode()).hexdigest()
        if self.cache_path and self.cache_path.exists():
            cached = json.loads(self.cache_path.read_text())
            if cached.get("key") == key:  # ugyanaz a tartalom és ugyanaz az embedding: nincs API hívás
                return cached["vectors"]
        vectors = []
        for start in range(0, len(texts), self.batch_size):  # az API-k egy kérésben korlátozott számú szöveget fogadnak
            vectors += self.embed_fn(texts[start:start + self.batch_size], "document")
        if self.cache_path:
            self.cache_path.write_text(json.dumps({"key": key, "vectors": vectors}))
        return vectors

    # --- keresés -----------------------------------------------------------

    def search(self, query: str, mode: str = "hybrid", k: int = 3, where: dict | None = None,
               query_vector: list[float] | None = None, k_rrf: int = 60) -> list[dict]:
        """A k legjobb találat. where: metaadat-szűrés (pl. {"source": "gyik"}). A query_vector megadásával
        az embedding hívás kihagyható (kiértékelésnél az összes kérdést egyszerre embeddingeljük)."""
        if mode not in MODES:
            raise ValueError(f"Ismeretlen mód: {mode}. Lehetséges: {', '.join(MODES)}")
        candidates = [i for i, item in enumerate(self.items)
                      if not where or all(item["metadata"].get(key) == value for key, value in where.items())]

        bm25_scores = {i: s for i, s in zip(range(len(self.items)), self.bm25.scores(query)) if i in candidates}
        bm25_rank = rank(bm25_scores, skip_zero=True)  # ami nem tartalmaz keresett szót, nincs a rangsoron

        emb_scores, emb_rank = {}, {}
        if mode != "bm25":
            query_vector = query_vector or self.embed_fn([query], "query")[0]
            emb_scores = {i: cosine_similarity(query_vector, self.vectors[i]) for i in candidates}
            emb_rank = rank(emb_scores)

        if mode == "bm25":
            scores = bm25_scores
            ordered = sorted(bm25_rank, key=bm25_rank.get)
        elif mode == "embedding":
            scores = emb_scores
            ordered = sorted(emb_rank, key=emb_rank.get)
        else:
            # RRF: minden rangsor 1/(k + helyezés) pontot ad; a két lista pontszám-skálája nem számít, csak a sorrend
            scores = {i: sum(1 / (k_rrf + r[i]) for r in (bm25_rank, emb_rank) if i in r) for i in candidates}
            ordered = sorted(scores, key=scores.get, reverse=True)

        return [
            {
                "index": i,
                "text": self.items[i]["text"],
                "metadata": self.items[i]["metadata"],
                "score": scores[i],
                "ranks": {"bm25": bm25_rank.get(i), "embedding": emb_rank.get(i)},
            }
            for i in ordered[:k]
        ]


# ---------------------------------------------------------------------------
# Kiértékelés: recall@k
# ---------------------------------------------------------------------------


def evaluate(retriever: Retriever, cases: list[tuple[str, int]], ks=(1, 3)) -> dict[str, dict[int, float]]:
    """cases: [(kérdés, a keresett dokumentum indexe)]. Visszaadja módonként, hogy a kérdések hány részében
    van ott a helyes dokumentum a top-k-ban (recall@k)."""
    # az összes kérdést egyetlen kérésben embeddingeljük (kevesebb API hívás, rate limit barát)
    query_vectors = retriever.embed_fn([q for q, _ in cases], "query")
    result = {}
    for mode in MODES:
        result[mode] = {}
        for k in ks:
            found = sum(
                expected in [h["index"] for h in retriever.search(q, mode=mode, k=k, query_vector=vec)]
                for (q, expected), vec in zip(cases, query_vectors)
            )
            result[mode][k] = found / len(cases)
    return result


# ---------------------------------------------------------------------------
# Demó
# ---------------------------------------------------------------------------

# Címkézett kérdések: (kérdés, a helyes dokumentum indexe a DOCUMENTS listában). Szándékosan átfogalmazott kérdések.
EVAL_CASES = [
    ("Mennyi idő alatt kapom vissza a pénzem?", 0),
    ("Nem tudok belépni a fiókomba", 1),
    ("Mikor ér ide a rendelésem?", 2),
    ("Hány órakor lehet telefonálni?", 3),
    ("Biztonságos a bankkártyás fizetés?", 4),
    ("Meddig javítják ingyen, ha elromlik?", 5),
    ("Hol tölthetem le az appot?", 6),
    ("Mikor nyitnak a strandok?", 7),
]

if __name__ == "__main__":
    from embedding_search import DOCUMENTS, embed_offline, embed_voyage

    offline = "--offline" in sys.argv
    retriever = Retriever(embed_offline if offline else embed_voyage, cache_path=Path(__file__).resolve().parent / "index_cache.json")
    retriever.add_documents([{"text": d, "metadata": {"source": "gyik", "id": n}} for n, d in enumerate(DOCUMENTS)])
    print(f"Index kész: {len(DOCUMENTS)} dokumentum ({'offline' if offline else 'Voyage embedding'} + BM25)\n")

    if "--eval" in sys.argv:
        print(f"Kiértékelés {len(EVAL_CASES)} címkézett kérdéssel (recall@k: a helyes dokumentum a top-k-ban van-e):\n")
        results = evaluate(retriever, EVAL_CASES)
        print(f"   {'mód':<10} {'recall@1':>9} {'recall@3':>9}")
        for mode, recalls in results.items():
            print(f"   {mode:<10} {recalls[1]:>8.0%} {recalls[3]:>9.0%}")
    else:
        print("Kérdezz valamit; a három mód top 3 találata egymás mellett látszik. 'exit' a kilépéshez.\n")
        while True:
            query = input("🔎 Kérdés: ").strip()
            if query.lower() == "exit":
                break
            if not query:
                continue
            query_vector = retriever.embed_fn([query], "query")[0]  # egyszer embeddingelünk, mindhárom mód ezt használja
            for mode in MODES:
                print(f"   {mode}:")
                for h in retriever.search(query, mode=mode, k=3, query_vector=query_vector):
                    print(f"      {h['score']:7.3f}  {h['text'][:75]}...")
            print()
