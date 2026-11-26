"""Szemantikus keresés text embeddinggel: a jelentés alapján találja meg a hasonló szövegeket.

Menete:
  1. Minden dokumentumhoz kiszámoljuk az embeddinget (számlista), és eltároljuk.
  2. A kérdéshez is kiszámoljuk az embeddinget.
  3. A kérdés vektorát összehasonlítjuk a dokumentumokéval (koszinusz-hasonlóság), és a legközelebbit adjuk.

Az Anthropic nem ad saját embedding modellt, ezért a Voyage AI-t használjuk (VOYAGE_API_KEY kell a .env-be
vagy a környezetbe). Külön csomag nem kell: sima HTTP hívás és tiszta Python.

Indítás:
    python embedding_search.py             # Voyage AI embedding (szemantikus)
    python embedding_search.py --offline   # kulcs nélkül: csak a működést mutatja, szóegyezésre keres (NEM szemantikus)
"""

import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = os.environ.get("VOYAGE_MODEL", "voyage-3.5")

DOCUMENTS = [
    "A visszaküldési határidő 30 nap, a pénzt 5 munkanapon belül visszautaljuk.",
    "Hogyan állíthatom vissza az elfelejtett jelszavamat? Kattints a bejelentkezésnél az 'Elfelejtett jelszó' linkre.",
    "A csomagot általában 2-3 munkanapon belül szállítjuk, a szállítás 15 000 forint felett ingyenes.",
    "Az ügyfélszolgálat hétfőtől péntekig 8 és 18 óra között érhető el telefonon és e-mailben.",
    "A kártyás fizetés biztonságos, a bankkártya adatait nem tároljuk a szervereinken.",
    "A garancia a vásárlástól számított 2 évig érvényes, a számla bemutatásával.",
    "Az alkalmazás letölthető az App Store-ból és a Google Play-ről, mindkettő ingyenes.",
    "A Balaton déli partján nyáron kellemes a víz, a strandok 9 és 19 óra között tartanak nyitva.",
]


# ---------------------------------------------------------------------------
# Embedding számolás
# ---------------------------------------------------------------------------


def embed_voyage(texts: list[str], input_type: str) -> list[list[float]]:
    """Voyage AI embedding. Az input_type 'document' vagy 'query': a modell másképp optimalizálja a kettőt."""
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        sys.exit("Nincs VOYAGE_API_KEY. Add meg a .env-ben (voyageai.com), vagy indítsd --offline móddal.")
    body = json.dumps({"input": texts, "model": VOYAGE_MODEL, "input_type": input_type}).encode()
    request = urllib.request.Request(
        VOYAGE_URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                data = json.load(resp)["data"]
            return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:  # rate limit (pl. fizetési mód nélküli fiókon kb. 3 kérés/perc): várunk és újrapróbáljuk
                wait = int(e.headers.get("retry-after") or 21)
                print(f"   ⏳ Voyage rate limit, várok {wait} mp-et ({attempt}/3 újrapróbálás)...", flush=True)
                time.sleep(wait)
                continue
            sys.exit(f"Voyage hiba (HTTP {e.code}): {e.read().decode(errors='replace')[:300]}")


def embed_offline(texts: list[str], input_type: str, dims: int = 256) -> list[list[float]]:
    """Kulcs nélküli helyettesítő: szavak hash-elése egy rögzített méretű vektorba (bag-of-words).

    Ez NEM szemantikus: csak a közös szavakat veszi észre, a szinonimákat nem. A mechanikát mutatja be.
    """
    vectors = []
    for text in texts:
        plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()  # ékezetek nélkül
        vector = [0.0] * dims
        for word in re.findall(r"[a-z0-9]{3,}", plain):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
        vectors.append(vector)
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """A két vektor által bezárt szög koszinusza: 1 = azonos irány (nagyon hasonló), 0 = nincs kapcsolat."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def search(query_vector: list[float], doc_vectors: list[list[float]], top_k: int = 3) -> list[tuple[float, int]]:
    scored = sorted(((cosine_similarity(query_vector, v), i) for i, v in enumerate(doc_vectors)), reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Interaktív keresés
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    offline = "--offline" in sys.argv
    embed = embed_offline if offline else embed_voyage

    print(f"Dokumentumok embeddingelése ({'offline, szóegyezés' if offline else VOYAGE_MODEL})...")
    doc_vectors = embed(DOCUMENTS, "document")  # egyszer kell kiszámolni, érdemes eltárolni (fájl, vektoradatbázis)
    print(f"Kész: {len(doc_vectors)} dokumentum, egy vektor {len(doc_vectors[0])} számból áll.")
    print(f"Példa az első vektor elejére: {[round(x, 3) for x in doc_vectors[0][:5]]}\n")
    print("Kérdezz valamit, pl. 'Mennyi idő alatt kapom vissza a pénzem?' vagy 'Hová menjek nyaralni?'. 'exit' a kilépéshez.\n")

    while True:
        query = input("🔎 Kérdés: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        query_vector = embed([query], "query")[0]
        for score, i in search(query_vector, doc_vectors):
            print(f"   {score:5.2f}  {DOCUMENTS[i]}")
        print()
