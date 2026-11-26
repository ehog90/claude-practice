"""Prompt caching példa: a nagy, ismétlődő promptrészt egyszer fizetjük meg teljes áron, utána olcsón olvassuk.

A probléma: ha minden kérésben ugyanazt a nagy szöveget küldjük (hosszú system prompt, dokumentum, tool-lista),
akkor azt minden alkalommal újra feldolgozza és kiszámlázza az API. A cache ezt kiváltja:

  1. cache írás  (cache_creation_input_tokens): az első kérés a prefixet eltárolja, ~1,25x áron
  2. cache olvasás (cache_read_input_tokens):   a következő kérések ugyanezt ~0,1x áron olvassák, és gyorsabbak is
  3. a cache alapból 5 percig él, és minden olvasás megújítja ("ttl": "1h" esetén egy óráig)

A legfontosabb szabály: a cache ELŐTAG-egyezésre (prefix match) épül. A kérés sorrendje tools -> system -> messages,
és a cache a megadott töréspontig tartó teljes előtagra érvényes. Ha az előtag BÁRMELY bájtja változik, onnantól minden
újra feldolgozásra kerül. Ezért:
  - a stabil tartalom (system prompt, dokumentum, tool-lista) menjen előre, változatlanul,
  - a változó rész (kérdés, időbélyeg, kérés-azonosító) mindig a töréspont UTÁN legyen.

A demó három szakaszból áll:
  A) helyes használat: ugyanaz a nagy system prompt, más-más kérdések -> az első ír, a többi olvas
  B) tipikus hiba: időbélyeg a system prompt elején -> minden kérésnél más az előtag, a cache sosem talál
  C) a javított változat: az időbélyeg a kérdés mellett, a töréspont után -> újra működik

Indítás:  python caching_demo.py [--fresh]   (--fresh: tiszta cache-ről indul, akkor is, ha 5 percen belül újrafuttatod)

Ellenőrzés: mindig a `usage` mezőket nézd. Ha a cache_read_input_tokens végig 0, valami "csendben" érvényteleníti a cache-t.
Alsó korlát: a cache-elhető előtagnak elég hosszúnak kell lennie (modellfüggő, kb. néhány ezer token), különben nem is próbálkozik.
"""

import os
import sys
import time
import uuid
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]

# Listaárak USD / 1M token (Claude Sonnet 5.5 szerint; más modellnél, vagy ha a végpontod más árat számol, módosítsd)
PRICE_INPUT, PRICE_OUTPUT = 2.00, 10.00
PRICE_CACHE_WRITE = PRICE_INPUT * 1.25  # 5 perces cache írása
PRICE_CACHE_READ = 0.10


# ---------------------------------------------------------------------------
# Egy nagy, stabil dokumentum (a cache-elni kívánt rész)
# ---------------------------------------------------------------------------

TOPICS = ["visszaküldés", "szállítás", "garancia", "fizetés", "adatkezelés", "ügyfélszolgálat", "kuponok", "számlázás"]


def build_policy_document(sections: int = 48) -> str:
    """Egy hosszú, szakaszokra osztott szabályzatot állít elő (determinisztikusan: ugyanaz a szöveg minden futásnál)."""
    parts = ["ÁLTALÁNOS SZERZŐDÉSI FELTÉTELEK (részletes változat)\n"]
    for n in range(1, sections + 1):
        topic = TOPICS[n % len(TOPICS)]
        parts.append(
            f"{n}. szakasz: {topic.capitalize()} ({n}. pont)\n"
            f"A(z) {topic} kapcsán a szolgáltató a vásárlástól számított {10 + n} napon belül fogadja a bejelentéseket. "
            f"Az ügyfél a {100 + n * 7}. számú nyomtatvány kitöltésével kezdeményezheti az eljárást, amelyet a szolgáltató "
            f"legkésőbb {2 + n % 5} munkanapon belül elbírál. A(z) {n}. pontban rögzített díjak és határidők a mindenkori "
            f"hatályos árlista szerint módosulhatnak, a módosításról az ügyfelet 15 nappal korábban értesítjük. "
            f"Vitás esetben a(z) {topic} ügyében a fogyasztóvédelmi szabályok az irányadók.\n"
        )
    return "\n".join(parts)


DOCUMENT = build_policy_document()
INSTRUCTIONS = "Ügyfélszolgálati asszisztens vagy. Az alábbi szabályzat alapján válaszolj, röviden, magyarul, markup nélkül.\n\n"

QUESTIONS = [
    "Hány napon belül fogadják a visszaküldési bejelentéseket az 5. szakasz szerint?",
    "Hányas számú nyomtatványt kell kitölteni a 12. szakasz szerinti eljáráshoz?",
    "Hány munkanapon belül bírálják el a 20. szakasz szerinti ügyet?",
]


# ---------------------------------------------------------------------------
# Kérés + használat kiírása
# ---------------------------------------------------------------------------


def ask(question: str, system_text: str, user_prefix: str = "") -> dict:
    """Egy kérés. A system prompt blokkján lévő cache_control a TÖRÉSPONT: az előtag (tools + system) idáig cache-elődik.
    A kérdés (user üzenet) a töréspont UTÁN van, ezért szabadon változhat.

    Megjegyzés: létezik "automatikus" változat is (felső szintű `cache_control=` paraméter a messages.create-en),
    de nem minden végpont támogatja: ezen a gatewayen nem hatott (a cache mezők végig 0 maradtak),
    ezért itt az explicit, blokk szintű töréspontot használjuk, ami mindenhol működik.
    """
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=[{
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},  # 5 perc; hosszabbhoz: {"type": "ephemeral", "ttl": "1h"}
        }],
        messages=[{"role": "user", "content": f"{user_prefix}{question}"}],
    )
    u = response.usage
    return {
        "uncached": u.input_tokens,                            # a töréspont utáni, nem cache-elt bemenet
        "write": u.cache_creation_input_tokens or 0,           # most írtuk a cache-be
        "read": u.cache_read_input_tokens or 0,                # a cache-ből olvastuk
        "output": u.output_tokens,
        "seconds": time.perf_counter() - start,
        "text": "".join(b.text for b in response.content if b.type == "text"),
    }


def cost(r: dict) -> float:
    return (r["uncached"] * PRICE_INPUT + r["write"] * PRICE_CACHE_WRITE + r["read"] * PRICE_CACHE_READ + r["output"] * PRICE_OUTPUT) / 1e6


def print_table(rows: list[dict]):
    print(f"   {'#':>2} {'nem cache-elt':>14} {'cache írás':>11} {'cache olvasás':>14} {'idő':>6} {'költség':>10}")
    for n, r in enumerate(rows, start=1):
        print(f"   {n:>2} {r['uncached']:>14} {r['write']:>11} {r['read']:>14} {r['seconds']:>5.1f}s ${cost(r):>8.5f}")
    print(f"   {'':>2} {'':>41} összesen ${sum(cost(r) for r in rows):>8.5f}\n")


if __name__ == "__main__":
    print(f"A dokumentum {len(DOCUMENT):,} karakter ({len(DOCUMENT.split())} szó).\n")

    # --- A) helyes használat: stabil előtag ---------------------------------
    print("A) Stabil system prompt, változó kérdések: az 1. kérés ír, a többi olvas")
    stable_system = INSTRUCTIONS + DOCUMENT
    if "--fresh" in sys.argv:
        # Ha 5 percen belül újrafuttatod, a cache még meleg, és már az 1. kérés is olvas. A --fresh egy futásonként
        # egyedi azonosítót tesz az előtag elejére, így minden futás tiszta cache-ről indul (a futáson belül állandó).
        stable_system = f"[futtatás: {uuid.uuid4()}]\n" + stable_system
    rows_a = [ask(q, stable_system) for q in QUESTIONS]  # sorban, mert a cache az első válasz megkezdése után érhető el
    print_table(rows_a)
    for q, r in zip(QUESTIONS, rows_a):
        print(f"   ❓ {q}\n   💬 {r['text']}\n")

    # --- B) tipikus hiba: az időbélyeg az előtag elején ---------------------
    print("B) HIBA: időbélyeg a system prompt elején -> minden kérésnél más az előtag, a cache sosem talál")
    rows_b = []
    for q in QUESTIONS[:2]:
        rows_b.append(ask(q, f"Jelenlegi idő: {datetime.now().isoformat()}\n{INSTRUCTIONS}{DOCUMENT}"))
        time.sleep(1.1)  # hogy az időbélyeg biztosan eltérjen
    print_table(rows_b)

    # --- C) javítva: a változó rész a töréspont után -------------------------
    print("C) JAVÍTVA: az időbélyeg a kérdés mellett van, az előtag változatlan -> újra olvas a cache")
    rows_c = [ask(q, stable_system, user_prefix=f"[Jelenlegi idő: {datetime.now().isoformat(timespec='seconds')}] ") for q in QUESTIONS[:2]]
    print_table(rows_c)

    # --- Összegzés -------------------------------------------------------------
    print("Összegzés: a cache_read_input_tokens az első kérés után A) és C) esetén nagy, B) esetén végig 0.")
    print("Ha ezt a számot nem látod nőni, keresd a csendes érvénytelenítőt: időbélyeg, UUID, rendezetlen JSON, változó tool-lista.")
