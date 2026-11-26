import asyncio
import json
import os
import re
import time
from utils import client


# ============================================================================
# MI EZ? - Teljes prompt evaluátor konkurrens futtatással
# ============================================================================
#
# Ez a fájl egyesíti a korábbi példákat egyetlen, komplett eszközbe:
#   - eval.py                → LLM-bíró (model-based grading)
#   - eval_code_grade.py     → kód alapú értékelés (code-based grading)
#   - eval_grade_by_model.py → több modell összehasonlítása
#
# ÚJ: KONKURRENS FUTTATÁS
# A korábbi scriptek egyesével, sorban hívták az API-t - lassú volt. Itt
# asyncio.gather-rel PÁRHUZAMOSAN futtatjuk a teszteseteket. Egy
# Semaphore korlátozza, hány hívás futhat egyszerre, nehogy rate limitbe
# ütközzünk (a bifrost proxy és az API is korlátozza a párhuzamos kéréseket).
#
# Minden tesztesethez megadható a grader típusa:
#   - KÓD alapú: contains, regex, number_range, all_of, any_of, json_schema
#   - LLM alapú: llm_judge (egy bíró modell értékel 1-10 skálán)
# ============================================================================


SYSTEM_PROMPT = "Válaszolj tömören és pontosan. Csak a kért információt add meg."
GRADER_MODEL = "us.anthropic.claude-fable-5-2"  # a legjobb elérhető modell bíróként
MAX_CONCURRENCY = 5  # egyszerre legfeljebb ennyi API-hívás futhat


TEST_CASES = [
    {"input": "Mi Magyarország fővárosa?", "grader": {"type": "contains", "value": "budapest"}},
    {"input": "Mennyi 15 + 27?", "grader": {"type": "regex", "pattern": r"\b42\b"}},
    {"input": "Mi a víz kémiai képlete?", "grader": {"type": "regex", "pattern": r"[Hh]2?₂?[Oo]"}},
    {"input": "Melyik a legnagyobb bolygó a Naprendszerben?", "grader": {"type": "contains", "value": "jupiter"}},
    {"input": "Mi a Föld kerülete az egyenlítőnél (kb.)?", "grader": {"type": "number_range", "min": 35000, "max": 45000}},
    {"input": "Sorolj fel három magyar folyót.", "grader": {"type": "all_of", "values": ["duna", "tisza"]}},
    {"input": "Melyik évben alapították a Fideszt?", "grader": {"type": "contains", "value": "1988"}},
    {
        "input": (
            "Adj vissza KIZÁRÓLAG egy JSON objektumot Orbán Viktorról, más szöveg nélkül:\n"
            '{"name": "teljes név", "birth_year": 1963, "party": "párt neve"}'
        ),
        "grader": {
            "type": "json_schema",
            "required_keys": ["name", "birth_year", "party"],
            "field_types": {"name": str, "birth_year": int, "party": str},
            "field_contains": {"party": "fidesz"},
        },
    },
    {
        "input": "Magyarázd el röviden, mi a fotoszintézis.",
        # Nyílt végű kérdés - itt nincs egyetlen jó kulcsszó, LLM-bíró kell
        "grader": {
            "type": "llm_judge",
            "criteria": "A válasz helyesen elmagyarázza, hogy a növények fényből energiát állítanak elő, és megemlíti a szén-dioxidot vagy az oxigént.",
        },
    },
    {
        "input": "Írj egy rövid, biztató üzenetet egy diáknak vizsga előtt.",
        "grader": {
            "type": "llm_judge",
            "criteria": "A válasz pozitív, biztató hangvételű, és vizsgára vonatkozik. Magyarul van.",
        },
    },
]


# --------------------------------------------------------------------------
# Kód alapú graderek (szinkron, nincs API-hívás)
# --------------------------------------------------------------------------

def extract_first_number(text):
    match = re.search(r"[\d\s]+", text.replace(",", "").replace(".", ""))
    if match:
        num_str = match.group().replace(" ", "")
        return int(num_str) if num_str else None
    return None


def grade_with_code(output, grader):
    # Determinisztikus, kód alapú értékelés - 1-10 pontot ad (0 vagy 10)
    text = output.lower().strip()
    t = grader["type"]

    if t == "contains":
        ok = grader["value"].lower() in text
        detail = f"tartalmaz: '{grader['value']}'" if ok else f"hiányzik: '{grader['value']}'"
    elif t == "regex":
        m = re.search(grader["pattern"], output, re.IGNORECASE)
        ok = m is not None
        detail = f"illeszkedett: '{m.group()}'" if ok else f"nem illeszkedett: '{grader['pattern']}'"
    elif t == "number_range":
        num = extract_first_number(output)
        ok = num is not None and grader["min"] <= num <= grader["max"]
        detail = f"{num} a [{grader['min']}, {grader['max']}] tartományban" if ok else f"{num} kívül esik vagy hiányzik"
    elif t == "all_of":
        missing = [v for v in grader["values"] if v.lower() not in text]
        ok = not missing
        detail = "mind megvan" if ok else f"hiányzik: {missing}"
    elif t == "any_of":
        found = [v for v in grader["values"] if v.lower() in text]
        ok = bool(found)
        detail = f"megtalálva: {found}" if ok else f"egyik sem: {grader['values']}"
    elif t == "json_schema":
        ok, detail = grade_json_schema(output, grader)
    else:
        ok, detail = False, f"ismeretlen grader típus: {t}"

    return {"score": 10 if ok else 0, "detail": detail}


def grade_json_schema(output, grader):
    raw = output.strip()
    is_list = grader.get("root_type") == list
    start = raw.find("[") if is_list else raw.find("{")
    end = raw.rfind("]") if is_list else raw.rfind("}")
    if start == -1 or end == -1:
        return False, "nem található JSON"
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        return False, f"érvénytelen JSON: {e}"

    root_type = grader.get("root_type", dict)
    if not isinstance(parsed, root_type):
        return False, f"várt: {root_type.__name__}, kapott: {type(parsed).__name__}"

    if isinstance(parsed, list):
        min_len = grader.get("min_length", 0)
        if len(parsed) < min_len:
            return False, f"lista {len(parsed)} elem, min {min_len}"
        return True, f"JSON tömb, {len(parsed)} elem"

    missing = [k for k in grader.get("required_keys", []) if k not in parsed]
    if missing:
        return False, f"hiányzó mezők: {missing}"
    for key, exp_type in grader.get("field_types", {}).items():
        if key in parsed and not isinstance(parsed[key], exp_type):
            return False, f"'{key}' típusa {type(parsed[key]).__name__}, várt: {exp_type.__name__}"
    for key, exp_val in grader.get("field_contains", {}).items():
        if exp_val and exp_val.lower() not in str(parsed[key]).lower():
            return False, f"'{key}' nem tartalmazza: '{exp_val}'"
    return True, f"valid JSON, mezők: {list(parsed.keys())}"


# --------------------------------------------------------------------------
# LLM alapú grader (aszinkron, külön API-hívás)
# --------------------------------------------------------------------------

async def grade_with_llm(user_input, output, grader):
    # Egy bíró modell értékel 1-10 skálán, JSON-t ad vissza
    prompt = f"""Értékeld az alábbi AI-választ a megadott szempont alapján.

Kérdés: {user_input}
Válasz: {output}
Értékelési szempont: {grader['criteria']}

Adj vissza KIZÁRÓLAG egy JSON objektumot, más szöveg nélkül:
{{"score": 1-10 közötti egész, "reasoning": "rövid indoklás"}}"""

    message = await client.messages.create(
        model=GRADER_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = next(b.text for b in message.content if b.type == "text").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        return {"score": 0, "detail": f"nem sikerült JSON-t kinyerni: {raw[:60]}"}
    data = json.loads(raw[s:e + 1])
    return {"score": data.get("score", 0), "detail": data.get("reasoning", "")}


# --------------------------------------------------------------------------
# Egy teszteset futtatása + értékelése (konkurrens egység)
# --------------------------------------------------------------------------

async def run_prompt(model, user_input):
    message = await client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
    )
    return next(b.text for b in message.content if b.type == "text")


async def evaluate_case(index, case, model, semaphore):
    # A semaphore biztosítja, hogy egyszerre legfeljebb MAX_CONCURRENCY
    # teszteset fusson. A 'with' blokkon belül vagyunk "slotban".
    async with semaphore:
        try:
            output = await run_prompt(model, case["input"])
            if case["grader"]["type"] == "llm_judge":
                result = await grade_with_llm(case["input"], output, case["grader"])
            else:
                result = grade_with_code(output, case["grader"])
        except Exception as e:
            output, result = "", {"score": 0, "detail": f"hiba: {e}"}

    # Egy dict-et adunk vissza, hogy az eredeti sorrend visszaállítható legyen
    return {
        "index": index,
        "input": case["input"],
        "output": output,
        "grader_type": case["grader"]["type"],
        "score": result["score"],
        "detail": result["detail"],
    }


# --------------------------------------------------------------------------
# Fő futtató: minden tesztesetet párhuzamosan indít
# --------------------------------------------------------------------------

async def main():
    model = os.environ["ANTHROPIC_MODEL"]
    print(f"🧪 Konkurrens prompt evaluátor")
    print(f"   Modell      : {model}")
    print(f"   Tesztesetek : {len(TEST_CASES)}")
    print(f"   Párhuzamos  : max {MAX_CONCURRENCY} egyszerre\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    start = time.time()

    # Az összes tesztesetet EGYSZERRE indítjuk; a semaphore szabályozza,
    # ténylegesen hány fut párhuzamosan. A gather megvárja mindet.
    tasks = [
        evaluate_case(i, case, model, semaphore)
        for i, case in enumerate(TEST_CASES, 1)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    # Eredeti sorrendben nyomtatjuk (a gather sorrendtartó, de biztos ami biztos)
    results.sort(key=lambda r: r["index"])
    total_score = 0
    for r in results:
        mark = "✅" if r["score"] >= 6 else "❌"
        total_score += r["score"]
        print(f"{mark} {r['index']:2}. [{r['grader_type']}] {r['input'][:55]}")
        print(f"      Válasz: {r['output'].strip()[:80]}")
        print(f"      Pont  : {r['score']}/10 — {r['detail']}\n")

    avg = total_score / len(results)
    passed = sum(1 for r in results if r["score"] >= 6)
    print("=" * 60)
    print(f"📊 Átlag: {avg:.1f}/10 | Sikeres: {passed}/{len(results)} | Idő: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
