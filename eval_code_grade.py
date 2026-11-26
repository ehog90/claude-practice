import asyncio
import json
import os
import re
from utils import client


# ============================================================================
# MI EZ? - Kód alapú értékelés (code-based grading)
# ============================================================================
#
# Az eval.py és eval_grade_by_model.py LLM-bírót használt az értékeléshez.
# Ez drága, lassú, és maga is hibázhat. Ha a helyes válasz egyértelműen
# ellenőrizhető (szám, dátum, kulcsszó, regex), érdemesebb KÓDDAL értékelni.
#
# Előnyök az LLM-bíróhoz képest:
#   - Gyorsabb: nincs extra API-hívás értékelésenként
#   - Olcsóbb: feleannyi token-t költ
#   - Determinisztikus: ugyanaz a bemenet mindig ugyanazt az eredményt adja
#   - Átlátható: pontosan látod, miért bukott el egy teszt
#
# Különböző grader típusok ebben a fájlban:
#   - contains      → tartalmaz-e egy adott szöveget? (kis/nagybetű-érzéketlen)
#   - exact         → pontosan megegyezik-e? (strip után)
#   - regex         → illeszkedik-e egy mintára?
#   - number_range  → egy szám beleesik-e egy tartományba?
#   - all_of        → több contains feltétel mindegyike teljesül-e?
#   - any_of        → több contains feltétel legalább egyike teljesül-e?
#   - json_schema   → valid JSON-e, és megvannak-e a kötelező mezők helyes típussal?
# ============================================================================


SYSTEM_PROMPT = "Válaszolj tömören és pontosan. Csak a kért információt add meg."

# Minden tesztesethez meg kell adni a grader típusát és paramétereit.
# Ez a dataset nem JSON-ból töltődik, mert a grader logika kódban van.
TEST_CASES = [
    {
        "input": "Mi Magyarország fővárosa?",
        "grader": {"type": "contains", "value": "budapest"},
    },
    {
        "input": "Mennyi 15 + 27?",
        # Pontosan a 42-t keresi számként — „42-t", „42." is megfelel
        "grader": {"type": "regex", "pattern": r"\b42\b"},
    },
    {
        "input": "Mi a víz kémiai képlete?",
        # H2O vagy H₂O (rendes és unicode index) is elfogadható
        "grader": {"type": "regex", "pattern": r"[Hh]2?₂?[Oo]"},
    },
    {
        "input": "Melyik a legnagyobb bolygó a Naprendszerben?",
        "grader": {"type": "contains", "value": "jupiter"},
    },
    {
        "input": "Mikor született Mozart?",
        # Az évet keresi — 1756 bárhol a szövegben
        "grader": {"type": "contains", "value": "1756"},
    },
    {
        "input": "Mi a Föld kerülete az egyenlítőnél (kb.)?",
        # 35 000 és 45 000 közé kell esnie az első talált számnak
        "grader": {"type": "number_range", "min": 35000, "max": 45000},
    },
    {
        "input": "Fordítsd le angolra: 'Jó reggelt kívánok!'",
        "grader": {"type": "contains", "value": "good morning"},
    },
    {
        "input": "Sorolj fel három magyar folyót.",
        # Mindháromnak benne kell lennie
        "grader": {"type": "all_of", "values": ["duna", "tisza"]},
    },
    {
        "input": "Ki írta a Toldi című eposzt?",
        "grader": {"type": "contains", "value": "arany"},
    },
    {
        "input": "Melyik évben alapították a Fideszt?",
        "grader": {"type": "contains", "value": "1988"},
    },
    {
        "input": "Mikor lett először Orbán Viktor Magyarország miniszterelnöke?",
        "grader": {"type": "contains", "value": "1998"},
    },
    {
        "input": "Melyik párthoz tartozik Orbán Viktor?",
        "grader": {"type": "contains", "value": "fidesz"},
    },
    {
        "input": "Ki Magyarország jelenlegi miniszterelnöke?",
        # 2026 után Magyar Péter — either works
        "grader": {"type": "any_of", "values": ["magyar péter", "magyar peter"]},
    },
    # --- JSON SÉMA TESZTESETEK ---
    # Ezek azt ellenőrzik, hogy a modell valid, strukturált JSON-t ad-e vissza.
    # A json_schema grader három dolgot vizsgál:
    #   1. Valid JSON-e a válasz? (parse-olható?)
    #   2. Megvannak-e a kötelező mezők? (required_keys)
    #   3. Helyes típusú-e minden mező? (field_types: str, int, float, list, dict, bool)
    {
        "input": (
            "Adj vissza KIZÁRÓLAG egy JSON objektumot Orbán Viktorról, más szöveg nélkül:\n"
            '{"name": "teljes név", "birth_year": 1963, "party": "párt neve"}'
        ),
        "grader": {
            "type": "json_schema",
            "required_keys": ["name", "birth_year", "party"],
            "field_types": {"name": str, "birth_year": int, "party": str},
            # Opcionális értékellenőrzés: a mező értéke tartalmazza-e a stringet?
            "field_contains": {"party": "fidesz", "birth_year": None},
        },
    },
    {
        "input": (
            "Adj vissza KIZÁRÓLAG egy JSON objektumot Magyarország adataival, más szöveg nélkül:\n"
            '{"capital": "főváros", "population_millions": 9.7, "neighbors": ["ország1", "ország2"]}'
        ),
        "grader": {
            "type": "json_schema",
            "required_keys": ["capital", "population_millions", "neighbors"],
            "field_types": {"capital": str, "population_millions": float, "neighbors": list},
            "field_contains": {"capital": "budapest"},
        },
    },
    {
        "input": (
            "Adj vissza KIZÁRÓLAG egy JSON tömböt, más szöveg nélkül, "
            "amelyben 3 magyar folyó neve szerepel stringként: [\"folyó1\", \"folyó2\", \"folyó3\"]"
        ),
        "grader": {
            "type": "json_schema",
            # Lista esetén a root_type jelzi, hogy tömböt várunk (nem objektumot)
            "root_type": list,
            "min_length": 3,
        },
    },
    {
        "input": (
            "Adj vissza KIZÁRÓLAG egy JSON objektumot, más szöveg nélkül:\n"
            '{"question": "Mi 2+2?", "answer": 4, "is_math": true}'
        ),
        "grader": {
            "type": "json_schema",
            "required_keys": ["question", "answer", "is_math"],
            "field_types": {"question": str, "answer": int, "is_math": bool},
        },
    },
]


def extract_first_number(text):
    # Kiszedi az első számot a szövegből (pl. „kb. 40 075 km" → 40075)
    match = re.search(r"[\d\s]+", text.replace(",", "").replace(".", ""))
    if match:
        num_str = match.group().replace(" ", "")
        return int(num_str) if num_str else None
    return None


def run_grader(output, grader):
    # Kód alapú értékelő — típustól függően más logikát futtat
    text = output.lower().strip()
    t = grader["type"]

    if t == "contains":
        passed = grader["value"].lower() in text
        detail = f"tartalmaz: '{grader['value']}'" if passed else f"hiányzik: '{grader['value']}'"

    elif t == "exact":
        passed = text == grader["value"].lower()
        detail = "pontos egyezés" if passed else f"várt: '{grader['value']}', kapott: '{text[:60]}'"

    elif t == "regex":
        match = re.search(grader["pattern"], output, re.IGNORECASE)
        passed = match is not None
        detail = f"minta illeszkedett: '{match.group()}'" if passed else f"minta nem illeszkedett: '{grader['pattern']}'"

    elif t == "number_range":
        num = extract_first_number(output)
        if num is None:
            passed, detail = False, "nem találtam számot a válaszban"
        else:
            passed = grader["min"] <= num <= grader["max"]
            detail = f"{num} {'beleesik' if passed else 'kívül esik'} a [{grader['min']}, {grader['max']}] tartományon"

    elif t == "all_of":
        missing = [v for v in grader["values"] if v.lower() not in text]
        passed = len(missing) == 0
        detail = "mind megvan" if passed else f"hiányzik: {missing}"

    elif t == "any_of":
        found = [v for v in grader["values"] if v.lower() in text]
        passed = len(found) > 0
        detail = f"megtalálva: {found}" if passed else f"egyik sem szerepel: {grader['values']}"

    elif t == "json_schema":
        # 1. lépés: kinyerjük a JSON-t (a modell néha magyarázatot is fűz hozzá)
        raw = output.strip()
        start = raw.find("[") if grader.get("root_type") == list else raw.find("{")
        end = raw.rfind("]") if grader.get("root_type") == list else raw.rfind("}")

        if start == -1 or end == -1:
            return {"pass": False, "detail": "nem található JSON a válaszban"}

        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as e:
            return {"pass": False, "detail": f"érvénytelen JSON: {e}"}

        # 2. lépés: gyökér típus ellenőrzés (lista vs. objektum)
        root_type = grader.get("root_type", dict)
        if not isinstance(parsed, root_type):
            return {"pass": False, "detail": f"várt típus: {root_type.__name__}, kapott: {type(parsed).__name__}"}

        # 3. lépés: lista minimális hossz
        if isinstance(parsed, list):
            min_len = grader.get("min_length", 0)
            if len(parsed) < min_len:
                return {"pass": False, "detail": f"lista hossza {len(parsed)}, minimum {min_len} szükséges"}
            return {"pass": True, "detail": f"valid JSON tömb, {len(parsed)} elemmel: {parsed}"}

        # 4. lépés: kötelező mezők megléte
        missing = [k for k in grader.get("required_keys", []) if k not in parsed]
        if missing:
            return {"pass": False, "detail": f"hiányzó mezők: {missing}"}

        # 5. lépés: mezők típusellenőrzése
        for key, expected_type in grader.get("field_types", {}).items():
            if key in parsed and not isinstance(parsed[key], expected_type):
                return {"pass": False, "detail": f"'{key}' mező típusa {type(parsed[key]).__name__}, várt: {expected_type.__name__}"}

        # 6. lépés: opcionális értékellenőrzés (tartalmaz-e egy stringet?)
        for key, expected_val in grader.get("field_contains", {}).items():
            if expected_val and key in parsed:
                actual = str(parsed[key]).lower()
                if expected_val.lower() not in actual:
                    return {"pass": False, "detail": f"'{key}' mező értéke '{parsed[key]}', nem tartalmazza: '{expected_val}'"}

        passed = True
        detail = f"valid JSON, mezők: {list(parsed.keys())}, értékek: { {k: parsed[k] for k in list(parsed.keys())[:3]} }"

    else:
        passed, detail = False, f"ismeretlen grader típus: {t}"

    return {"pass": passed, "detail": detail}


async def run_prompt(user_input):
    message = await client.messages.create(
        model=os.environ["ANTHROPIC_MODEL"],
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
    )
    return next(b.text for b in message.content if b.type == "text")


async def main():
    print(f"🧪 Kód alapú kiértékelés — modell: {os.environ['ANTHROPIC_MODEL']}\n")
    passed_count = 0

    for i, case in enumerate(TEST_CASES, 1):
        output = await run_prompt(case["input"])
        result = run_grader(output, case["grader"])

        mark = "✅" if result["pass"] else "❌"
        if result["pass"]:
            passed_count += 1

        print(f"{mark} {i:2}. {case['input']}")
        print(f"      Válasz : {output.strip()[:100]}")
        print(f"      Grader : [{case['grader']['type']}] {result['detail']}\n")

    total = len(TEST_CASES)
    pct = 100 * passed_count // total
    bar = "█" * passed_count + "░" * (total - passed_count)
    print(f"📊 Eredmény: {passed_count}/{total} [{bar}] {pct}%")


if __name__ == "__main__":
    asyncio.run(main())
