import asyncio
import json
import os
from utils import client


# ============================================================================
# MI EZ? - Prompt kiértékelés (prompt evaluation / "eval")
# ============================================================================
#
# A prompt kiértékelés arra szolgál, hogy megmérjük: mennyire jól teljesít egy
# prompt. Ahelyett, hogy kézzel, szemre ellenőriznénk minden választ, egy
# automatizált tesztsorozatot futtatunk - pont úgy, mint a szoftverteszteknél.
#
# A működés három lépésből áll:
#   1. ADATHALMAZ (dataset.json): előre összeállított kérdések, és hozzájuk
#      tartozó elvárások (mi számít helyes válasznak).
#   2. FUTTATÁS (run_prompt): a tesztelendő promptot lefuttatjuk minden
#      bemeneten, és elmentjük a modell válaszát.
#   3. OSZTÁLYOZÁS (grade): mivel a szöveges válaszokat nehéz gépileg
#      ellenőrizni, egy MÁSIK modellhívást használunk "bíróként"
#      (LLM-as-judge). A bíró eldönti, hogy a válasz megfelel-e az elvárásnak.
#
# A végén egy összesített pontszámot kapunk (pl. 3/3 sikeres). Ha módosítjuk a
# promptot, ezzel a futtatással objektíven összehasonlíthatjuk, hogy jobb vagy
# rosszabb lett-e - ez a "hillclimbing", a prompt fokozatos javítása.
# ============================================================================


# A kiértékelendő prompt - ezt teszteljük a példákon
SYSTEM_PROMPT = "Válaszolj tömören és pontosan. Csak a kért információt add meg."

# A teszteseteket külső JSON fájlból töltjük be, hogy könnyen bővíthető legyen
# anélkül, hogy a kódhoz kellene nyúlni
with open(os.path.join(os.path.dirname(__file__), "eval_dataset.json"), encoding="utf-8") as f:
    TEST_CASES = json.load(f)


async def run_prompt(user_input):
    # Lefuttatja a tesztelendő promptot egy adott bemeneten
    message = await client.messages.create(
        model=os.environ["ANTHROPIC_MODEL"],
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
    )
    return next(b.text for b in message.content if b.type == "text")


async def grade(user_input, output, criteria):
    # LLM-bíró: egy külön modellhívás értékeli a kimenetet a szempont alapján.
    # JSON formátumot kérünk, hogy gépileg feldolgozható legyen a pontszám.
    grader_prompt = f"""Értékeld az alábbi AI-választ a megadott szempont alapján.

Kérdés: {user_input}
Válasz: {output}
Értékelési szempont: {criteria}

Adj vissza KIZÁRÓLAG egy JSON objektumot, más szöveg nélkül, ebben a formában:
{{"pass": true vagy false, "reason": "rövid indoklás"}}"""

    message = await client.messages.create(
        model=os.environ["ANTHROPIC_MODEL"],
        max_tokens=300,
        messages=[{"role": "user", "content": grader_prompt}],
    )
    raw = message.content[0].text.strip()

    # Kinyerjük a JSON-t a válaszból (a modell néha magyarázatot is fűz hozzá)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"pass": False, "reason": f"Nem sikerült JSON-t kinyerni: {raw}"}
    return json.loads(raw[start : end + 1])


async def main():
    print("🧪 Prompt kiértékelés indítása...\n")
    passed = 0

    for i, case in enumerate(TEST_CASES, 1):
        # Minden teszteset: prompt futtatás, majd értékelés
        output = await run_prompt(case["input"])
        result = await grade(case["input"], output, case["criteria"])

        mark = "✅" if result["pass"] else "❌"
        if result["pass"]:
            passed += 1

        print(f"{mark} Teszt {i}: {case['input']}")
        print(f"   Válasz: {output}")
        print(f"   Indoklás: {result['reason']}\n")

    # Összesített eredmény
    total = len(TEST_CASES)
    print(f"📊 Eredmény: {passed}/{total} teszt sikeres ({100 * passed // total}%)")


if __name__ == "__main__":
    asyncio.run(main())
