import asyncio
import json
import os
from utils import client


# ============================================================================
# MI EZ? - Modellek összehasonlítása (grade by model)
# ============================================================================
#
# Az eval.py-ban egy modellt teszteltünk egy prompton. Ez a fájl egy lépéssel
# tovább megy: ugyanazokat a teszteseteket TÖBB modellen futtatjuk, és
# összehasonlítjuk az eredményeket.
#
# Mire jó ez?
#   - Megmutatja, melyik modell teljesít jobban egy adott feladaton.
#   - Segít eldönteni, megéri-e drágább modellt használni (pontosabb-e?).
#   - Felfedheti, hogy egy olcsóbb modell ugyanolyan jól teljesít-e.
#
# A bíró (grader) minden esetben UGYANAZ a modell marad, hogy az értékelés
# konzisztens legyen. Csak a tesztelendő modell változik.
#
# Kimenet: modellenként egy összesített pontszám, egymás mellett.
# ============================================================================


SYSTEM_PROMPT = "Válaszolj tömören és pontosan. Csak a kért információt add meg."

# A tesztelendő modellek listája - ezeket hasonlítjuk össze
# (us.anthropic.* prefix szükséges a bifrost proxy miatt; haiku-5-5 nem elérhető)
MODELS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5-5",
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5-5",
    "us.anthropic.claude-fable-5",
    "us.anthropic.claude-fable-5-1",
    "us.anthropic.claude-fable-5-2",
]

# A bíró modell - ez értékeli az összes választ, nem változik
GRADER_MODEL = "us.anthropic.claude-fable-5-2"

with open(os.path.join(os.path.dirname(__file__), "eval_dataset.json"), encoding="utf-8") as f:
    TEST_CASES = json.load(f)


async def run_prompt(model, user_input):
    # Lefuttatja a promptot a megadott modellen
    message = await client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
    )
    return next(b.text for b in message.content if b.type == "text")


async def grade(user_input, output, criteria):
    # A bíró részletes, strukturált értékelést ad vissza JSON formában.
    # strengths/weaknesses tömbök + indoklás + 1-10 pontszám.
    grader_prompt = f"""Értékeld az alábbi AI-választ a megadott szempont alapján.

Kérdés: {user_input}
Válasz: {output}
Értékelési szempont: {criteria}

Adj vissza KIZÁRÓLAG egy JSON objektumot, más szöveg nélkül, ebben a formában:
{{
  "strengths": ["erősség 1", "erősség 2"],
  "weaknesses": ["gyengeség 1", "gyengeség 2"],
  "reasoning": "rövid összefoglaló indoklás",
  "score": 7
}}

A score 1-10 közötti egész szám legyen (10 = tökéletes, 1 = teljesen helytelen).
Ha nincs erősség vagy gyengeség, adj vissza üres tömböt."""

    message = await client.messages.create(
        model=GRADER_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": grader_prompt}],
    )
    raw = next(b.text for b in message.content if b.type == "text").strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {"strengths": [], "weaknesses": [], "reasoning": f"Nem sikerült JSON-t kinyerni: {raw}", "score": 0}
    return json.loads(raw[start : end + 1])


def print_grade(i, case, output, result):
    # Szépen formázott kimenet egy tesztesethez
    score = result.get("score", 0)
    stars = "★" * score + "☆" * (10 - score)
    mark = "✅" if score >= 6 else "❌"

    print(f"\n{mark} {i:2}. {case['input']}")
    print(f"   Válasz : {output.strip()}")
    print(f"   Pontszám: [{stars}] {score}/10")
    print(f"   Indoklás: {result.get('reasoning', '')}")

    if result.get("strengths"):
        for s in result["strengths"]:
            print(f"   ✔ {s}")
    if result.get("weaknesses"):
        for w in result["weaknesses"]:
            print(f"   ✘ {w}")


async def eval_model(model):
    # Egy teljes tesztsorozat futtatása egyetlen modellen
    print(f"\n🤖 Modell: {model}")
    print("-" * 60)
    total_score = 0

    for i, case in enumerate(TEST_CASES, 1):
        output = await run_prompt(model, case["input"])
        result = await grade(case["input"], output, case["criteria"])
        total_score += result.get("score", 0)
        print_grade(i, case, output, result)

    avg = total_score / len(TEST_CASES)
    print(f"\n📊 Átlagpontszám: {avg:.1f}/10")
    return avg


async def main():
    print("🧪 Modellek összehasonlítása indul...\n")
    print(f"📋 Tesztesetek száma: {len(TEST_CASES)}")
    print(f"⚖️  Bíró modell: {GRADER_MODEL}")

    # Minden modell kiértékelése sorban
    results = {}
    for model in MODELS:
        avg = await eval_model(model)
        results[model] = avg

    # Összefoglaló táblázat az összes modell átlagpontszámával
    print("\n" + "=" * 60)
    print("📈 ÖSSZEFOGLALÓ")
    print("=" * 60)
    for model, avg in results.items():
        filled = int(avg)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"{model.split('.')[-1]:<45} [{bar}] {avg:.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
