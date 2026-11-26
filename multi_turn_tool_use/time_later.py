"""Interaktív multi-turn tool use példa: "Mennyi idő lesz N egység múlva?"

A modell két egymásra épülő toolt használ, ezért több körre van szükség:
  1. get_current_datetime  -> megtudja, mennyi az idő most
  2. add_time              -> hozzáadja az N egységet az előző eredményhez
A második hívás bemenete az első eredménye, ezért nem kérhető egyszerre: ez a multi-turn.
"""

import json
import os
from datetime import datetime, timedelta

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]

# ---------------------------------------------------------------------------
# Toolok: függvények
# ---------------------------------------------------------------------------

UNITS = ["seconds", "minutes", "hours", "days", "weeks"]


def get_current_datetime() -> str:
    """Az aktuális helyi dátum és idő ISO formátumban, pl. '2026-11-26T13:25:00'."""
    return datetime.now().isoformat(timespec="seconds")


def add_time(datetime_iso: str, amount: float, unit: str) -> str:
    """Hozzáad `amount` darab `unit` egységet a megadott időponthoz, az eredmény ISO formátumú."""
    if unit not in UNITS:
        raise ValueError(f"Ismeretlen egység: {unit}. Lehetséges: {', '.join(UNITS)}")
    start = datetime.fromisoformat(datetime_iso)
    return (start + timedelta(**{unit: amount})).isoformat(timespec="seconds")


TOOL_FUNCTIONS = {"get_current_datetime": get_current_datetime, "add_time": add_time}

# ---------------------------------------------------------------------------
# Toolok: sémák
# ---------------------------------------------------------------------------

tools = [
    {
        "name": "get_current_datetime",
        "description": "Visszaadja az aktuális helyi dátumot és időt ISO formátumban (YYYY-MM-DDTHH:MM:SS).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_time",
        "description": (
            "Hozzáad egy időtartamot egy megadott időponthoz, és visszaadja az új időpontot "
            "ISO formátumban. Az időpontot (datetime_iso) a get_current_datetime-tól kapod."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "datetime_iso": {
                    "type": "string",
                    "description": "Kiindulási időpont ISO formátumban, pl. '2026-11-26T13:25:00'.",
                },
                "amount": {
                    "type": "number",
                    "description": "Hány egységet adjunk hozzá (lehet negatív is, a múltba lépéshez).",
                },
                "unit": {
                    "type": "string",
                    "enum": UNITS,
                    "description": "Az időegység.",
                },
            },
            "required": ["datetime_iso", "amount", "unit"],
        },
    },
]

# ---------------------------------------------------------------------------
# Multi-turn ciklus
# ---------------------------------------------------------------------------


def run_tool(block) -> dict:
    """Egy tool_use blokkból tool_result blokkot készít."""
    try:
        result = TOOL_FUNCTIONS[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": block.id, "content": f"Hiba: {e}", "is_error": True}


def pause(step_mode: bool):
    """Lépésenkénti módban megvárja az Enter-t, hogy nyugodtan végig lehessen nézni a lépéseket."""
    if step_mode:
        input("   ⏎ Enter a folytatáshoz...")


def ask(messages: list, question: str, step_mode: bool) -> str:
    """Egy kérdést megválaszol; a `messages` bővül, így a következő kérdésnél a modell emlékszik."""
    messages.append({"role": "user", "content": question})
    step = 0

    for _ in range(10):  # biztonsági korlát: legfeljebb 10 modellhívás kérdésenként
        step += 1
        print(f"\n── {step}. lépés: modell hívása ──")
        response = client.messages.create(model=model, max_tokens=500, messages=messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        # A modell esetleges szöveges gondolatmenete a tool hívás előtt
        for block in response.content:
            if block.type == "text" and block.text.strip() and response.stop_reason == "tool_use":
                print(f"   💬 A modell: {block.text.strip()}")

        # Ha a modell már nem kér toolt, kész a válasz.
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Különben lefuttatjuk az összes kért toolt, és egy user üzenetben visszaküldjük.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"   🔧 Tool kérés:  {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                pause(step_mode)
                result = run_tool(block)
                icon = "❌" if result.get("is_error") else "📥"
                print(f"   {icon} tool_result: {result['content']}")
                tool_results.append(result)
        messages.append({"role": "user", "content": tool_results})
        pause(step_mode)

    return "[Túl sok lépés, megszakítva.]"


if __name__ == "__main__":
    print("Kérdezz időpontokról, pl. 'Mennyi idő lesz 90 perc múlva?' vagy 'És 2 nap múlva?'")
    print("'exit' a kilépéshez.\n")
    step_mode = input("Lépésenként (Enter-rel továbblépve)? [i/N]: ").strip().lower() in ("i", "igen", "y")

    messages = []  # a teljes beszélgetés; így a követő kérdések ("és 2 nap múlva?") is működnek
    while True:
        question = input("\n👤 Te: ").strip()
        if question.lower() == "exit":
            print("👋 Viszlát!")
            break
        if not question:
            continue
        answer = ask(messages, question, step_mode)
        print(f"\n🤖 Válasz: {answer}")
