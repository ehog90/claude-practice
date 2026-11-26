"""Interaktív chat, amiben Claude használhatja a toolokat (időjárás, dátum/idő).

Működés dióhányban:
  1. A felhasználó üzenetét elküldjük a modellnek a tool sémákkal együtt.
  2. Ha a modell tool-t akar hívni, a stop_reason "tool_use" lesz.
  3. Lefuttatjuk a kért toolokat, az eredményt visszaküldjük user üzenetként.
  4. Ismételjük, amíg a modell már nem kér toolt: ekkor jön a végső szöveges válasz.
"""

import asyncio
import os

from dotenv import load_dotenv
import anthropic

from tools import tools, run_tool_uses

# Betölti a .env fájlt (ANTHROPIC_API_KEY, ANTHROPIC_MODEL) a környezeti változókba.
load_dotenv()

# Aszinkron kliens; az API kulcsot automatikusan a környezetből veszi.
client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = "Válaszolj tömören. Ne formázd markuppal, konzolból nyomjuk."
# Biztonsági korlát: ha a modell valamiért végtelenül hívogatna toolokat, ne ragadjunk be.
MAX_TOOL_ROUNDS = 10


async def chat_with_tools(messages, effort="medium"):
    """Hívja a modellt, és amíg tool_use-t kér, lefuttatja a toolokat. Visszaadja a végső szöveget.

    A `messages` listát helyben bővíti, így a beszélgetés előzménye megmarad a hívások között.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.messages.create(
            model=os.environ["ANTHROPIC_MODEL"],  # a modell a .env-ből jön
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,  # ettől tud a modell a toolokról
            output_config={"effort": effort},  # mennyit "gondolkodjon" a modell
        )
        # A teljes content-et (tool_use blokkokkal együtt) vissza kell tenni az előzménybe,
        # különben a következő hívásban a tool_result-nak nem lenne párja, és az API hibát dobna.
        messages.append({"role": "assistant", "content": response.content})

        # Ha nem tool_use a leállás oka (pl. end_turn), a modell végzett: kinyerjük a szöveget.
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text") or (
                f"[Nincs szöveges válasz (stop_reason={response.stop_reason})]"
            )

        # Csak tájékoztatásul: kiírjuk, milyen toolt hív a modell és milyen inputtal.
        for block in response.content:
            if block.type == "tool_use":
                print(f"🔧 {block.name}({block.input})")
        # A tool eredményeket user szerepkörű üzenetként küldjük vissza (az API ezt várja),
        # majd a ciklus újra meghívja a modellt, hogy az eredmények alapján válaszoljon.
        messages.append({"role": "user", "content": run_tool_uses(response.content)})

    return "[Túl sok tool kör, megszakítva.]"


async def main():
    # Effort szint bekérése; érvénytelen vagy üres bemenetnél "medium".
    effort = input("⚡ Effort (low/medium/high/xhigh/max, alapért. medium): ").strip().lower()
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        effort = "medium"

    print("🤖 Tool chat kész! 'exit' a kilépéshez.\n")
    messages = []  # a beszélgetés teljes előzménye; így a modell emlékszik a korábbiakra
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        if not user_input:  # üres Enter-t figyelmen kívül hagyjuk
            continue
        messages.append({"role": "user", "content": user_input})
        print(f"🤖 Assistant: {await chat_with_tools(messages, effort)}\n")


# Csak közvetlen indításnál (python tool_chat.py) fut le.
if __name__ == "__main__":
    asyncio.run(main())
