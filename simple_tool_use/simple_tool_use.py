"""A legegyszerűbb tool use + tool_result példa: egy tool, két API hívás, semmi más."""

import os
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()  # beolvassa az ANTHROPIC_API_KEY-t és az ANTHROPIC_MODEL-t a .env-ből
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]

# 1. A tool sémája: ebből tudja a modell, hogy létezik, és hogyan kell hívni.
tools = [{
    "name": "get_current_time",
    "description": "Visszaadja az aktuális helyi időt (óra:perc).",
    "input_schema": {"type": "object", "properties": {}},  # nincs paramétere
}]

messages = [{"role": "user", "content": "Hány óra van?"}]

# 2. Első hívás: a modell látja a toolt, és kérni fogja a futtatását (stop_reason = "tool_use").
response = client.messages.create(model=model, max_tokens=500, messages=messages, tools=tools)
tool_use = next(b for b in response.content if b.type == "tool_use")
print("A modell ezt kéri:", tool_use.name, tool_use.input)

# 3. Mi futtatjuk le a függvényt (a modell csak kérte).
result = datetime.now().strftime("%H:%M")

# 4. Az előzményhez hozzáadjuk a modell kérését (assistant) és az eredményt (user, tool_result).
messages.append({"role": "assistant", "content": response.content})
messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": tool_use.id,  # ez köti az eredményt a kéréshez
        "content": result,           # az eredmény mindig string
    }],
})

# 5. Második hívás: a modell az eredmény alapján megfogalmazza a végső választ.
final = client.messages.create(model=model, max_tokens=500, messages=messages, tools=tools)
print("Válasz:", final.content[0].text)
