"""Az összes tool egy helyen: függvények + JSON sémák + végrehajtó (dispatcher).

Három része van:
  1. A tool függvények (get_weather, get_current_datetime, calculator).
  2. A JSON sémák, amiket a modellnek küldünk. A séma mondja meg, hogy létezik-e a tool,
     mire való, és milyen paraméterekkel hívható. A modell nem futtat kódot, csak
     egy tool_use blokkot ad vissza (név + input), a hívást mi végezzük el.
  3. A "híd": run_tool / run_tool_uses, ami a modell kérését lefuttatja, és az eredményt
     tool_result formában adja vissza.

Új tool hozzáadása: írj egy függvényt, egy sémát, és vedd fel a TOOLS listába.
"""

import ast
import json
import operator
import urllib.parse
import urllib.request
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. Tool függvények
# ---------------------------------------------------------------------------

# Az Open-Meteo előrejelző végpontja. Ingyenes, regisztráció nélkül használható.
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(lat: float, lon: float, timeout: float = 10.0) -> dict:
    """Visszaadja az aktuális időjárást a megadott koordinátákhoz (Open-Meteo API).

    Példa visszatérési érték:
        {"temperature": 12.3, "windspeed": 5.4, "winddirection": 210,
         "weathercode": 3, "is_day": 1, "time": "2026-11-26T10:00",
         "units": {"temperature": "°C", "windspeed": "km/h"}}

    A weathercode WMO kód: pl. 0 = derült, 3 = borult, 61 = gyenge eső, 95 = zivatar.
    """
    # Bemenet ellenőrzése: a szélesség -90..90, a hosszúság -180..180 fok lehet.
    # Így rossz adatnál nem megy fölösleges hálózati hívás, és érthető a hibaüzenet.
    if not -90 <= lat <= 90:
        raise ValueError(f"Érvénytelen szélesség: {lat}")
    if not -180 <= lon <= 180:
        raise ValueError(f"Érvénytelen hosszúság: {lon}")

    # A query string összeállítása (urlencode gondoskodik a megfelelő kódolásról).
    # A current_weather=true az aktuális mérést kéri, nem az órás előrejelzést.
    params = urllib.parse.urlencode(
        {"latitude": lat, "longitude": lon, "current_weather": "true"}
    )

    # HTTP GET kérés. A timeout megakadályozza, hogy a program örökre ráragadjon,
    # ha a szerver nem válaszol. A `with` lezárja a kapcsolatot.
    with urllib.request.urlopen(f"{OPEN_METEO_URL}?{params}", timeout=timeout) as resp:
        data = json.load(resp)

    # Az API válaszából csak a hasznos részt tartjuk meg, egyszerű dict-be rendezve.
    current = data["current_weather"]
    units = data.get("current_weather_units", {})  # pl. {"temperature": "°C", ...}
    return {
        "temperature": current["temperature"],
        "windspeed": current["windspeed"],
        "winddirection": current["winddirection"],  # fok: 0 = É, 90 = K, 180 = D, 270 = Ny
        "weathercode": current["weathercode"],
        "is_day": current["is_day"],  # 1 = nappal, 0 = éjszaka
        "time": current["time"],
        "units": {
            "temperature": units.get("temperature", "°C"),
            "windspeed": units.get("windspeed", "km/h"),
        },
    }


def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Visszaadja az aktuális helyi dátumot/időt a megadott strftime formátumban.

    Gyakori formátumkódok: %Y év, %m hónap, %d nap, %H óra (24h), %M perc, %S másodperc,
    %A a nap neve.

    Példák:
        get_current_datetime()             -> "2026-11-26 11:45:03"
        get_current_datetime("%Y.%m.%d.")  -> "2026.11.26."
        get_current_datetime("%H:%M")      -> "11:45"
    """
    # Üres formátumnál a strftime üres stringet adna, ami félrevezető lenne,
    # ezért inkább hibát jelzünk.
    if not format:
        raise ValueError("A formátum nem lehet üres.")
    # datetime.now() a gép helyi idejét adja (nem UTC-t), a strftime formázza szöveggé.
    return datetime.now().strftime(format)


# A kalkulátor által engedélyezett műveletek. Az eval() helyett az ast modullal
# járjuk be a kifejezést, és csak ezeket engedjük, így nem futtatható tetszőleges kód.
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 1000  # a túl nagy hatvány (pl. 9**9**9) lefagyasztaná a programot


def _eval_node(node):
    """Egy kifejezésfa csomópontját értékeli ki rekurzívan, csak engedélyezett elemekkel."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    # Szám literál (a bool-t kizárjuk, mert az int alosztálya)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Túl nagy kitevő (max {_MAX_EXPONENT}).")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Nem támogatott elem a kifejezésben.")


def calculator(expression: str) -> float | int:
    """Kiértékel egy matematikai kifejezést.

    Támogatott: + - * / // % ** , zárójelek, egyoperandusú +/-, tizedes számok.

    Példák:
        calculator("2 + 3 * 4")      -> 14
        calculator("(2 + 3) * 4")    -> 20
        calculator("10 / 4")         -> 2.5
        calculator("2 ** 10")        -> 1024
    """
    if not expression or not expression.strip():
        raise ValueError("A kifejezés nem lehet üres.")
    try:
        # mode="eval": egyetlen kifejezést vár (nem utasítást), a fát utána mi járjuk be.
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ValueError(f"Érvénytelen kifejezés: {expression}")
    return _eval_node(tree)


# ---------------------------------------------------------------------------
# 2. JSON sémák
# ---------------------------------------------------------------------------

# Az Anthropic API tool formátuma három kulcsból áll:
#   name         - a tool azonosítója (ezt kapjuk vissza a tool_use blokkban)
#   description  - szabad szöveg; minél pontosabb, annál jobban használja a modell
#   input_schema - JSON Schema a paraméterekről
get_weather_schema = {
    "name": "get_weather",
    "description": (
        "Lekéri az aktuális időjárást egy adott földrajzi koordinátapárhoz "
        "(hőmérséklet, szélsebesség, szélirány, weathercode, nappal/éjszaka)."
    ),
    "input_schema": {
        "type": "object",  # a paraméterek egy objektumban érkeznek
        "properties": {
            "lat": {
                "type": "number",
                "description": "Földrajzi szélesség fokban (-90 és 90 között).",
                "minimum": -90,
                "maximum": 90,
            },
            "lon": {
                "type": "number",
                "description": "Földrajzi hosszúság fokban (-180 és 180 között).",
                "minimum": -180,
                "maximum": 180,
            },
        },
        # A modellnek mindkettőt meg kell adnia, különben nem hívható a tool.
        "required": ["lat", "lon"],
    },
}

get_current_datetime_schema = {
    "name": "get_current_datetime",
    "description": (
        "Visszaadja az aktuális helyi dátumot és időt a megadott "
        "strftime formátumban."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": (
                    "strftime formátumstring, pl. '%Y-%m-%d %H:%M:%S' vagy "
                    "'%Y.%m.%d.'. Alapértelmezés: '%Y-%m-%d %H:%M:%S'."
                ),
                "default": "%Y-%m-%d %H:%M:%S",
                "minLength": 1,
            },
        },
        # Nincs kötelező paraméter: ha a modell kihagyja, a függvény alapértéke él.
        "required": [],
    },
}

calculator_schema = {
    "name": "calculator",
    "description": (
        "Kiértékel egy matematikai kifejezést pontosan. Használd minden számításhoz, "
        "ne fejben számolj. Támogatott: + - * / // % ** és zárójelek."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A kiszámolandó kifejezés, pl. '(2 + 3) * 4 / 5'.",
                "minLength": 1,
            },
        },
        "required": ["expression"],
    },
}

# Ezt a listát kell átadni az API-nak a `tools=` paraméterben.
tools = [get_weather_schema, get_current_datetime_schema, calculator_schema]

# ---------------------------------------------------------------------------
# 3. Végrehajtó (dispatcher)
# ---------------------------------------------------------------------------

# tool név -> Python függvény. A kulcs a sémából jön, így a két hely nem csúszhat szét
# (elgépelt név helyett a séma `name` mezőjét használjuk).
TOOL_FUNCTIONS = {
    get_weather_schema["name"]: get_weather,
    get_current_datetime_schema["name"]: get_current_datetime,
    calculator_schema["name"]: calculator,
}


def _execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Lefuttat egy tool-t; (tool_result szöveg, hiba volt-e) párt ad vissza. Nem dob kivételt."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        # A modell ritkán, de kérhet nem létező toolt; ezt is neki jelezzük vissza.
        return f"Hiba: ismeretlen tool: {name}", True
    try:
        # A tool_input dict kulcsai megegyeznek a függvény paraméterneveivel,
        # ezért ** kicsomagolással közvetlenül átadható.
        result = func(**tool_input)
    except Exception as e:
        # Ha kivétel keletkezik (pl. rossz koordináta, hálózati hiba), a program nem áll le:
        # a hibaüzenetet a modell megkapja, és tud rá reagálni (pl. javított adattal újrapróbál).
        return f"Hiba a(z) {name} futtatásakor: {e}", True
    # A tool_result tartalma string kell legyen: a dict-et JSON-ná alakítjuk.
    # ensure_ascii=False megőrzi az ékezeteket és a °C jelet.
    return (result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)), False


def run_tool(name: str, tool_input: dict) -> str:
    """Lefuttat egy tool-t név és input alapján; a tool_result content-nek megfelelő stringet ad."""
    return _execute_tool(name, tool_input)[0]


def run_tool_uses(content_blocks) -> list[dict]:
    """A modell válaszának tool_use blokkjaiból tool_result blokkokat készít.

    Bemenet: a modell tool_use blokkja, pl. (két párhuzamos hívással):
        ToolUseBlock(id="toolu_01A", name="calculator", input={"expression": "2340 * 17.5 / 100"})
        ToolUseBlock(id="toolu_01B", name="get_current_datetime", input={"format": "%H:%M"})

    Kimenet: egy tool_result blokk minden tool_use-hoz, ugyanabban a sorrendben:
        [
            {"type": "tool_result", "tool_use_id": "toolu_01A", "content": "409.5"},
            {"type": "tool_result", "tool_use_id": "toolu_01B", "content": "13:14"},
        ]

    Ezt a listát egyetlen user üzenet content-jeként kell visszaküldeni:
        {"role": "user", "content": [<tool_result blokkok>]}

    Tool eredmény példák (a "content" mező tartalma):
        get_weather          -> '{"temperature": 2.5, "windspeed": 12.6, "winddirection": 54,
                                 "weathercode": 61, "is_day": 1, "time": "2026-11-26T12:00",
                                 "units": {"temperature": "°C", "windspeed": "km/h"}}'
        get_current_datetime -> '2026.11.26.'
        calculator           -> '2.5'
    Hibás hívásnál a content a hibaszöveg, és a blokk "is_error": True mezőt is kap:
        {"type": "tool_result", "tool_use_id": "toolu_01C", "is_error": True,
         "content": "Hiba a(z) calculator futtatásakor: division by zero"}
    Az is_error jelzi a modellnek, hogy a hívás nem sikerült, így tud javítani és újrapróbálni.
    """
    results = []
    for block in content_blocks:
        if block.type != "tool_use":  # a text blokkokat kihagyjuk
            continue
        content, is_error = _execute_tool(block.name, block.input)
        result = {
            "type": "tool_result",
            # A tool_use_id köti össze az eredményt a modell kérésével.
            # Egy válaszban több tool_use is lehet (párhuzamos hívás), mindegyikre kell eredmény.
            "tool_use_id": block.id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        results.append(result)
    return results


# Gyors kézi teszt API hívás nélkül: python tools.py
if __name__ == "__main__":
    print(run_tool("get_current_datetime", {"format": "%Y.%m.%d."}))  # normál eset
    print(run_tool("get_weather", {"lat": 47.4979, "lon": 19.0402}))  # hálózati hívás
    print(run_tool("get_weather", {"lat": 999, "lon": 0}))  # hibás input -> hibaszöveg
    print(run_tool("nincs_ilyen", {}))  # ismeretlen tool -> hibaszöveg
    print(run_tool("calculator", {"expression": "(2 + 3) * 4 / 8"}))  # 2.5
    print(run_tool("calculator", {"expression": "1 / 0"}))  # nullával osztás -> hibaszöveg
    print(run_tool("calculator", {"expression": "__import__('os')"}))  # tiltott -> hibaszöveg
    print(run_tool("calculator", {"expression": "9 ** 9 ** 9"}))  # túl nagy kitevő -> hibaszöveg

    # tool_result példák: hamis tool_use blokkokból, API hívás nélkül
    from pprint import pprint
    from types import SimpleNamespace as Block

    fake_tool_uses = [
        Block(type="text", text="Kiszámolom."),  # text blokk: kimarad az eredményből
        Block(type="tool_use", id="toolu_01A", name="calculator", input={"expression": "2340 * 17.5 / 100"}),
        Block(type="tool_use", id="toolu_01B", name="get_current_datetime", input={"format": "%H:%M"}),
        Block(type="tool_use", id="toolu_01C", name="calculator", input={"expression": "1 / 0"}),
    ]
    print("\ntool_result blokkok:")
    pprint(run_tool_uses(fake_tool_uses), width=100)
