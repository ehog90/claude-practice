"""Image prompt példa: Claude-nak képet adunk a kérdés mellé, és ő a képről válaszol (vision).

Egy üzenet content-je blokkok listája; a kép egy külön `image` blokk, a kérdés egy `text` blokk:

    {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}},
        {"type": "text", "text": "Mi látható a képen?"},
    ]}

Jó tudni:
  - A képet a kérdés ELÉ tegyük: így jobb a válasz minőség.
  - Több kép is küldhető; ilyenkor érdemes minden kép elé címkét írni ("Kép 1:", "Kép 2:"), hogy a kérdésben hivatkozni lehessen rájuk.
  - Támogatott formátumok: JPEG, PNG, GIF, WebP. A Claude API-n egy kép legfeljebb 5 MB lehet.
  - A kép tokenekbe kerül: nagyjából (szélesség x magasság) / 750 token. A nagyon nagy képeket a rendszer úgyis
    lekicsinyíti (a hosszabbik oldal kb. 1568 pixelre), ezért a felesleges méret csak lassít és drágít.
  - Az URL-ből töltött kép is megadható `{"type": "url", "url": "..."}` forrással base64 helyett.

Indítás:
    python image_prompt.py                           # a program által rajzolt minta oszlopdiagram + alapkérdés
    python image_prompt.py kep.jpg                   # alapkérdés: JSON a képen olvasható szöveggel és a 3 domináns színnel
    python image_prompt.py kep.jpg "Mi van a képen?" # saját kérdés: szabad szöveges válasz
    python image_prompt.py --compare                 # két minta diagram összehasonlítása (több kép egy kérésben)
"""

import argparse
import base64
import json
import os
import re
import struct
import sys
import zlib
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]

HERE = Path(__file__).resolve().parent
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # a Claude API korlátja képenként


# ---------------------------------------------------------------------------
# Minta kép: egyszerű oszlopdiagram, külső csomag nélkül (PNG-t kézzel írunk)
# ---------------------------------------------------------------------------


def make_sample_image(path: Path, heights: list[int], colors: list[tuple[int, int, int]]):
    """Fehér háttéren, szürke tengelyek mellett oszlopokat rajzol, és PNG-ként elmenti. Az oszlopok magassága pixel."""
    width, height = 400, 300
    pixels = [[(255, 255, 255)] * width for _ in range(height)]
    baseline = height - 30  # az alsó tengely sora
    for y in range(20, baseline + 1):  # függőleges tengely
        pixels[y][40] = (120, 120, 120)
    for x in range(40, width - 20):  # vízszintes tengely
        pixels[baseline][x] = (120, 120, 120)
    bar_width, gap = 50, 25
    for n, (h, color) in enumerate(zip(heights, colors)):
        left = 60 + n * (bar_width + gap)
        for y in range(baseline - h, baseline):
            for x in range(left, left + bar_width):
                pixels[y][x] = color
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)  # soronként 0-s szűrőbájt

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    path.write_bytes(png + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


RED, BLUE, GREEN, ORANGE = (220, 50, 47), (38, 105, 220), (46, 160, 67), (245, 150, 30)


def ensure_samples():
    make_sample_image(HERE / "minta_diagram_1.png", [90, 210, 150, 50], [RED, BLUE, GREEN, ORANGE])
    make_sample_image(HERE / "minta_diagram_2.png", [90, 120, 150, 200], [RED, BLUE, GREEN, ORANGE])


# ---------------------------------------------------------------------------
# A kérés összeállítása és elküldése
# ---------------------------------------------------------------------------


def image_block(path: Path) -> dict:
    """Egy képfájlból image blokkot készít (base64)."""
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if not media_type:
        sys.exit(f"Nem támogatott formátum: {path.suffix}. Támogatott: {', '.join(MEDIA_TYPES)}")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        sys.exit(f"A kép túl nagy ({len(data) / 1e6:.1f} MB), legfeljebb 5 MB lehet. Kicsinyítsd le.")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode()}}


# A kért JSON szerkezete: a képen olvasható szöveg + a három domináns szín.
#
# A JSON-t egy "tool" bemenetén keresztül kérjük vissza: a modell egy report_image_info nevű toolt hív, és a tool
# input_schema-ja ez a séma. A tool bemenete már kész, parse-olt dict, nem kell JSON-t kiszedegetni a szövegből.
# (Közvetlen Anthropic API-n erre az `output_config={"format": {"type": "json_schema", ...}}` is jó. Ezen a végponton viszont
# az 400-as hibát adott: kikényszerített tool_choice-ra fordítódik le, amit ez a modell nem fogad el. A tool + `auto`
# tool_choice mindenhol működik.)
# A "pontosan 3 elem" és a hex formátum a sémában nem szabályozható, ezért a válaszban ellenőrizzük (lásd validate_image_info).
IMAGE_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "A képen olvasható szöveg, egy mondatban, üres sor nélkül. Ha nincs szöveg, üres string."},
        "dominant_colors": {
            "type": "array",
            "description": "A kép három domináns színe, a legjellemzőbbtől a kevésbé jellemzőig.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A szín magyar neve, pl. 'sötétzöld'."},
                    "hex": {"type": "string", "description": "A szín hex kódja '#RRGGBB' formában."},
                },
                "required": ["name", "hex"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["text", "dominant_colors"],
    "additionalProperties": False,
}


def validate_image_info(info: dict) -> dict:
    """A sémával nem kifejezhető szabályok ellenőrzése: pontosan 3 szín, érvényes #RRGGBB hex kódokkal."""
    colors = info["dominant_colors"]
    if len(colors) != 3:
        raise ValueError(f"3 domináns színt kértünk, {len(colors)} jött: {colors}")
    for color in colors:
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color["hex"]):
            raise ValueError(f"Érvénytelen hex kód: {color['hex']}")
    return info


def ask_about_images(paths: list[Path], question: str, schema: dict | None = None) -> str | dict:
    """Kép(ek) + kérdés. Séma megadásakor a válasz a sémának megfelelő dict, különben szabad szöveg."""
    content = []
    for n, path in enumerate(paths, start=1):
        if len(paths) > 1:
            content.append({"type": "text", "text": f"Kép {n}:"})  # címke, hogy a kérdésben hivatkozni lehessen rá
        content.append(image_block(path))
    content.append({"type": "text", "text": question})  # a kérdés a képek UTÁN jön

    extra = {}
    if schema:
        extra = {
            "tools": [{
                "name": "report_image_info",
                "description": "A képről kinyert adatok rögzítése. A kérdésre MINDIG ennek a toolnak a meghívásával válaszolj.",
                "input_schema": schema,
            }],
            "tool_choice": {"type": "auto"},  # kikényszerítés helyett a promptból és a tool leírásából tudja, hogy hívnia kell
        }
        content.append({"type": "text", "text": "Válaszolj a report_image_info tool meghívásával."})
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system="Válaszolj magyarul, röviden, markup nélkül, mert konzolban jelenik meg.",
        messages=[{"role": "user", "content": content}],
        **extra,
    )
    print(f"   (token használat: {response.usage.input_tokens} bemenet, {response.usage.output_tokens} kimenet)")
    if not schema:
        return "".join(b.text for b in response.content if b.type == "text")

    if response.stop_reason in ("max_tokens", "refusal"):  # ilyenkor a tool bemenete hiányos vagy hiányzik
        raise RuntimeError(f"A válasz nem teljes (stop_reason={response.stop_reason}).")
    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise RuntimeError("A modell nem hívta meg a toolt, szöveggel válaszolt: " + "".join(b.text for b in response.content if b.type == "text"))
    return validate_image_info(tool_use.input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kép + kérdés Claude-nak")
    parser.add_argument("image", nargs="?", help="a kép útvonala (alapértelmezett: a minta diagram)")
    parser.add_argument("question", nargs="?", help="a kérdés a képről")
    parser.add_argument("--compare", action="store_true", help="a két minta diagram összehasonlítása")
    args = parser.parse_args()

    ensure_samples()
    schema = None  # séma nélkül szabad szöveges a válasz
    if args.compare:
        paths = [HERE / "minta_diagram_1.png", HERE / "minta_diagram_2.png"]
        question = "Hasonlítsd össze a két diagramot: melyik oszlop változott, merre és körülbelül mennyit?"
    else:
        paths = [Path(args.image)] if args.image else [HERE / "minta_diagram_1.png"]
        if args.question:
            question = args.question
        else:
            # alapértelmezett: JSON a képen olvasható szöveggel és a három domináns színnel
            question = ("Mi van a képre írva? Add vissza a képen olvasható szöveget egy egész mondatban, felesleges szöveg és üres sor nélkül, "
                        "valamint a kép három domináns színét (név és hex kód).")
            schema = IMAGE_INFO_SCHEMA

    print(f"Képek: {', '.join(p.name for p in paths)}\nKérdés: {question}\n")
    result = ask_about_images(paths, question, schema)
    print("🤖 " + (json.dumps(result, ensure_ascii=False, indent=2) if schema else result))
