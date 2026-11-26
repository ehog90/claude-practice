"""Text editor tool példa: Claude fájlokat néz meg és szerkeszt egy `workspace/` mappában.

Az Anthropic beépített ("Anthropic-defined") text editor toolját használjuk. Ennek nincs
saját input_schema-ja: a modell ismeri a parancsait, de a végrehajtás a mi dolgunk (kliensoldali tool).

    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}      (indítás: --builtin)

Az alapértelmezett mód ugyanezt a toolt egyedi (saját sémás) tool-ként adja meg, azonos névvel és
parancsokkal, mert egyes végpontok a beépített típust nem alakítják valódi tool hívássá (lásd CUSTOM_TOOL).

A modell a `tool_use.input.command` mezőben négyféle parancsot kérhet:

    view         path, [view_range]            fájl tartalma sorszámokkal / mappa listázása
    create       path, file_text               fájl létrehozása (vagy felülírása)
    str_replace  path, old_str, new_str        pontosan egy előfordulás cseréje
    insert       path, insert_line, insert_text  szöveg beszúrása a megadott sor UTÁN (0 = a fájl elejére)

Hiba esetén tool_result-ot küldünk is_error=True-val, hogy a modell javítani tudjon.

FIGYELEM: a modell által adott `path` nem megbízható adat. Minden útvonalat a WORKSPACE
mappára korlátozunk (lásd safe_path), így a modell nem tud kiszökni belőle.
"""

import os
import shutil
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]

WORKSPACE = (Path(__file__).resolve().parent / "workspace").resolve()
MAX_VIEW_CHARS = 10_000  # a view kimenetének felső korlátja (a toolnak is megadjuk max_characters-ként)

# 1) Az Anthropic beépített text editor tool definíciója: nincs séma, a modell "tudja" a parancsokat.
BUILTIN_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": MAX_VIEW_CHARS}

# 2) Ugyanaz a tool egyedi tool-ként, saját sémával. Ugyanazt a nevet és ugyanazokat a parancsokat használja,
#    ezért a végrehajtó kód (run_editor_tool) változatlan. Erre azért van szükség, mert a beépített definíciót
#    nem minden végpont alakítja valódi tool hívássá: ezen a végponton (us.anthropic.* modellazonosító) a modell
#    a hívást sima szövegként írta ki (`<invoke name=...>`), tool_use blokk nélkül. Az egyedi definíció működik.
CUSTOM_TOOL = {
    "name": "str_replace_based_edit_tool",
    "description": (
        "Fájlok megtekintése és szerkesztése a munkakönyvtárban. Parancsok: "
        "view (fájl tartalma sorszámokkal vagy mappa listázása; a view_range [kezdő, záró] sorokat ad, a záró -1 a fájl vége), "
        "create (fájl létrehozása vagy felülírása a file_text tartalommal), "
        "str_replace (az old_str PONTOSAN egy előfordulását cseréli new_str-re; hiba, ha 0 vagy több találat van), "
        "insert (az insert_text beszúrása az insert_line. sor UTÁN, 0 = a fájl elejére)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"], "description": "A végrehajtandó parancs."},
            "path": {"type": "string", "description": "A fájl vagy mappa útvonala a munkakönyvtárhoz képest, pl. 'szamolo.py'."},
            "file_text": {"type": "string", "description": "create: a fájl teljes tartalma."},
            "old_str": {"type": "string", "description": "str_replace: a lecserélendő szöveg (pontosan egyszer szerepelhet a fájlban)."},
            "new_str": {"type": "string", "description": "str_replace: az új szöveg."},
            "insert_line": {"type": "integer", "description": "insert: ennyiedik sor UTÁN szúr be (0 = a fájl elejére)."},
            "insert_text": {"type": "string", "description": "insert: a beszúrandó szöveg."},
            "view_range": {"type": "array", "items": {"type": "integer"}, "description": "view: [kezdő sor, záró sor], a záró -1 a fájl vége."},
        },
        "required": ["command", "path"],
    },
}

# Alapból az egyedi változatot használjuk (ez működik minden végponton); `--builtin` kapcsolóval a beépítettet.
TOOLS = [BUILTIN_TOOL if "--builtin" in sys.argv else CUSTOM_TOOL]

SYSTEM_PROMPT = (
    "Egy kis programozó asszisztens vagy, aki a munkakönyvtárban lévő fájlokat nézheti és szerkesztheti "
    "a text editor tooljával. Az útvonalak a munkakönyvtárhoz képest relatívak (pl. 'szamolo.py', nem abszolút). "
    "Szerkesztés előtt nézd meg a fájlt. Kis, célzott módosításokhoz a str_replace-t használd. "
    "Válaszolj magyarul, röviden, markup nélkül, mert konzolban jelenik meg."
)

# Szín csak valódi terminálban (NO_COLOR-t tiszteletben tartva)
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def color(text: str, code: int) -> str:
    return f"\033[38;5;{code}m{text}\033[0m" if USE_COLOR else text


# ---------------------------------------------------------------------------
# A tool végrehajtása
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """A modellnek visszaküldendő hibaüzenet (tool_result is_error=True)."""


def safe_path(raw: str) -> Path:
    """A modell által adott útvonalat a WORKSPACE-re korlátozza. Kiszökési kísérletnél ToolError."""
    path = Path(raw)
    full = (path if path.is_absolute() else WORKSPACE / path).resolve()  # a resolve() feloldja a '..'-t és a symlinkeket
    if full != WORKSPACE and not full.is_relative_to(WORKSPACE):
        raise ToolError(f"Hozzáférés megtagadva: az útvonal a munkakönyvtáron kívülre mutat ({raw}).")
    return full


def cmd_view(path: Path, view_range=None, **_) -> str:
    if not path.exists():
        raise ToolError(f"Nem található: {path.relative_to(WORKSPACE) if path != WORKSPACE else '.'}")
    if path.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return "\n".join(entries) or "(üres mappa)"
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = 1, len(lines)
    if view_range:  # [kezdő sor, záró sor], a -1 a fájl végét jelenti (1-től számolva)
        start = view_range[0]
        end = len(lines) if view_range[1] == -1 else view_range[1]
        if not 1 <= start <= len(lines) or end < start:
            raise ToolError(f"Érvénytelen view_range {view_range}: a fájl {len(lines)} soros.")
    numbered = "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))
    return numbered[:MAX_VIEW_CHARS] + ("\n... (levágva)" if len(numbered) > MAX_VIEW_CHARS else "")


def cmd_create(path: Path, file_text: str = None, **_) -> str:
    if file_text is None:
        raise ToolError("A create parancshoz kell a file_text.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # ajánlott: felülírás előtt mentés
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    path.write_text(file_text, encoding="utf-8")
    return f"A fájl sikeresen létrehozva: {path.relative_to(WORKSPACE)}"


def cmd_str_replace(path: Path, old_str: str = None, new_str: str = "", **_) -> str:
    if not path.is_file():
        raise ToolError(f"Nem található fájl: {path.relative_to(WORKSPACE)}")
    if not old_str:
        raise ToolError("A str_replace parancshoz kell az old_str.")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        raise ToolError("Nincs találat a cserélendő szövegre. Ellenőrizd a szöveget (szóközök, sortörések), és próbáld újra.")
    if count > 1:
        raise ToolError(f"{count} találat van a cserélendő szövegre. Adj meg több környezetet, hogy egyértelmű legyen.")
    path.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return "A szöveg sikeresen kicserélve, pontosan egy helyen."


def cmd_insert(path: Path, insert_line: int = None, insert_text: str = None, **_) -> str:
    if not path.is_file():
        raise ToolError(f"Nem található fájl: {path.relative_to(WORKSPACE)}")
    if insert_line is None or insert_text is None:
        raise ToolError("Az insert parancshoz kell az insert_line és az insert_text.")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not 0 <= insert_line <= len(lines):
        raise ToolError(f"Érvénytelen insert_line: {insert_line}. A fájl {len(lines)} soros (0 = a fájl elejére).")
    if lines and insert_line == len(lines) and not lines[-1].endswith("\n"):
        lines[-1] += "\n"  # a fájl végére szúrunk: az utolsó sort le kell zárni
    lines[insert_line:insert_line] = [insert_text if insert_text.endswith("\n") else insert_text + "\n"]
    path.write_text("".join(lines), encoding="utf-8")
    return f"A szöveg sikeresen beszúrva a(z) {insert_line}. sor után."


COMMANDS = {"view": cmd_view, "create": cmd_create, "str_replace": cmd_str_replace, "insert": cmd_insert}


def run_editor_tool(tool_input: dict) -> str:
    """Egy text editor parancs végrehajtása; ToolError-t dob, ha a modellnek hibát kell jelezni."""
    command = tool_input.get("command")
    if command not in COMMANDS:
        raise ToolError(f"Ismeretlen parancs: {command}. Lehetséges: {', '.join(COMMANDS)}")
    if "path" not in tool_input:
        raise ToolError("Hiányzik a path.")
    args = {k: v for k, v in tool_input.items() if k not in ("command", "path")}
    try:
        return COMMANDS[command](safe_path(tool_input["path"]), **args)
    except (OSError, UnicodeDecodeError) as e:  # pl. bináris fájl, jogosultság
        raise ToolError(f"Fájlhiba: {e}")


# ---------------------------------------------------------------------------
# Megjelenítés
# ---------------------------------------------------------------------------


def show_call(tool_input: dict):
    """A modell kérésének kiírása; str_replace-nél kis diff (piros = régi, zöld = új)."""
    cmd, path = tool_input.get("command"), tool_input.get("path")
    print(f"   📝 {color(cmd, 117)} {color(path, 250)}")
    if cmd == "str_replace":
        for line in (tool_input.get("old_str") or "").splitlines():
            print("      " + color(f"- {line}", 203))
        for line in (tool_input.get("new_str") or "").splitlines():
            print("      " + color(f"+ {line}", 114))
    elif cmd == "insert":
        for line in (tool_input.get("insert_text") or "").splitlines():
            print("      " + color(f"+ {line}", 114))
    elif cmd == "create":
        n = len((tool_input.get("file_text") or "").splitlines())
        print("      " + color(f"({n} soros új fájl)", 244))


# ---------------------------------------------------------------------------
# Chat ciklus: addig megy, amíg a modell szerkeszteni/nézni akar
# ---------------------------------------------------------------------------


def chat(messages: list, question: str) -> str:
    messages.append({"role": "user", "content": question})
    while True:
        response = client.messages.create(
            model=model, max_tokens=8000, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            if response.stop_reason == "max_tokens":
                print(color("   ⚠️  A válasz elérte a max_tokens korlátot.", 208))
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            show_call(block.input)
            try:
                output, is_error = run_editor_tool(block.input), False
            except ToolError as e:
                output, is_error = str(e), True
                print("      " + color(f"❌ {output}", 203))
            result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
            if is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})


SAMPLE = '''def atlag(szamok):
    return sum(szamok) / (len(szamok) + 1)


print(atlag([2, 4, 6]))
'''


def ensure_workspace():
    """Első indításkor létrehoz egy kis hibás mintafájlt, amin ki lehet próbálni a szerkesztést."""
    if not WORKSPACE.exists():
        WORKSPACE.mkdir()
        (WORKSPACE / "szamolo.py").write_text(SAMPLE, encoding="utf-8")


if __name__ == "__main__":
    ensure_workspace()
    print(f"Munkakönyvtár: {WORKSPACE}")
    print("Próbáld: 'Mi van a workspace-ben?', 'Javítsd ki a szamolo.py hibáját', 'Hozz létre egy teszt.py-t'. 'exit' a kilépéshez.\n")
    messages = []
    while True:
        question = input("👤 Te: ").strip()
        if question.lower() == "exit":
            print("👋 Viszlát!")
            break
        if question:
            print(f"\n🤖 {chat(messages, question)}\n")
