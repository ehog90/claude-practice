"""Tool loop példa: a modell addig hívja a toolokat, amíg már nincs rájuk szüksége.

A lényeg egyetlen ciklus:

    response = modell hívása
    while response.stop_reason == "tool_use":    # amíg toolt kér...
        toolok lefuttatása, tool_result visszaküldése
        response = modell hívása újra            # ...a modell dönti el, mikor elég
    # innen a response a végső szöveges válasz

Hogy hány kör lesz, azt nem mi döntjük el, hanem a modell: egy egyszerű kérdésre 1-2,
egy összetettebbre (pl. több időpont és hét napja) akár 5-6 tool hívás is kellhet.
"""

import csv
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]

# ---------------------------------------------------------------------------
# Toolok: függvények
# ---------------------------------------------------------------------------

UNITS = ["seconds", "minutes", "hours", "days", "weeks"]
WEEKDAYS = ["hétfő", "kedd", "szerda", "csütörtök", "péntek", "szombat", "vasárnap"]


def get_current_datetime() -> str:
    """Az aktuális helyi idő ISO formátumban."""
    return datetime.now().isoformat(timespec="seconds")


def add_time(datetime_iso: str, amount: float, unit: str) -> str:
    """Hozzáad `amount` darab `unit` egységet az időponthoz (negatív érték a múltba lép)."""
    if unit not in UNITS:
        raise ValueError(f"Ismeretlen egység: {unit}. Lehetséges: {', '.join(UNITS)}")
    return (datetime.fromisoformat(datetime_iso) + timedelta(**{unit: amount})).isoformat(timespec="seconds")


def get_weekday(datetime_iso: str) -> str:
    """Megmondja, a hét melyik napjára esik az időpont."""
    return WEEKDAYS[datetime.fromisoformat(datetime_iso).weekday()]


# Geocoding: OpenStreetMap Nominatim (ingyenes, API kulcs nélkül; forward és reverse is tudja).
# Használati szabály: legfeljebb 1 kérés/másodperc, és kötelező az azonosító User-Agent.
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = "tool-use-demo/1.0"


def _nominatim(endpoint: str, **params) -> dict | list:
    query = urllib.parse.urlencode({**params, "format": "jsonv2", "accept-language": "hu"})
    req = urllib.request.Request(f"{NOMINATIM_URL}/{endpoint}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def geocode(query: str, limit: int = 3) -> list[dict]:
    """Forward geocoding: helynév/cím -> koordináták. A legjobb találatok listáját adja."""
    if not query.strip():
        raise ValueError("A keresett hely nem lehet üres.")
    results = _nominatim("search", q=query, limit=limit)
    if not results:
        raise ValueError(f"Nincs találat erre: {query}")
    return [{"name": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in results]


def reverse_geocode(lat: float, lon: float) -> dict:
    """Reverse geocoding: koordináták -> a legközelebbi cím/hely."""
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Érvénytelen koordináta: {lat}, {lon}")
    result = _nominatim("reverse", lat=lat, lon=lon)
    if "error" in result:
        raise ValueError(f"Ezen a koordinátán nincs cím: {lat}, {lon}")
    return {"address": result["display_name"], "details": result.get("address", {})}


# Időjárás: Open-Meteo (ingyenes, API kulcs nélkül).
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(lat: float, lon: float) -> dict:
    """Az aktuális időjárás a megadott koordinátákhoz (hőmérséklet, szél, weathercode, nappal/éjszaka).

    A weathercode WMO kód: pl. 0 = derült, 3 = borult, 61 = gyenge eső, 95 = zivatar.
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Érvénytelen koordináta: {lat}, {lon}")
    query = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "current_weather": "true"})
    with urllib.request.urlopen(f"{OPEN_METEO_URL}?{query}", timeout=10) as resp:
        data = json.load(resp)
    current = data["current_weather"]
    units = data.get("current_weather_units", {})
    return {
        "temperature": current["temperature"],
        "windspeed": current["windspeed"],
        "winddirection": current["winddirection"],  # fok: 0 = É, 90 = K, 180 = D, 270 = Ny
        "weathercode": current["weathercode"],
        "is_day": current["is_day"],  # 1 = nappal, 0 = éjszaka
        "time": current["time"],
        "units": {"temperature": units.get("temperature", "°C"), "windspeed": units.get("windspeed", "km/h")},
    }


EARTH_RADIUS_KM = 6371.0088  # a Föld átlagos sugara


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    """Két koordináta közötti légvonalbeli (gömbi) távolság a haversine képlettel.

    Ez a nagyköri távolság a Föld felszínén, nem az útvonal hossza. A hiba gömbmodell
    miatt jellemzően kb. 0,5% alatti.
    """
    for lat, lon in ((lat1, lon1), (lat2, lon2)):
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Érvénytelen koordináta: {lat}, {lon}")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    # haversine: a = sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2),  d = 2R · asin(√a)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    km = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
    return {"kilometers": round(km, 2), "miles": round(km / 1.609344, 2)}


def destination_point(lat: float, lon: float, distance: float, bearing: float, unit: str = "km") -> dict:
    """Megadja azt a koordinátát, ahová egy pontból `distance` távolságra, `bearing` irányba jutunk.

    bearing fokban: 0 = észak, 90 = kelet, 180 = dél, 270 = nyugat (köztes irányok is lehetnek).
    A távolság a Föld felszínén, nagykör mentén értendő (gömbmodell, a calculate_distance inverze).
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Érvénytelen koordináta: {lat}, {lon}")
    if unit not in ("km", "miles"):
        raise ValueError(f"Ismeretlen egység: {unit}. Lehetséges: km, miles")
    if distance < 0:
        raise ValueError("A távolság nem lehet negatív (az ellenkező irányhoz használd a bearing-et).")
    km = distance * 1.609344 if unit == "miles" else distance

    delta = km / EARTH_RADIUS_KM  # szögtávolság radiánban
    theta = math.radians(bearing)
    phi1, lambda1 = math.radians(lat), math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    new_lon = (math.degrees(lambda2) + 540) % 360 - 180  # visszahozzuk a -180..180 tartományba
    return {"lat": round(math.degrees(phi2), 6), "lon": round(new_lon, 6)}


def get_timezone(lat: float, lon: float) -> dict:
    """Koordinátákhoz tartozó IANA időzóna neve (pl. 'Europe/Budapest').

    Az Open-Meteo `timezone=auto` beállítása adja vissza a koordinátához tartozó zónát.
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Érvénytelen koordináta: {lat}, {lon}")
    query = urllib.parse.urlencode(
        {"latitude": lat, "longitude": lon, "current_weather": "true", "timezone": "auto"}
    )
    with urllib.request.urlopen(f"{OPEN_METEO_URL}?{query}", timeout=10) as resp:
        data = json.load(resp)
    return {"timezone": data["timezone"], "abbreviation": data.get("timezone_abbreviation")}


def get_time_in_timezone(timezone: str, datetime_iso: str | None = None) -> dict:
    """Átváltja az időpontot a megadott IANA időzóna helyi idejére.

    A `datetime_iso` lehet időzónával (pl. '2026-11-26T12:00:00+00:00') vagy anélkül; az utóbbit
    a gép helyi idejének tekintjük (ahogy a get_current_datetime adja). Ha elmarad, a pillanatnyi idő.
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Ismeretlen időzóna: {timezone}. IANA nevet adj meg, pl. 'Europe/Budapest'.")
    # .astimezone() időzóna nélküli (naiv) időpontnál a gép helyi idejét feltételezi
    moment = datetime.fromisoformat(datetime_iso).astimezone() if datetime_iso else datetime.now().astimezone()
    local = moment.astimezone(tz)
    return {
        "datetime": local.isoformat(timespec="seconds"),
        "timezone": timezone,
        "abbreviation": local.tzname(),
        "utc_offset": local.strftime("%z"),
    }


# A CSV a py fájl mellé kerül, a futtatás helyétől függetlenül.
CSV_PATH = Path(__file__).resolve().parent / "weather_log.csv"
CSV_HEADER = [
    "🕒 Mentve", "🏙️ Település", "🌍 Ország", "📍 Szélesség", "📍 Hosszúság", "🌤️ Időjárás",
    "🌡️ Hőmérséklet (°C)", "💨 Szél (km/h)", "🧭 Szélirány (°)", "🌞 Napszak",
    "🌐 Idő (UTC)", "🕰️ Helyi idő", "🗺️ Időzóna",
]


def flag_emoji(country_code: str) -> str:
    """Kétbetűs országkódból (pl. 'hu') zászló emojit készít (🇭🇺): minden betű egy 'regionális jelző' karakter."""
    code = (country_code or "").upper()
    if len(code) != 2 or not code.isalpha():
        return ""
    return "".join(chr(0x1F1E6 + ord(ch) - ord("A")) for ch in code)


def country_cell(details: dict) -> str:
    """A CSV 'Ország' cellája: zászló + név, pl. '🇭🇺 Magyarország' (zászló nélkül, ha nincs országkód)."""
    return f"{flag_emoji(details.get('country_code', ''))} {details.get('country', '')}".strip()


def _append_csv(path: Path, header: list[str], rows: list[list]):
    """Sorokat fűz a CSV-hez; ha a fájl még nincs, létrehozza és kiírja a fejlécet."""
    is_new = not path.exists()
    # utf-8-sig: a BOM miatt az Excel és a Numbers is helyesen mutatja az emojikat és az ékezeteket.
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerows(rows)


def save_weather_to_csv(lat: float, lon: float) -> dict:
    """Az adott koordináta aktuális időjárását és helyadatait új sorként a weather_log.csv-be menti.

    Maga gyűjti össze az adatokat (időjárás, település és ország, időzóna, helyi idő), így a modellnek
    csak a koordinátát kell megadnia. A fájl a py mellett jön létre, ha még nincs; ha van, bővül.
    """
    weather = get_weather(lat, lon)
    place = reverse_geocode(lat, lon)
    tz_name = get_timezone(lat, lon)["timezone"]
    # Az Open-Meteo ideje UTC, időzóna nélkül; megjelöljük, hogy UTC, és átváltjuk a helyi időre.
    local = get_time_in_timezone(tz_name, weather["time"] + ":00+00:00")

    details = place["details"]
    city = next((details[k] for k in ("city", "town", "village", "hamlet", "suburb", "municipality", "county") if k in details), "")
    emoji, desc, _ = WEATHER_CODES.get(weather["weathercode"], ("🌡️", f"Ismeretlen ({weather['weathercode']})", 0))

    row = [
        datetime.now().isoformat(timespec="seconds"),
        city,
        country_cell(details),
        lat,
        lon,
        f"{emoji} {desc}",
        weather["temperature"],
        weather["windspeed"],
        weather["winddirection"],
        "🌞 Nappal" if weather["is_day"] else "🌙 Éjszaka",
        weather["time"].replace("T", " "),
        f"{local['datetime'][:16].replace('T', ' ')} ({local['abbreviation']}, UTC{local['utc_offset']})",
        tz_name,
    ]
    _append_csv(CSV_PATH, CSV_HEADER, [row])
    return {"saved": True, "file": str(CSV_PATH), "row": dict(zip(CSV_HEADER, row))}


FORECAST_CSV_PATH = Path(__file__).resolve().parent / "forecast_log.csv"
FORECAST_CSV_HEADER = [
    "🕒 Mentve", "🏙️ Település", "🌍 Ország", "📍 Szélesség", "📍 Hosszúság", "🔮 Előrejelzett idő (helyi)",
    "🌐 Előrejelzett idő (UTC)", "🗺️ Időzóna", "🌤️ Időjárás", "🌡️ Hőmérséklet (°C)", "🤒 Hőérzet (°C)",
    "💧 Csapadék esélye (%)", "☔ Csapadék (mm)", "💨 Szél (km/h)", "🧭 Szélirány (°)", "🌞 Napszak",
]


def save_forecast_to_csv(lat: float, lon: float, datetime_iso: str | None = None, hours: int = 1,
                         place_local_time: bool = False) -> dict:
    """Az előrejelzést (óránként egy sor) a forecast_log.csv-be menti, a py mellé.

    Ugyanazokat a paramétereket és időkezelést használja, mint a get_forecast, a hely- és
    időadatokat (település, ország, UTC és helyi idő) pedig maga gyűjti össze.
    """
    forecast = get_forecast(lat, lon, datetime_iso, hours, place_local_time)
    details = reverse_geocode(lat, lon)["details"]
    city = next((details[k] for k in ("city", "town", "village", "hamlet", "suburb", "municipality", "county") if k in details), "")
    tz_name = forecast["timezone"]
    saved_at = datetime.now().isoformat(timespec="seconds")

    rows = []
    for h in forecast["forecast"]:
        local = datetime.fromisoformat(h["time"]).replace(tzinfo=ZoneInfo(tz_name))
        emoji, desc, _ = WEATHER_CODES.get(h["weathercode"], ("🌡️", f"Ismeretlen ({h['weathercode']})", 0))
        rows.append([
            saved_at, city, country_cell(details), lat, lon,
            local.strftime("%Y-%m-%d %H:%M"),
            local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M"),
            tz_name,
            f"{emoji} {desc}",
            h["temperature"], h["feels_like"], h["precipitation_probability"], h["precipitation_mm"],
            h["windspeed"], h["winddirection"],
            "🌞 Nappal" if h["is_day"] else "🌙 Éjszaka",
        ])
    _append_csv(FORECAST_CSV_PATH, FORECAST_CSV_HEADER, rows)
    return {"saved": True, "file": str(FORECAST_CSV_PATH), "rows_saved": len(rows),
            "rows": [dict(zip(FORECAST_CSV_HEADER, r)) for r in rows]}


def get_forecast(lat: float, lon: float, datetime_iso: str | None = None, hours: int = 1,
                 place_local_time: bool = False) -> dict:
    """Óránkénti előrejelzés a koordinátához, a megadott időponttól kezdve (alapból: most).

    Az időpont értelmezése:
      - időzónával megadott (pl. '2026-11-27T12:00:00+09:00'): abszolút időpont, mindig pontos;
      - időzóna nélkül, place_local_time=False: a gép helyi ideje (ahogy a get_current_datetime adja);
      - időzóna nélkül, place_local_time=True: a hely helyi ideje (pl. "holnap délben Tokióban").
    Az időpontot a legközelebbi egész órára kerekíti. Az eredmény a hely helyi idejében van.
    Az Open-Meteo legfeljebb 16 napra előre ad óránkénti előrejelzést.
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Érvénytelen koordináta: {lat}, {lon}")
    if not 1 <= hours <= 48:
        raise ValueError("Az órák száma 1 és 48 között lehet.")

    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 16,
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,"
                  "weather_code,wind_speed_10m,wind_direction_10m,is_day",
    })
    with urllib.request.urlopen(f"{OPEN_METEO_URL}?{query}", timeout=10) as resp:
        data = json.load(resp)

    tz_name = data["timezone"]
    tz = ZoneInfo(tz_name)

    # A kért időpont átváltása a hely helyi idejére (időzóna nélkül, "naiv" időként).
    if datetime_iso is None:
        target = datetime.now(tz).replace(tzinfo=None)
    else:
        moment = datetime.fromisoformat(datetime_iso)
        if moment.tzinfo is None and place_local_time:
            target = moment  # már a hely helyi ideje
        else:
            # naiv idő -> a gép helyi idejének vesszük (.astimezone()), majd a hely zónájába váltunk
            target = moment.astimezone(tz).replace(tzinfo=None)
    target = (target + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)  # legközelebbi óra

    hourly = data["hourly"]
    times = hourly["time"]  # a hely helyi ideje, pl. '2026-11-26T14:00'
    key = target.strftime("%Y-%m-%dT%H:00")
    if key not in times:
        raise ValueError(
            f"Erre az időpontra ({key}, {tz_name} helyi idő) nincs előrejelzés. "
            f"Elérhető tartomány: {times[0]} és {times[-1]} között."
        )
    start = times.index(key)

    entries = []
    for i in range(start, min(start + hours, len(times))):
        entries.append({
            "time": times[i],
            "temperature": hourly["temperature_2m"][i],
            "feels_like": hourly["apparent_temperature"][i],
            "precipitation_probability": hourly["precipitation_probability"][i],
            "precipitation_mm": hourly["precipitation"][i],
            "weathercode": hourly["weather_code"][i],
            "windspeed": hourly["wind_speed_10m"][i],
            "winddirection": hourly["wind_direction_10m"][i],
            "is_day": hourly["is_day"][i],
        })
    units = data["hourly_units"]
    return {
        "timezone": tz_name,
        "times_are": "a hely helyi ideje",
        "units": {"temperature": units["temperature_2m"], "windspeed": units["wind_speed_10m"],
                  "precipitation": units["precipitation"]},
        "forecast": entries,
    }


TOOL_FUNCTIONS = {"get_current_datetime": get_current_datetime, "add_time": add_time, "get_weekday": get_weekday,
                  "geocode": geocode, "reverse_geocode": reverse_geocode,
                  "get_weather": get_weather, "calculate_distance": calculate_distance,
                  "destination_point": destination_point,
                  "get_timezone": get_timezone, "get_time_in_timezone": get_time_in_timezone,
                  "save_weather_to_csv": save_weather_to_csv,
                  "get_forecast": get_forecast, "save_forecast_to_csv": save_forecast_to_csv}

# ---------------------------------------------------------------------------
# Toolok: sémák
# ---------------------------------------------------------------------------

DATETIME_PROP = {"type": "string", "description": "Időpont ISO formátumban, pl. '2026-11-26T13:25:00'."}

tools = [
    {
        "name": "get_current_datetime",
        "description": "Visszaadja az aktuális helyi dátumot és időt ISO formátumban.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_time",
        "description": "Hozzáad egy időtartamot egy időponthoz, és visszaadja az új időpontot ISO formátumban.",
        "input_schema": {
            "type": "object",
            "properties": {
                "datetime_iso": DATETIME_PROP,
                "amount": {"type": "number", "description": "Hány egység (negatív a múltba lép)."},
                "unit": {"type": "string", "enum": UNITS, "description": "Az időegység."},
            },
            "required": ["datetime_iso", "amount", "unit"],
        },
    },
    {
        "name": "get_weekday",
        "description": "Megmondja, a hét melyik napjára esik egy időpont.",
        "input_schema": {
            "type": "object",
            "properties": {"datetime_iso": DATETIME_PROP},
            "required": ["datetime_iso"],
        },
    },
    {
        "name": "geocode",
        "description": (
            "Forward geocoding: egy helynévhez vagy címhez megadja a koordinátákat (lat, lon). "
            "Több találatot is visszaadhat, a legrelevánsabb az első."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Helynév vagy cím, pl. 'Budapest, Kossuth tér' vagy 'Eiffel-torony'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Találatok maximális száma (alapért. 3)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "reverse_geocode",
        "description": "Reverse geocoding: koordinátákhoz (lat, lon) megadja a legközelebbi címet vagy helyet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Szélesség fokban."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Hosszúság fokban."},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Lekéri az aktuális időjárást egy koordinátapárhoz (hőmérséklet, szélsebesség, szélirány, "
            "weathercode, nappal/éjszaka). Helynévhez előbb a geocode-dal kérd le a koordinátákat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Szélesség fokban."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Hosszúság fokban."},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "calculate_distance",
        "description": (
            "Kiszámolja két koordináta közötti légvonalbeli távolságot (nagykör, haversine) "
            "kilométerben és mérföldben. Nem útvonaltávolság. Helynevekhez előbb a geocode-dal "
            "kérd le a koordinátákat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat1": {"type": "number", "minimum": -90, "maximum": 90, "description": "Az 1. pont szélessége."},
                "lon1": {"type": "number", "minimum": -180, "maximum": 180, "description": "Az 1. pont hosszúsága."},
                "lat2": {"type": "number", "minimum": -90, "maximum": 90, "description": "A 2. pont szélessége."},
                "lon2": {"type": "number", "minimum": -180, "maximum": 180, "description": "A 2. pont hosszúsága."},
            },
            "required": ["lat1", "lon1", "lat2", "lon2"],
        },
    },
    {
        "name": "destination_point",
        "description": (
            "Megadja, hogy egy kiindulási koordinátától adott irányban (bearing) adott távolságra "
            "milyen koordináta van, pl. 'mi van 100 km-rel északra'. Az irányt fokban kell megadni: "
            "0 = észak, 90 = kelet, 180 = dél, 270 = nyugat (köztes irányok pl. 45 = északkelet). "
            "Az eredmény koordinátájához a reverse_geocode-dal kérhető le a hely neve."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Kiindulási szélesség."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Kiindulási hosszúság."},
                "distance": {"type": "number", "minimum": 0, "description": "A megteendő távolság (nem negatív)."},
                "bearing": {"type": "number", "description": "Irány fokban: 0 = É, 90 = K, 180 = D, 270 = Ny."},
                "unit": {"type": "string", "enum": ["km", "miles"], "description": "A távolság egysége (alapért. km)."},
            },
            "required": ["lat", "lon", "distance", "bearing"],
        },
    },
    {
        "name": "get_timezone",
        "description": (
            "Megadja egy koordinátához (lat, lon) tartozó IANA időzóna nevét, pl. 'Europe/Budapest'. "
            "Helynévhez előbb a geocode-dal kérd le a koordinátákat. Az eredményt a get_time_in_timezone-nak add tovább."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Szélesség fokban."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Hosszúság fokban."},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "get_time_in_timezone",
        "description": (
            "Megmondja, mennyi az idő egy adott IANA időzónában (pl. 'Asia/Tokyo') egy megadott időpont alapján. "
            "Az időpont lehet időzónával (offsettel) vagy anélkül; az utóbbit a gép helyi idejének veszi. "
            "Ha nincs időpont megadva, a pillanatnyi időt váltja át."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA időzónanév, pl. 'Europe/Budapest', 'America/New_York'."},
                "datetime_iso": {"type": "string", "description": "Átváltandó időpont ISO formátumban (opcionális)."},
            },
            "required": ["timezone"],
        },
    },
    {
        "name": "save_weather_to_csv",
        "description": (
            "Elmenti egy koordináta aktuális időjárását és helyadatait (település, ország, koordináta, UTC és helyi idő, "
            "időzóna) egy CSV fájlba. CSAK akkor hívd, ha a felhasználó kifejezetten kéri a mentést, naplózást "
            "vagy CSV-be írást; soha ne mentsd magadtól. Az adatokat a tool maga gyűjti össze, neked csak a "
            "koordinátát kell megadnod (helynévhez előbb a geocode-dal kérd le)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Szélesség fokban."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Hosszúság fokban."},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "get_forecast",
        "description": (
            "Óránkénti időjárás-előrejelzés egy koordinátához (hőmérséklet, hőérzet, csapadék esélye és mennyisége, "
            "szél, weathercode), legfeljebb 16 napra előre, a megadott időponttól kezdve. Ha nincs időpont, a mostantól. "
            "Az időpontot a legközelebbi órára kerekíti, az eredmény a hely helyi idejében van. "
            "Időpont megadása: ha a felhasználó a hely helyi idejére gondol (pl. 'holnap délben Tokióban'), "
            "add meg a datetime_iso-t időzóna nélkül és állítsd place_local_time-ot true-ra. Ha a gép idejéhez képest "
            "számolt időpont (pl. '3 óra múlva', amit a get_current_datetime + add_time adott), hagyd false-on. "
            "Offsettel megadott időpont (pl. +09:00) mindig abszolút. Helynévhez előbb a geocode-dal kérd le a koordinátákat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Szélesség fokban."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Hosszúság fokban."},
                "datetime_iso": {"type": "string", "description": "Az előrejelzés kezdő időpontja ISO formátumban (opcionális, alapért.: most)."},
                "hours": {"type": "integer", "minimum": 1, "maximum": 48, "description": "Hány egymást követő órát adjon vissza (alapért. 1). Pl. 'délután' = 6."},
                "place_local_time": {"type": "boolean", "description": "true: az időzóna nélküli datetime_iso a hely helyi ideje; false (alapért.): a gép helyi ideje."},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "save_forecast_to_csv",
        "description": (
            "Elmenti az óránkénti előrejelzést (óránként egy sor: település, ország, koordináta, helyi és UTC idő, "
            "időjárás, hőmérséklet, hőérzet, csapadék, szél) a forecast_log.csv fájlba. CSAK akkor hívd, ha a felhasználó "
            "kifejezetten kéri az előrejelzés mentését, naplózását vagy CSV-be írását; soha ne mentsd magadtól. "
            "Az időparaméterek ugyanúgy működnek, mint a get_forecast-nál (datetime_iso, hours, place_local_time)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "minimum": -90, "maximum": 90, "description": "Szélesség fokban."},
                "lon": {"type": "number", "minimum": -180, "maximum": 180, "description": "Hosszúság fokban."},
                "datetime_iso": {"type": "string", "description": "Az előrejelzés kezdő időpontja ISO formátumban (opcionális, alapért.: most)."},
                "hours": {"type": "integer", "minimum": 1, "maximum": 48, "description": "Hány egymást követő órát mentsen (alapért. 1)."},
                "place_local_time": {"type": "boolean", "description": "true: az időzóna nélküli datetime_iso a hely helyi ideje; false (alapért.): a gép helyi ideje."},
            },
            "required": ["lat", "lon"],
        },
    },
]

# ---------------------------------------------------------------------------
# Szép megjelenítés (csak a terminálnak: a modell továbbra is a nyers JSON-t kapja)
# ---------------------------------------------------------------------------

# Színek csak valódi terminálban (nem pipe-ban), a NO_COLOR-t tiszteletben tartva; FORCE_COLOR=1 felülírja.
USE_COLOR = bool(os.environ.get("FORCE_COLOR")) or (sys.stdout.isatty() and "NO_COLOR" not in os.environ)
RESET = "\033[0m"


def c(text, color: int | None = None, bold: bool = False, dim: bool = False) -> str:
    """Xterm 256 színnel (\033[38;5;Nm) színezi a szöveget; színtelen módban változatlanul adja vissza."""
    if not USE_COLOR:
        return str(text)
    codes = ("\033[1m" if bold else "") + ("\033[2m" if dim else "") + (f"\033[38;5;{color}m" if color is not None else "")
    return f"{codes}{text}{RESET}"


# WMO weathercode -> (emoji, magyar leírás, xterm szín)
WEATHER_CODES = {
    0: ("☀️", "Derült", 220), 1: ("🌤️", "Többnyire derült", 221), 2: ("⛅", "Változóan felhős", 252),
    3: ("☁️", "Borult", 245), 45: ("🌫️", "Köd", 247), 48: ("🌫️", "Zúzmarás köd", 251),
    51: ("🌦️", "Gyenge szitálás", 117), 53: ("🌦️", "Szitálás", 111), 55: ("🌦️", "Erős szitálás", 75),
    56: ("🌧️", "Fagyos szitálás", 159), 57: ("🌧️", "Erős fagyos szitálás", 159),
    61: ("🌧️", "Gyenge eső", 117), 63: ("🌧️", "Eső", 75), 65: ("🌧️", "Erős eső", 33),
    66: ("🌧️", "Fagyos eső", 159), 67: ("🌧️", "Erős fagyos eső", 159),
    71: ("🌨️", "Gyenge havazás", 195), 73: ("🌨️", "Havazás", 189), 75: ("❄️", "Erős havazás", 255),
    77: ("❄️", "Hószemcsék", 255), 80: ("🌦️", "Gyenge zápor", 117), 81: ("🌧️", "Zápor", 75),
    82: ("⛈️", "Heves zápor", 33), 85: ("🌨️", "Hózápor", 195), 86: ("❄️", "Erős hózápor", 255),
    95: ("⛈️", "Zivatar", 135), 96: ("⛈️", "Zivatar jégesővel", 135), 99: ("⛈️", "Heves zivatar jégesővel", 129),
}
# A szél irányából jön (fok); a nyíl azt mutatja, merre fúj. Index: round(fok / 45) % 8
WIND_DIRECTIONS = ["északi", "északkeleti", "keleti", "délkeleti", "déli", "délnyugati", "nyugati", "északnyugati"]
WIND_ARROWS = ["↓", "↙", "←", "↖", "↑", "↗", "→", "↘"]


def temp_color(t: float) -> int:
    """A hőmérséklethez tartozó xterm szín: fagy = cián, hideg = kék, enyhe = zöld, meleg = sárga, forró = piros."""
    for limit, color in ((0, 51), (10, 75), (18, 114), (26, 220), (32, 208)):
        if t <= limit:
            return color
    return 196


def wind_color(speed: float) -> int:
    for limit, color in ((15, 78), (30, 190), (50, 208)):
        if speed <= limit:
            return color
    return 196


def format_weather(w: dict) -> str:
    """A get_weather eredményét szép, színes, emojis blokká alakítja."""
    emoji, desc, desc_color = WEATHER_CODES.get(w["weathercode"], ("🌡️", f"Ismeretlen ({w['weathercode']})", 250))
    t_unit, w_unit = w["units"]["temperature"], w["units"]["windspeed"]
    idx = round(w["winddirection"] / 45) % 8
    when = w["time"].replace("T", " ")
    lines = [
        f"{emoji}  {c(desc, desc_color, bold=True)}  {c('·', dim=True)}  {c(when + ' UTC', dim=True)}",
        f"   🌡️  Hőmérséklet  {c(f'{w['temperature']} {t_unit}', temp_color(w['temperature']), bold=True)}",
        f"   💨 Szél         {c(f'{w['windspeed']} {w_unit}', wind_color(w['windspeed']), bold=True)}"
        f"  {WIND_ARROWS[idx]} {WIND_DIRECTIONS[idx]} ({w['winddirection']}°)",
        f"   {'🌞 Nappal' if w['is_day'] else '🌙 Éjszaka'}",
    ]
    return "\n".join(lines)


def format_forecast(f: dict) -> str:
    """A get_forecast eredményét óránkénti, színes, emojis sorokká alakítja."""
    t_unit, w_unit = f["units"]["temperature"], f["units"]["windspeed"]
    lines = [f"🔮 {c('Előrejelzés', bold=True)}  {c('·', dim=True)}  {c(f['timezone'] + ' helyi idő', dim=True)}"]
    for h in f["forecast"]:
        emoji, desc, desc_color = WEATHER_CODES.get(h["weathercode"], ("🌡️", f"Ismeretlen ({h['weathercode']})", 250))
        idx = round(h["winddirection"] / 45) % 8
        day, hour = h["time"].split("T")
        temp = c(f"{h['temperature']:>5.1f}{t_unit}", temp_color(h["temperature"]), bold=True)
        feels = c(f"(érzet {h['feels_like']:.1f}{t_unit})", dim=True)
        rain = f"💧 {h['precipitation_probability']:>3}% {h['precipitation_mm']:.1f} mm"
        wind = c(f"{h['windspeed']:>4.0f} {w_unit}", wind_color(h["windspeed"]))
        lines.append(
            f"   {c(day[5:] + ' ' + hour, 244)}  {emoji} {c(f'{desc:<18}', desc_color)} "
            f"🌡️ {temp} {feels}  {rain}  💨 {wind} {WIND_ARROWS[idx]}  {'🌞' if h['is_day'] else '🌙'}"
        )
    return "\n".join(lines)


# tool név -> megjelenítő függvény. Ami itt nincs, az nyers formában íródik ki.
# Új tool szebb kiírásához elég ide felvenni egy függvényt.
DISPLAY_FORMATTERS = {"get_weather": format_weather, "get_forecast": format_forecast}


def show_result(name: str, tool_input: dict, result: dict):
    """Kiírja a tool hívást és az eredményét; ismert toolnál (pl. get_weather) formázva."""
    call = f"{name}({json.dumps(tool_input, ensure_ascii=False)})"
    formatter = DISPLAY_FORMATTERS.get(name)
    if formatter and not result.get("is_error"):
        try:
            print(f"   🔧 {c(call, 244)}\n" + "\n".join("   " + line for line in formatter(json.loads(result["content"])).split("\n")))
            return
        except (KeyError, ValueError, TypeError):
            pass  # ha váratlan az adat, esünk vissza a nyers kiírásra
    print(f"   🔧 {c(call, 244)} -> {result['content']}")


# ---------------------------------------------------------------------------
# A tool loop
# ---------------------------------------------------------------------------


def run_tool(block) -> dict:
    """Egy tool_use blokkból tool_result blokkot készít (hiba esetén is_error jelzéssel)."""
    try:
        result = TOOL_FUNCTIONS[block.name](**block.input)
        # a dict/list eredményt JSON-ként küldjük, a többit simán stringként
        content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
        return {"type": "tool_result", "tool_use_id": block.id, "content": content}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": block.id, "content": f"Hiba: {e}", "is_error": True}


SYSTEM_PROMPT = (
    "Válaszolj magyarul, egyszerűen és tömören, köznapi nyelven. "
    "Ne használd a technikai részleteket: a weathercode-ot, a nyers koordinátákat és az ISO időformátumot "
    "ne említsd, az időjárást szavakkal írd le (pl. 'borult, 2 fok, gyenge északkeleti szél'). "
    "Az időt természetesen add meg (pl. '14:30', 'szombat'). "
    "Ne formázd markuppal, mert konzolban jelenik meg. "
    "Csak akkor mentsd CSV-be az adatokat, ha a felhasználó kifejezetten kéri; magadtól soha."
)


def call_model(messages):
    response = client.messages.create(
        model=model, max_tokens=1000, system=SYSTEM_PROMPT, messages=messages, tools=tools
    )
    messages.append({"role": "assistant", "content": response.content})
    return response


def chat(messages, question: str) -> str:
    messages.append({"role": "user", "content": question})
    response = call_model(messages)
    calls = 0

    # Addig megyünk, amíg a modell toolt kér. Amikor már nem, a válasza a végső szöveg.
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                calls += 1
                result = run_tool(block)
                show_result(block.name, block.input, result)
                tool_results.append(result)
        messages.append({"role": "user", "content": tool_results})
        response = call_model(messages)

    print(f"   ✅ A modellnek már nem kellett tool ({calls} tool hívás után)")
    return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    print("Kérdezz időről vagy helyekről, pl. 'Milyen napra esik 100 nap múlva?' vagy 'Hol van a 48.8584, 2.2945?' vagy 'Milyen idő van Pécsen?' vagy 'Milyen messze van Pécs Budapesttől?' vagy 'Mi van 100 km-rel északra Pécstől?' vagy 'Mennyi az idő most Tokióban?' vagy 'Milyen idő lesz holnap délben Pécsen?' 'exit' a kilépéshez.\n")
    messages = []
    while True:
        question = input("👤 Te: ").strip()
        if question.lower() == "exit":
            print("👋 Viszlát!")
            break
        if not question:
            continue
        print(f"\n🤖 Válasz: {chat(messages, question)}\n")
