from __future__ import annotations

from app.core.claude import EXTRACTION_MAX_TOKENS, FAST_MODEL, client as _client
from app.models.schemas import HardFilters

_SYSTEM = (
    "Extract Swiss real estate search filters from queries in any language (English, German, French, Italian). "
    "Only set fields explicitly mentioned as hard requirements; omit the rest. "
    "IMPORTANT: Words like 'idealerweise', 'gern', 'wenn möglich', 'ideally', 'preferably', 'if possible', 'nice to have' "
    "signal soft preferences — do NOT extract these as hard filters. Only extract features as hard filters when the user "
    "says 'must have', 'required', 'with X' as a firm requirement, or explicitly filters by it. "
    "Always normalize city names to their common English/German spelling (e.g. Genf/Genève → Geneva, Zürich → Zurich, Berne → Bern). "
    "When a district or neighborhood is mentioned (e.g. Oerlikon, Altstetten, Wiedikon, Kreis 4), set the parent city in 'city' (e.g. Zurich) AND set the district name in 'neighborhood'. "
    "When specific cities are mentioned, set ONLY those cities in the city field — do NOT set canton unless no city is given. "
    "Canton codes for reference only (use when no city given): "
    "Zurich/Zürich → ZH, Geneva/Genève/Genf → GE, Basel → BS, Bern → BE, Lausanne → VD, "
    "Lucerne/Luzern → LU, Zug → ZG, St. Gallen → SG, Winterthur → ZH, Lugano → TI. "
    "For room counts, set min_rooms only unless a range or maximum is explicitly stated. "
    "For area, extract min_area/max_area in square metres if mentioned. "
    "For available_from, use ISO date format YYYY-MM-DD (e.g. 'from June 2026' → '2026-06-01'). "
    "Always set offer_type to RENT unless the user explicitly wants to buy. "
    "If the user mentions proximity to a landmark or institution (e.g. 'near ETH', 'close to EPFL', 'near the train station'), "
    "set near_place to the landmark name (e.g. 'ETH Zürich', 'EPFL Lausanne', 'Zurich Hauptbahnhof'). "
    "Do NOT set latitude/longitude/radius_km yourself — leave those empty."
)

_TOOL = {
    "name": "extract_filters",
    "description": "Structured filters from a real estate query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city":            {"type": "array", "items": {"type": "string"}},
            "neighborhood":    {"type": "array", "items": {"type": "string"}, "description": "District/neighborhood names within the city (e.g. Oerlikon, Altstetten, Kreis 4)"},
            "postal_code":     {"type": "array", "items": {"type": "string"}},
            "canton":          {"type": "string", "description": "2-letter code e.g. ZH"},
            "min_price":       {"type": "integer"},
            "max_price":       {"type": "integer"},
            "min_rooms":       {"type": "number"},
            "max_rooms":       {"type": "number"},
            "min_area":        {"type": "integer", "description": "Minimum living area in sqm"},
            "max_area":        {"type": "integer", "description": "Maximum living area in sqm"},
            "available_from":  {"type": "string", "description": "ISO date YYYY-MM-DD, latest move-in date acceptable"},
            "near_place":      {"type": "string", "description": "Landmark/institution name to geocode (e.g. 'ETH Zürich', 'EPFL Lausanne'). Do not set latitude/longitude yourself."},
            "latitude":        {"type": "number"},
            "longitude":       {"type": "number"},
            "radius_km":       {"type": "number"},
            "features":        {"type": "array", "items": {"type": "string", "enum": ["balcony", "elevator", "parking", "garage", "fireplace", "child_friendly", "pets_allowed", "temporary", "new_build", "wheelchair_accessible", "private_laundry", "minergie_certified", "furnished", "garden"]}},
            "offer_type":      {"type": "string", "enum": ["RENT", "SALE"]},
            "object_category": {"type": "array", "items": {"type": "string", "enum": ["Wohnung", "Haus", "Studio", "Loft", "Villa", "Maisonette", "Attika", "Dachwohnung", "Reihenhaus", "Terrassenhaus", "Terrassenwohnung", "Doppeleinfamilienhaus", "Mehrfamilienhaus", "Bauernhaus", "Einzelzimmer", "WG-Zimmer", "Ferienwohnung", "Ferienimmobilie", "Gewerbeobjekt", "Parkplatz", "Einzelgarage", "Tiefgarage"]}},
            "sort_by":         {"type": "string", "enum": ["price_asc", "price_desc", "rooms_asc", "rooms_desc"]},
            "limit":           {"type": "integer"},
            "offset":          {"type": "integer"},
        },
        "required": [],
    },
}


def extract_hard_facts(query: str) -> HardFilters:
    # --- HARDCODED for local testing — swap comments to use live API ---
    response = _client.messages.create(
        model=FAST_MODEL,
        max_tokens=EXTRACTION_MAX_TOKENS,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "extract_filters"},
        messages=[{"role": "user", "content": query}],
    )
    tool_input = next(b.input for b in response.content if b.type == "tool_use")
    # return HardFilters(**tool_input)
    # Query: "3-room bright apartment in Zurich under 2800 CHF with balcony"
    # tool_input = {
    #     "city": ["Zurich"],
    #     "max_price": 2800,
    #     "min_rooms": 3.0,
    #     "features": ["balcony"],
    #     "offer_type": "RENT",
    # }
    return HardFilters(**tool_input)
