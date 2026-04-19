from __future__ import annotations

from typing import Any

from app.core.claude import EXTRACTION_MAX_TOKENS, FAST_MODEL, client as _client

_SYSTEM = (
    "Extract soft preferences from a Swiss real estate query — things that matter for ranking but "
    "are not exact database filters. Return only what is explicitly or clearly implied in the query. "
    "Do not invent preferences not mentioned."
)

_TOOL = {
    "name": "extract_soft_facts",
    "description": "Soft preferences for ranking listings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "raw_query":          {"type": "string", "description": "Original query verbatim"},
            "bright":             {"type": "boolean", "description": "User wants bright/sunny place"},
            "quiet":              {"type": "boolean", "description": "User wants quiet/peaceful area"},
            "modern":             {"type": "boolean", "description": "User wants modern/renovated"},
            "nice_views":         {"type": "boolean", "description": "User wants views/panorama"},
            "close_to_transport": {"type": "boolean", "description": "User wants public transport nearby"},
            "family_friendly":    {"type": "boolean", "description": "User wants family-friendly, near schools/kindergarten"},
            "student":            {"type": "boolean", "description": "Student accommodation"},
            "modern_kitchen":     {"type": "boolean", "description": "User wants modern/equipped kitchen"},
            "furnished":          {"type": "boolean", "description": "User wants furnished place"},
            "max_commute_minutes":{"type": "integer", "description": "Max commute time in minutes if mentioned"},
            "commute_destination":{"type": "string",  "description": "Destination for commute e.g. ETH Zurich"},
            "move_in_month":      {"type": "string",  "description": "Desired move-in month e.g. June 2025"},
            "affordable":         {"type": "boolean", "description": "User wants cheap/affordable but gave no exact price"},
            "expat":              {"type": "boolean", "description": "Expat or international relocating to Switzerland"},
            "close_to_university":{"type": "boolean", "description": "User wants to be near a university or campus"},
            "parking":            {"type": "boolean", "description": "User wants parking or garage"},
            "near_lake":          {"type": "boolean", "description": "User wants to be near a lake or water body"},
            "safe_neighborhood":  {"type": "boolean", "description": "User explicitly wants a safe, secure neighborhood"},
            "good_schools_nearby":{"type": "boolean", "description": "User wants good schools or kindergartens nearby"},
            "keywords":           {"type": "array", "items": {"type": "string"}, "description": "Other notable preferences not covered above"},
        },
        "required": ["raw_query"],
    },
}


def extract_soft_facts(query: str) -> dict[str, Any]:
    if not query:
        return {"raw_query": query}

    response = _client.messages.create(
        model=FAST_MODEL,
        max_tokens=EXTRACTION_MAX_TOKENS,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "extract_soft_facts"},
        messages=[{"role": "user", "content": query}],
    )
    tool_input = next(b.input for b in response.content if b.type == "tool_use")
    return tool_input
