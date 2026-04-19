"""
Geocode all unique cities in listings_coords.jsonl using Nominatim,
save results to city_centers.json.

Run once:
  cd /workshop/geo_features
  python3 geocode_cities.py
"""
import json, time
from pathlib import Path
import httpx

HERE      = Path(__file__).parent
COORDS    = HERE / "listings_coords.jsonl"
OUT       = HERE / "city_centers.json"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS   = {"User-Agent": "datathon2026-geo/1.0"}


def geocode(city: str, client: httpx.Client):
    q = city if "switzerland" in city.lower() else f"{city} Switzerland"
    try:
        resp = client.get(NOMINATIM, params={"q": q, "format": "json", "limit": 1, "countrycodes": "ch"},
                          headers=HEADERS, timeout=10.0)
        hits = resp.json()
        if hits:
            return float(hits[0]["lat"]), float(hits[0]["lon"])
    except Exception:
        pass
    return None


def main():
    cities = {json.loads(l).get("city") for l in COORDS.open()}
    cities.discard(None)
    cities = sorted(cities)
    print(f"Geocoding {len(cities)} cities...")

    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    results = dict(existing)
    todo = [c for c in cities if c not in results]
    print(f"  {len(existing)} cached, {len(todo)} remaining")

    with httpx.Client() as client:
        for i, city in enumerate(todo):
            coord = geocode(city, client)
            results[city] = list(coord) if coord else None
            if (i + 1) % 100 == 0:
                OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(f"  {i+1}/{len(todo)} saved")
            time.sleep(1.0)  # Nominatim rate limit: 1 req/s

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    found = sum(1 for v in results.values() if v)
    print(f"Done. {found}/{len(results)} cities found → {OUT}")


if __name__ == "__main__":
    main()
