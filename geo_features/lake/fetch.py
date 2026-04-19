"""
Fetch lake/reservoir locations from Overpass for all of Switzerland,
then compute nearest-lake distance for every listing in the DB.

Outputs:
  lake_locs.json          — raw OSM locations [{lat, lon}, ...]
  lake_distances.jsonl    — one line per listing: {listing_id, dist_lake_km}

Run:
  python3 fetch.py
"""
import json, math, time
from pathlib import Path
import httpx

HERE     = Path(__file__).parent
COORDS   = HERE.parent / "listings_coords.jsonl"
LOCS_OUT = HERE / "lake_locs.json"
DIST_OUT = HERE / "lake_distances.jsonl"
OVERPASS = "https://overpass-api.de/api/interpreter"
CH_BBOX  = "45.8,5.9,47.9,10.6"
HEADERS  = {"User-Agent": "datathon2026-geo/1.0"}

QUERY = f"""[out:json][timeout:90];
(
  way["natural"="water"]["water"~"lake|reservoir"]({CH_BBOX});
  relation["natural"="water"]["water"~"lake|reservoir"]({CH_BBOX});
);
out center;"""


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    d = math.radians
    a = math.sin(d(lat2-lat1)/2)**2 + math.cos(d(lat1))*math.cos(d(lat2))*math.sin(d(lon2-lon1)/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def fetch_locations():
    if LOCS_OUT.exists():
        print(f"Loading cached {LOCS_OUT.name}...")
        return json.loads(LOCS_OUT.read_text())

    print("Fetching lake locations from Overpass...", flush=True)
    t0 = time.time()
    resp = httpx.post(OVERPASS, data={"data": QUERY}, headers=HEADERS, timeout=120.0)
    locs = []
    for el in resp.json().get("elements", []):
        if "center" in el:
            locs.append({"lat": el["center"]["lat"], "lon": el["center"]["lon"]})
        elif "lat" in el and "lon" in el:
            locs.append({"lat": el["lat"], "lon": el["lon"]})
    LOCS_OUT.write_text(json.dumps(locs, indent=2))
    print(f"  {len(locs)} lakes saved → {LOCS_OUT.name}  ({time.time()-t0:.1f}s)")
    return locs


def compute_distances(locs):
    rows = [(r["listing_id"], r["lat"], r["lon"]) for r in (json.loads(l) for l in COORDS.open())]
    print(f"Computing distances for {len(rows)} listings against {len(locs)} lakes...")
    t0 = time.time()
    loc_tuples = [(l["lat"], l["lon"]) for l in locs]
    with DIST_OUT.open("w") as f:
        for i, (lid, lat, lon) in enumerate(rows):
            dist = round(min(haversine(lat, lon, alat, alon) for alat, alon in loc_tuples), 3) if loc_tuples else None
            f.write(json.dumps({"listing_id": str(lid), "dist_lake_km": dist}) + "\n")
            if (i+1) % 2000 == 0:
                print(f"  {i+1}/{len(rows)}  {time.time()-t0:.0f}s")
    print(f"  Done. {len(rows)} rows → {DIST_OUT.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    locs = fetch_locations()
    compute_distances(locs)
