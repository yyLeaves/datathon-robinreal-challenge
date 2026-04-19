"""
Merge all 5 feature distance files + pre-computed city-center distances
into the listing_geo table in the DB.

Run after all five fetch.py scripts and geocode_cities.py have completed:
  cd /workshop/geo_features
  python3 build_listing_geo.py

listing_geo schema:
  listing_id, dist_lake_km, dist_park_km, dist_school_km,
  dist_transport_km, dist_shop_km, dist_city_center_km
"""
import json, math, sqlite3, time
from pathlib import Path

HERE         = Path(__file__).parent
DB_PATH      = Path("/workshop/datathon-robinreal-challenge/data/listings.db")
COORDS       = HERE / "listings_coords.jsonl"
CITY_CENTERS = HERE / "city_centers.json"

FEATURE_FILES = {
    "dist_lake_km":      HERE / "lake/lake_distances.jsonl",
    "dist_park_km":      HERE / "park/park_distances.jsonl",
    "dist_school_km":    HERE / "school/school_distances.jsonl",
    "dist_transport_km": HERE / "transport/transport_distances.jsonl",
    "dist_shop_km":      HERE / "shop/shop_distances.jsonl",
}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    d = math.radians
    a = math.sin(d(lat2-lat1)/2)**2 + math.cos(d(lat1))*math.cos(d(lat2))*math.sin(d(lon2-lon1)/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def load_feature(path: Path, col: str) -> dict[str, float | None]:
    print(f"  Loading {path.name}...", end=" ", flush=True)
    data = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            data[str(r["listing_id"])] = r.get(col)
    print(f"{len(data)} rows")
    return data


def main():
    missing = [str(p) for p in FEATURE_FILES.values() if not p.exists()]
    if missing:
        print("Missing distance files — run each fetch.py first:")
        for m in missing:
            print(f"  {m}")
        return

    print("Loading feature distance files...")
    features: dict[str, dict] = {}
    for col, path in FEATURE_FILES.items():
        features[col] = load_feature(path, col)

    rows = [(json.loads(l)["listing_id"], json.loads(l).get("lat"), json.loads(l).get("lon"), json.loads(l).get("city"))
            for l in COORDS.open()]

    city_centers = {}
    if CITY_CENTERS.exists():
        print("Loading city centers from local file...")
        city_centers = json.loads(CITY_CENTERS.read_text())
    else:
        print("WARNING: city_centers.json not found, run geocode_cities.py first. dist_city_center_km will be null.")

    con = sqlite3.connect(DB_PATH)
    print("Creating listing_geo table...")
    con.execute("""
        CREATE TABLE IF NOT EXISTS listing_geo (
            listing_id TEXT PRIMARY KEY,
            dist_lake_km REAL,
            dist_park_km REAL,
            dist_school_km REAL,
            dist_transport_km REAL,
            dist_shop_km REAL,
            dist_city_center_km REAL
        )
    """)

    print(f"Inserting {len(rows)} rows...")
    t0 = time.time()
    batch = []
    for lid, lat, lon, city in rows:
        lid = str(lid)
        center = city_centers.get(city) if city else None
        dist_city = round(haversine(lat, lon, center[0], center[1]), 3) if center and lat and lon else None
        batch.append((
            lid,
            features["dist_lake_km"].get(lid),
            features["dist_park_km"].get(lid),
            features["dist_school_km"].get(lid),
            features["dist_transport_km"].get(lid),
            features["dist_shop_km"].get(lid),
            dist_city,
        ))
    con.executemany(
        """INSERT OR REPLACE INTO listing_geo
           (listing_id, dist_lake_km, dist_park_km, dist_school_km,
            dist_transport_km, dist_shop_km, dist_city_center_km)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        batch,
    )
    con.commit()
    con.close()
    print(f"Done. {len(batch)} rows written in {time.time()-t0:.1f}s → {DB_PATH}")
    city_nulls = sum(1 for r in batch if r[6] is None)
    print(f"  dist_city_center_km: {len(batch)-city_nulls} computed, {city_nulls} null")


if __name__ == "__main__":
    main()
