"""
Pre-compute geographic distances for every listing and store in the DB.

Adds a table `listing_geo` with one row per listing:
    listing_id, dist_lake_km, dist_park_km, dist_school_km,
    dist_transport_km, dist_shop_km, dist_city_center_km

Run once:
    python3 precompute_geo_features.py

soft_filtering.py can then join on listing_id instead of calling Overpass.
"""
from __future__ import annotations
import math, time, json, sys
from pathlib import Path
from collections import defaultdict
import httpx

DB_PATH   = Path("/workshop/datathon-robinreal-challenge/data/listings.db")
OVERPASS  = "https://overpass-api.de/api/interpreter"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS   = {"User-Agent": "datathon2026-precompute/1.0"}

# Switzerland bbox with padding
CH_BBOX = "45.8,5.9,47.9,10.6"

AMENITY_QUERIES = {
    "lake": f"""[out:json][timeout:60];
(
  way["natural"="water"]["water"~"lake|reservoir"]({CH_BBOX});
  relation["natural"="water"]["water"~"lake|reservoir"]({CH_BBOX});
);
out center;""",
    "park": f"""[out:json][timeout:60];
(
  way["leisure"~"park|playground|nature_reserve"]({CH_BBOX});
  node["leisure"~"park|playground"]({CH_BBOX});
);
out center;""",
    "school": f"""[out:json][timeout:60];
(
  node["amenity"~"school|kindergarten"]({CH_BBOX});
  way["amenity"~"school|kindergarten"]({CH_BBOX});
);
out center;""",
    "transport": f"""[out:json][timeout:60];
(
  node["public_transport"="stop_position"]({CH_BBOX});
  node["railway"~"station|halt|tram_stop"]({CH_BBOX});
);
out center;""",
    "shop": f"""[out:json][timeout:60];
(
  node["shop"~"supermarket|convenience|mall"]({CH_BBOX});
);
out center;""",
}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    d_lat, d_lon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_overpass(name: str, query: str) -> list[tuple[float, float]]:
    print(f"  Fetching {name} from Overpass...", end=" ", flush=True)
    t0 = time.time()
    try:
        resp = httpx.post(OVERPASS, data={"data": query}, headers=HEADERS, timeout=90.0)
        locations = []
        for el in resp.json().get("elements", []):
            if "center" in el:
                locations.append((el["center"]["lat"], el["center"]["lon"]))
            elif "lat" in el and "lon" in el:
                locations.append((el["lat"], el["lon"]))
        print(f"{len(locations)} locations  {time.time()-t0:.1f}s")
        return locations
    except Exception as e:
        print(f"ERROR: {e}")
        return []


def nearest_km(lat, lon, locations):
    if not locations:
        return None
    return round(min(haversine(lat, lon, alat, alon) for alat, alon in locations), 3)


def city_center_km(lat, lon, city, city_centers):
    center = city_centers.get(city)
    if not center:
        return None
    return round(haversine(lat, lon, *center), 3)


def geocode_cities(cities: list[str]) -> dict[str, tuple[float, float]]:
    result = {}
    for city in cities:
        q = city if "switzerland" in city.lower() else f"{city} Switzerland"
        try:
            resp = httpx.get(NOMINATIM, params={"q": q, "format": "json", "limit": 1, "countrycodes": "ch"},
                             headers=HEADERS, timeout=5.0)
            hits = resp.json()
            if hits:
                result[city] = (float(hits[0]["lat"]), float(hits[0]["lon"]))
        except Exception:
            pass
    return result


def main():
    import sqlite3
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row

    # Create table
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
    con.commit()

    # Load listings with coordinates
    rows = con.execute(
        "SELECT listing_id, latitude, longitude, city FROM listings WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()
    print(f"Loaded {len(rows)} listings with coordinates")

    # Fetch all amenity locations from Overpass (one query per type, covers all CH)
    amenity_locs = {}
    for name, query in AMENITY_QUERIES.items():
        amenity_locs[name] = fetch_overpass(name, query)

    # Geocode unique cities for city-center distance
    cities = list({r["city"] for r in rows if r["city"]})
    print(f"Geocoding {len(cities)} unique cities...")
    city_centers = geocode_cities(cities)
    print(f"  Got centers for {len(city_centers)}/{len(cities)} cities")

    # Compute distances for each listing
    print(f"Computing distances for {len(rows)} listings...")
    t0 = time.time()
    batch = []
    for i, row in enumerate(rows):
        lat, lon = row["latitude"], row["longitude"]
        batch.append((
            str(row["listing_id"]),
            nearest_km(lat, lon, amenity_locs["lake"]),
            nearest_km(lat, lon, amenity_locs["park"]),
            nearest_km(lat, lon, amenity_locs["school"]),
            nearest_km(lat, lon, amenity_locs["transport"]),
            nearest_km(lat, lon, amenity_locs["shop"]),
            city_center_km(lat, lon, row["city"], city_centers),
        ))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(rows)}  {time.time()-t0:.0f}s")

    con.executemany(
        """INSERT OR REPLACE INTO listing_geo
           (listing_id, dist_lake_km, dist_park_km, dist_school_km,
            dist_transport_km, dist_shop_km, dist_city_center_km)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        batch,
    )
    con.commit()
    con.close()

    total = time.time() - t0
    print(f"\nDone. {len(batch)} rows written in {total:.1f}s → {DB_PATH}")


if __name__ == "__main__":
    main()
