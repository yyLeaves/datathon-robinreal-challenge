"""
One-time script to reverse-geocode SRED listings missing city/canton/postal_code.
Uses local reverse_geocoder (no API calls). Run once after DB import.

Usage:
    uv run python scripts/enrich_sred_locations.py
"""
from __future__ import annotations

from pathlib import Path

import reverse_geocoder

from app.db import get_connection

# Swiss canton codes keyed by ISO 3166-2 admin1 code from GeoNames
_ADMIN1_TO_CANTON: dict[str, str] = {
    "Aargau": "AG", "Appenzell Ausserrhoden": "AR", "Appenzell Innerrhoden": "AI",
    "Basel-Landschaft": "BL", "Basel-Stadt": "BS", "Bern": "BE",
    "Fribourg": "FR", "Geneva": "GE", "Glarus": "GL", "Graubünden": "GR",
    "Jura": "JU", "Lucerne": "LU", "Neuchâtel": "NE", "Nidwalden": "NW",
    "Obwalden": "OW", "Schaffhausen": "SH", "Schwyz": "SZ", "Solothurn": "SO",
    "St. Gallen": "SG", "Thurgau": "TG", "Ticino": "TI", "Uri": "UR",
    "Valais": "VS", "Vaud": "VD", "Zug": "ZG", "Zürich": "ZH",
}

DB_PATH = Path("data/listings.db")


def main() -> None:
    with get_connection(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT listing_id, latitude, longitude FROM listings "
            "WHERE city IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL"
        ).fetchall()

    if not rows:
        print("No listings need enrichment.")
        return

    print(f"Reverse-geocoding {len(rows)} listings...")
    coords = [(r["latitude"], r["longitude"]) for r in rows]
    results = reverse_geocoder.search(coords, mode=1, verbose=False)

    updates: list[tuple[str | None, str | None, str | None, str]] = []
    for row, result in zip(rows, results):
        city = result.get("name") or None
        admin1 = result.get("admin1", "")
        canton = _ADMIN1_TO_CANTON.get(admin1)
        updates.append((city, canton, None, row["listing_id"]))

    with get_connection(DB_PATH) as conn:
        conn.executemany(
            "UPDATE listings SET city = ?, canton = ?, postal_code = ? WHERE listing_id = ?",
            updates,
        )
        conn.commit()

    enriched = sum(1 for u in updates if u[0] is not None)
    print(f"Done. Enriched {enriched}/{len(rows)} listings with city/canton.")


if __name__ == "__main__":
    main()
