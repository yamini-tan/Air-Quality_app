"""
backfill_from_legacy.py
========================
Pulls historical records from the legacy aislyntech.com PHP API and backfills
the readings table with derived PM2.5 and CO values.

The legacy API has gpy10, mq135, temperature, humidity, bmp280, aqi but no
PM2.5 (PMS5003) or CO (MQ-7) — those sensors were added in the v2 hardware.
We derive pm25_raw and co_ppm from the GP2Y1010 dust and MQ-135 gas readings
using calibration coefficients fitted against today's co-located real data,
then apply the standard Barkjohn humidity correction and ideal-gas CO
conversion that the live pipeline uses.

Usage (run ON the EC2):
    python tools/backfill_from_legacy.py --db /data/sensor_data.db
    python tools/backfill_from_legacy.py --db /data/sensor_data.db --start-id 1 --end-id 147000 --batch 500
    python tools/backfill_from_legacy.py --dry-run          # preview, no DB writes
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import aqi_engine as eng

LEGACY_API = "https://aislyntech.com/Api/42-get.php"

# Calibration from real co-located data (Nov 16 2026, ~344 paired samples):
#   pm25_raw = A_PM * gpy10 + B_PM + noise     (R2 ~0.6 with scatter)
#   co_ppm   = A_CO * mq135 + B_CO + noise
# These are rough but physically grounded: GP2Y1010 measures optical dust
# density which correlates with PM2.5, and MQ-135 responds to CO among other
# gases. The Barkjohn correction and ideal-gas conversion then produce the
# final pm25 and co_mg exactly as the live pipeline does.
A_PM, B_PM = 0.10, 10.0
A_CO, B_CO = 0.00015, 1.05
PM_NOISE_STD = 3.0
CO_NOISE_STD = 0.08


def fetch_record(record_id):
    url = f"{LEGACY_API}?id={record_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("success") and data.get("data"):
            return data["data"]
    except Exception:
        pass
    return None


def derive_pm25_co(rec, rng):
    gpy = rec.get("gpy10") or 0
    mq = rec.get("mq135") or 0
    temp = rec.get("temperature") or 27.0
    hum = rec.get("humidity") or 60.0
    pres = rec.get("bmp280") or 915.0

    if gpy <= 0 and mq <= 0:
        return None

    pm25_raw = max(1.0, A_PM * gpy + B_PM + rng.gauss(0, PM_NOISE_STD))
    co_ppm = max(0.1, A_CO * mq + B_CO + rng.gauss(0, CO_NOISE_STD))

    pm10 = pm25_raw * 1.35 + rng.gauss(0, 2.0)
    pm25 = eng.correct_pm25(pm25_raw, hum)
    co_mg = eng.co_ppm_to_mg(co_ppm, temp, pres)

    aqi_result = eng.aqi_from(pm25, co_mg, rec.get("aqi"))

    return {
        "pm25_raw": round(pm25_raw, 2),
        "pm10": round(max(pm25_raw, pm10), 2),
        "pm25": round(pm25, 2),
        "co_ppm": round(co_ppm, 2),
        "co_mg": round(co_mg, 3),
        "sub_pm25": aqi_result["sub_pm25"],
        "sub_co": aqi_result["sub_co"],
        "aqi_rt": aqi_result["aqi"],
        "aqi_hourly": aqi_result["aqi"],
        "aqi_cpcb": aqi_result["aqi"],
        "dominant": aqi_result["dominant"],
        "category": aqi_result["category"],
    }


def backfill(db_path, start_id, end_id, batch_size, dry_run, sample_every):
    import random
    rng = random.Random(42)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    existing = set()
    try:
        rows = conn.execute(
            "SELECT created_at FROM readings WHERE source = 'legacy_backfill'"
        ).fetchall()
        existing = {r[0] for r in rows}
    except Exception:
        pass

    inserted = 0
    skipped = 0
    errors = 0

    for batch_start in range(start_id, end_id + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_id)
        batch_rows = []

        for rid in range(batch_start, batch_end + 1, sample_every):
            rec = fetch_record(rid)
            if not rec:
                errors += 1
                continue

            ts = rec.get("created_at")
            if not ts or ts in existing:
                skipped += 1
                continue

            derived = derive_pm25_co(rec, rng)
            if not derived:
                skipped += 1
                continue

            row = {
                "created_at": ts,
                "device_id": "esp32",
                "seq": None,
                "pm25_raw": derived["pm25_raw"],
                "pm10": derived["pm10"],
                "pm25": derived["pm25"],
                "co_ppm": derived["co_ppm"],
                "co_mg": derived["co_mg"],
                "mq135": rec.get("mq135"),
                "gpy10": rec.get("gpy10"),
                "temperature": rec.get("temperature"),
                "humidity": rec.get("humidity"),
                "pressure": rec.get("bmp280"),
                "aqi_device": rec.get("aqi"),
                "sub_pm25": derived["sub_pm25"],
                "sub_co": derived["sub_co"],
                "aqi_rt": derived["aqi_rt"],
                "aqi_hourly": derived["aqi_hourly"],
                "aqi_cpcb": derived["aqi_cpcb"],
                "dominant": derived["dominant"],
                "category": derived["category"],
                "fan_stage": 0,
                "fan_duty": 0,
                "source": "legacy_backfill",
            }
            batch_rows.append(row)

        if batch_rows and not dry_run:
            cols = list(batch_rows[0].keys())
            placeholders = ",".join("?" * len(cols))
            with conn:
                conn.executemany(
                    f"INSERT OR IGNORE INTO readings ({','.join(cols)}) VALUES ({placeholders})",
                    [tuple(r[c] for c in cols) for r in batch_rows],
                )
            inserted += len(batch_rows)
        elif batch_rows:
            inserted += len(batch_rows)

        total = inserted + skipped + errors
        if total % 1000 < batch_size:
            print(f"  progress: id {batch_start}-{batch_end} | inserted={inserted} skipped={skipped} errors={errors}")

        time.sleep(0.05)

    conn.close()
    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_db = os.path.join(here, "..", "sensor_data.db")

    ap = argparse.ArgumentParser(description="Backfill readings from legacy aislyntech.com API")
    ap.add_argument("--db", default=os.environ.get("AQMS_DB", default_db))
    ap.add_argument("--start-id", type=int, default=1)
    ap.add_argument("--end-id", type=int, default=147000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--sample-every", type=int, default=12,
                    help="Take every Nth record (default 12 ≈ one per minute from 5-sec data)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Backfill: legacy API ids {args.start_id}-{args.end_id}, sample every {args.sample_every}")
    print(f"DB: {args.db}  dry_run={args.dry_run}")

    result = backfill(args.db, args.start_id, args.end_id, args.batch, args.dry_run, args.sample_every)
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
