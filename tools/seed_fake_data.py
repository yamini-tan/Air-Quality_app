"""
Fills a TEST database with realistic fake history so every dashboard chart has data.
Never run this against the production database (/data/sensor_data.db).

  python tools/seed_fake_data.py --db test.db --days 14              # new hardware (PM2.5 + CO)
  python tools/seed_fake_data.py --db test.db --days 14 --legacy     # current firmware (dust AQI only)

Rows older than RETENTION_DAYS are then moved into aqi_history by retention.py
(run with --cleanup), which exercises the lifetime-AQI chart as well.
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aqi_engine as eng  # noqa: E402
import retention  # noqa: E402
from synthetic_data import generate  # noqa: E402

IST_NOW = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--legacy", action="store_true", help="current firmware: no PM2.5/CO, device AQI from dust")
    ap.add_argument("--cleanup", action="store_true", help="run retention afterwards (10-day raw window)")
    a = ap.parse_args()
    if os.path.abspath(a.db).replace("\\", "/").startswith("/data/"):
        raise SystemExit("Refusing to seed the production database.")

    # import app only to create the schema in this DB
    os.environ["AQMS_DB"] = a.db
    import app  # noqa: F401

    df = generate(days=a.days, start=str(IST_NOW - pd.Timedelta(days=a.days)))
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.set_index("created_at")
    if a.legacy:
        # GP2Y1010-based AQI exactly like the user's firmware (0..500 piecewise map of the ADC value)
        g = df["gpy10"]
        dev = np.select([g <= 500, g <= 1000, g <= 1800, g <= 2500],
                        [g / 500 * 50, 51 + (g - 501) / 499 * 49, 101 + (g - 1001) / 799 * 99, 201 + (g - 1801) / 699 * 99],
                        301 + (g - 2501) / 1594 * 199)
        df["aqi_device"] = np.clip(np.round(dev), 0, 500)
        df[["pm25", "pm25_raw", "pm10", "co_ppm", "co_mg"]] = np.nan
        rt = hourly = cpcb = df["aqi_device"]
        hourly = df["aqi_device"].rolling("60min").mean()
        cpcb = df["aqi_device"].rolling("24h").mean()
        dominant = np.where(df["aqi_device"].notna(), eng.DEVICE_AQI_LABEL, None)
        device, source = "old-node-sim", "legacy_api"
    else:
        df["aqi_device"] = np.nan
        rt = eng.vectorised_aqi(df["pm25"], df["co_mg"])
        hourly = eng.vectorised_aqi(df["pm25"].rolling("60min").mean(), df["co_mg"].rolling("60min").mean())
        cpcb = eng.vectorised_aqi(df["pm25"].rolling("24h").mean(), df["co_mg"].rolling("8h").mean())
        sp = df["pm25"].map(lambda v: eng.sub_index("pm25", v)).astype(float)
        sc = df["co_mg"].map(lambda v: eng.sub_index("co", v)).astype(float)
        df["sub_pm25"], df["sub_co"] = sp.round(1), sc.round(1)
        dominant = np.where(sp.fillna(-1) >= sc.fillna(-1), "PM2.5", "CO")
        device, source = "simulator", "esp32"

    df["aqi_rt"], df["aqi_hourly"], df["aqi_cpcb"] = rt.round(), hourly.round(), cpcb.round()
    df["dominant"] = dominant
    df["category"] = [eng.category(v)[0] if pd.notna(v) else None for v in df["aqi_rt"]]
    stage = np.select([df["aqi_rt"] >= 130, df["aqi_rt"] >= 101], [2, 1], 0)
    df["fan_stage"], df["fan_duty"] = stage, np.vectorize(eng.STAGE_DUTY.get)(stage)
    df["device_id"], df["source"] = device, source

    out = df.reset_index()
    out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["created_at", "device_id", "source", "pm25_raw", "pm10", "pm25", "co_ppm", "co_mg", "mq135", "gpy10",
            "temperature", "humidity", "pressure", "aqi_device", "sub_pm25", "sub_co", "aqi_rt", "aqi_hourly",
            "aqi_cpcb", "dominant", "category", "fan_stage", "fan_duty"]
    cols = [c for c in cols if c in out]
    out = out[cols].astype(object).where(out[cols].notna(), None)
    conn = sqlite3.connect(a.db)
    with conn:
        conn.executemany(f"INSERT INTO readings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         out.itertuples(index=False, name=None))
    conn.close()
    print(f"inserted {len(out)} rows ({a.days} days, {'legacy dust-AQI' if a.legacy else 'PM2.5 + CO'}) into {a.db}")
    if a.cleanup:
        print("retention:", retention.cleanup(a.db, float(os.environ.get("RETENTION_DAYS", 10))))


if __name__ == "__main__":
    main()
