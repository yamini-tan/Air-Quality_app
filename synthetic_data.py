"""
synthetic_data.py
=================
Generates a SYNTHETIC indoor data set (1-min resolution) with the same
channels as the new hardware (PMS5003, MQ-7, MQ-135, GP2Y1010, DHT22, BMP280).

PURPOSE: only to bootstrap the five forecasting models so the dashboard works
before enough real PM2.5/CO data has been logged. It is NOT measurement data
and must NOT be reported in the paper. Once the node has logged >= 7 days,
run `python train.py --source db` to retrain on the real readings.
"""

import numpy as np
import pandas as pd

from aqi_engine import correct_pm25, co_ppm_to_mg


def generate(days: int = 30, seed: int = 7, start: str = "2026-08-01 00:00:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    t = pd.date_range(start, periods=n, freq="1min")
    hour = t.hour.values + t.minute.values / 60.0
    day = np.arange(n) / 1440.0

    # slow multi-day background (weather / ventilation) as AR(1)
    drift = np.zeros(n)
    e = rng.normal(0, 0.02, n)
    for i in range(1, n):
        drift[i] = 0.9995 * drift[i - 1] + e[i]
    drift = drift / (np.abs(drift).max() + 1e-9)

    pm_raw = 32 + 10 * drift + 8 * np.sin(2 * np.pi * (hour - 20) / 24) + rng.normal(0, 2.0, n)
    co_ppm = 1.2 + 0.4 * drift + rng.normal(0, 0.15, n)

    def add_event(center_min, amp_pm, amp_co, rise=8, tau=60):
        idx = np.arange(n)
        dt = idx - center_min
        shape = np.where(dt < 0, 0, np.where(dt < rise, dt / rise, np.exp(-(dt - rise) / tau)))
        return amp_pm * shape, amp_co * shape

    for d in range(days):
        base = d * 1440
        weekend = (pd.Timestamp(start) + pd.Timedelta(days=d)).dayofweek >= 5
        # morning cooking
        if rng.random() < 0.75:
            c = base + int(rng.normal(7.5 * 60, 25))
            a, b = add_event(c, rng.uniform(40, 140), rng.uniform(2, 10), tau=rng.uniform(40, 80))
            pm_raw += a; co_ppm += b
        # evening cooking (strongest)
        if rng.random() < 0.92:
            c = base + int(rng.normal(19.8 * 60, 30))
            a, b = add_event(c, rng.uniform(90, 330) * (1.2 if weekend else 1.0),
                             rng.uniform(4, 18), tau=rng.uniform(60, 110))
            pm_raw += a; co_ppm += b
        # incense / mosquito coil (CO-rich combustion event) ~2 per week
        if rng.random() < 0.3:
            c = base + int(rng.uniform(17, 22) * 60)
            a, b = add_event(c, rng.uniform(20, 60), rng.uniform(15, 40), rise=15, tau=90)
            pm_raw += a; co_ppm += b

    pm_raw = np.clip(pm_raw, 1, 900)
    pm10 = pm_raw * rng.uniform(1.25, 1.45, n)
    co_ppm = np.clip(co_ppm, 0, 500)
    temp = 27.5 + 1.6 * np.sin(2 * np.pi * (hour - 15) / 24) + 0.5 * drift + rng.normal(0, 0.1, n)
    rh = 68 - 6 * np.sin(2 * np.pi * (hour - 15) / 24) + 4 * drift + rng.normal(0, 0.6, n)
    press = 912.7 + 0.8 * np.sin(4 * np.pi * (hour - 10) / 24) + 0.5 * np.sin(2 * np.pi * day / 5) + rng.normal(0, 0.05, n)
    mq135 = 1450 + 1.6 * pm_raw + 9 * co_ppm + rng.normal(0, 12, n)
    gpy10 = 330 + 1.9 * pm_raw + rng.normal(0, 15, n)

    df = pd.DataFrame({
        "created_at": t, "pm25_raw": pm_raw, "pm10": pm10, "co_ppm": co_ppm,
        "mq135": np.clip(mq135, 0, 4095), "gpy10": np.clip(gpy10, 0, 4095),
        "temperature": temp, "humidity": np.clip(rh, 0, 100), "pressure": press,
    })
    df["pm25"] = [correct_pm25(p, h) for p, h in zip(df.pm25_raw, df.humidity)]
    df["co_mg"] = [co_ppm_to_mg(c, tt, p) for c, tt, p in zip(df.co_ppm, df.temperature, df.pressure)]
    # knock out a few short blocks of samples in short gaps (like real Wi-Fi drop-outs)
    gaps = rng.random(n) < 0.0002
    for g in np.where(gaps)[0]:
        df.loc[g:g + rng.integers(5, 60), ["pm25", "co_mg"]] = np.nan
    return df


if __name__ == "__main__":
    d = generate()
    d.to_csv("synthetic_dataset.csv", index=False)
    print(d.describe().T)
