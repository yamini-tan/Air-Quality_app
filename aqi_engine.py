"""
aqi_engine.py
=============
Core signal-processing and index logic of the AQMS, implemented exactly as
described in the revised paper (AQMS_Revised_Paper_final_v5):

  Sec 4.3  MQ-7 CO: ppm -> mg/m3 using the measured room T and P   (Eq. 11)
  Sec 4.4  PMS5003 humidity correction (Barkjohn et al.)           (Eq. 12)
  Sec 4.5  CPCB sub-index interpolation + max-operator AQI         (Eq. 13, 14, Table 3)
  Sec 4.7  Staged fan controller with hysteresis                   (Table 5)
           Purification metrics R%, k, CADR_eff, t50, t90          (Eq. 8, 18, 19)
  Sec 5.1  Range filtering, duplicate removal, moving-median spike filter,
           1-min / 15-min / 1-h / 8-h / 24-h aggregation

Everything here is pure Python / pandas so it is shared by app.py (live)
and train.py (offline model training).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import numpy as np
import pandas as pd

# =====================================================================
# SENSOR METADATA (Table 2 of the paper) - also used by the dashboard
# =====================================================================
SENSORS = [
    {"key": "pm25", "sensor": "PMS5003", "quantity": "PM2.5 (fine particulate matter)",
     "unit": "µg/m³", "role": "CPCB pollutant", "icon": "🌫️",
     "extra": [{"key": "pm10", "label": "PM10", "unit": "µg/m³"},
               {"key": "pm25_raw", "label": "PM2.5 raw (uncorrected)", "unit": "µg/m³"}]},
    {"key": "co_mg", "sensor": "MQ-7", "quantity": "Carbon monoxide (CO)",
     "unit": "mg/m³", "role": "CPCB pollutant", "icon": "🔥",
     "extra": [{"key": "co_ppm", "label": "CO", "unit": "ppm"}]},
    {"key": "mq135", "sensor": "MQ-135", "quantity": "Broadband gas index (NH₃, NOx, benzene, smoke)",
     "unit": "ADC a.u.", "role": "Supplementary", "icon": "🧪", "extra": []},
    {"key": "gpy10", "sensor": "GP2Y1010AU0F", "quantity": "Optical dust density",
     "unit": "ADC a.u.", "role": "Supplementary", "icon": "💨", "extra": []},
    {"key": "temperature", "sensor": "DHT22", "quantity": "Temperature",
     "unit": "°C", "role": "Context / correction", "icon": "🌡️", "extra": []},
    {"key": "humidity", "sensor": "DHT22", "quantity": "Relative humidity",
     "unit": "%", "role": "Context / correction", "icon": "💧", "extra": []},
    {"key": "pressure", "sensor": "BMP280", "quantity": "Barometric pressure",
     "unit": "hPa", "role": "Context / CO conversion", "icon": "🧭", "extra": []},
]

# =====================================================================
# CPCB BREAKPOINTS (Table 3)  (B_LO, B_HI, I_LO, I_HI)
# The last band is open ended in the CPCB table; the upper concentration
# bound used for interpolation is the one used by the CPCB AQI calculator.
# =====================================================================
CPCB_BREAKPOINTS = {
    "pm25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
             (91, 120, 201, 300), (121, 250, 301, 400), (251, 380, 401, 500)],
    "co":   [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
             (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
}

CATEGORIES = [  # (upper AQI, name, colour)
    (50, "Good", "#16A34A"),
    (100, "Satisfactory", "#65A30D"),
    (200, "Moderately Polluted", "#CA8A04"),
    (300, "Poor", "#EA580C"),
    (400, "Very Poor", "#DC2626"),
    (500, "Severe", "#991B1B"),
]


def sub_index(pollutant: str, conc: Optional[float]) -> Optional[float]:
    """Eq. (13): linear interpolation between the bracketing CPCB breakpoints."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return None
    conc = max(0.0, float(conc))
    bands = CPCB_BREAKPOINTS[pollutant]
    for b_lo, b_hi, i_lo, i_hi in bands:
        if conc <= b_hi:
            val = (i_hi - i_lo) / (b_hi - b_lo) * (conc - b_lo) + i_lo
            # published breakpoints have small gaps (30 -> 31); clamp into band
            return float(min(max(val, i_lo), i_hi))
    return 500.0


DEVICE_AQI_LABEL = "Dust (device AQI)"


def aqi_from(pm25: Optional[float], co_mg: Optional[float], device_aqi: Optional[float] = None) -> dict:
    """Eq. (14): AQI = max sub-index, dominant pollutant = argmax.
    device_aqi: fallback for nodes without PMS5003 / MQ-7 (old firmware reports a
    GP2Y1010-based AQI); used only when neither PM2.5 nor CO is available."""
    subs = {"PM2.5": sub_index("pm25", pm25), "CO": sub_index("co", co_mg)}
    valid = {k: v for k, v in subs.items() if v is not None}
    if not valid and device_aqi is not None and not (isinstance(device_aqi, float) and math.isnan(device_aqi)):
        aqi = int(round(min(max(float(device_aqi), 0), 500)))
        name, color = category(aqi)
        return {"aqi": aqi, "dominant": DEVICE_AQI_LABEL, "category": name, "color": color,
                "sub_pm25": None, "sub_co": None}
    if not valid:
        return {"aqi": None, "dominant": None, "category": "No data",
                "color": "#94a3b8", "sub_pm25": None, "sub_co": None}
    dominant = max(valid, key=valid.get)
    aqi = int(round(valid[dominant]))
    name, color = category(aqi)
    return {"aqi": aqi, "dominant": dominant, "category": name, "color": color,
            "sub_pm25": None if subs["PM2.5"] is None else round(subs["PM2.5"], 1),
            "sub_co": None if subs["CO"] is None else round(subs["CO"], 1)}


def category(aqi: Optional[float]):
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return "No data", "#94a3b8"
    for upper, name, color in CATEGORIES:
        if aqi <= upper:
            return name, color
    return "Severe", "#991B1B"


# =====================================================================
# SENSOR CORRECTION / CONVERSION
# =====================================================================
M_CO = 28.01      # g/mol
R_GAS = 8.314     # J/(mol K)
PM_ALPHA, PM_BETA, PM_GAMMA = 0.524, -0.0862, 5.75   # Barkjohn et al. [9]


def co_ppm_to_mg(co_ppm: float, temp_c: float, pressure_hpa: float) -> float:
    """Eq. (11) with the measured room temperature and pressure."""
    if co_ppm is None:
        return None
    t_k = (temp_c if temp_c not in (None, 0) else 25.0) + 273.15
    p_pa = (pressure_hpa if pressure_hpa and pressure_hpa > 300 else 1013.25) * 100.0
    return co_ppm * M_CO * p_pa / (R_GAS * t_k) * 1e-3


def correct_pm25(raw: float, rh: float) -> float:
    """Eq. (12): C_corr = alpha*C_raw + beta*RH + gamma (clamped at 0)."""
    if raw is None:
        return None
    rh = rh if rh is not None else 50.0
    return max(0.0, PM_ALPHA * raw + PM_BETA * rh + PM_GAMMA)


# =====================================================================
# DATA FILTERING (Sec 5.1)
# =====================================================================
LIMITS = {
    "pm25_raw": (0, 1000), "pm10": (0, 1000), "co_ppm": (0, 2000),
    "mq135": (0, 4095), "gpy10": (0, 4095),
    "temperature": (-10, 60), "humidity": (0, 100), "pressure": (300, 1100),
    "aqi_device": (0, 500),
}


def within_limits(key, value) -> bool:
    lo, hi = LIMITS.get(key, (-math.inf, math.inf))
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and lo <= value <= hi


class SpikeFilter:
    """Moving-median filter (window 5) that removes isolated spikes."""

    def __init__(self, window: int = 5):
        self.buffers: dict[str, deque] = {}
        self.window = window

    def __call__(self, key: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        buf = self.buffers.setdefault(key, deque(maxlen=self.window))
        buf.append(value)
        return float(median(buf))


# =====================================================================
# STAGED CONTROLLER (Table 5)
# =====================================================================
STAGE_DUTY = {0: 0, 1: 50, 2: 100}       # percent
STAGE_NAME = {0: "Standby", 1: "Low / pre-emptive", 2: "Full purification"}
T_STAGE1_LO, T_STAGE2, T_FORECAST, T_STANDBY = 101, 130, 130, 90
STANDBY_HOLD_S = 600                      # 10 min below 90 before standby


@dataclass
class FanController:
    stage: int = 0
    below_since: Optional[float] = None
    reason: str = "start-up"

    def update(self, aqi_now: Optional[float], aqi_forecast: Optional[float], now_s: float) -> int:
        if aqi_now is None:
            return self.stage
        fc = aqi_forecast if aqi_forecast is not None else -1

        # hysteresis timer for the return to standby
        if aqi_now < T_STANDBY:
            if self.below_since is None:
                self.below_since = now_s
        else:
            self.below_since = None

        if aqi_now >= T_STAGE2:
            self.stage, self.reason = 2, f"AQI {aqi_now:.0f} ≥ {T_STAGE2}"
        elif aqi_now >= T_STAGE1_LO:
            self.stage, self.reason = 1, f"{T_STAGE1_LO} ≤ AQI {aqi_now:.0f} < {T_STAGE2}"
        elif fc >= T_FORECAST:
            self.stage, self.reason = 1, f"pre-emptive: forecast {fc:.0f} ≥ {T_FORECAST}"
        else:
            if self.stage == 2:
                self.stage, self.reason = 1, "stepping down from full speed"
            if self.stage > 0 and self.below_since is not None and now_s - self.below_since >= STANDBY_HOLD_S:
                self.stage, self.reason = 0, f"AQI < {T_STANDBY} for 10 min"
            elif self.stage > 0:
                held = 0 if self.below_since is None else now_s - self.below_since
                self.reason = f"hysteresis hold (below {T_STANDBY} for {held/60:.1f}/10 min)"
            elif self.stage == 0:
                self.reason = f"AQI {aqi_now:.0f} below Stage 1 threshold"
        return self.stage

    @property
    def duty(self) -> int:
        return STAGE_DUTY[self.stage]

    @property
    def pwm(self) -> int:
        return round(self.duty * 255 / 100)


# =====================================================================
# PURIFICATION (Eq. 8, 18, 19 ; Table 9)
# =====================================================================
def theoretical_purification(volume_m3: float, eta: float = 0.9995, q_eff_m3_min: float = 0.15,
                             k_nat_per_min: float = 0.0, minutes: float = 30.0) -> dict:
    k = k_nat_per_min + eta * q_eff_m3_min / volume_m3          # min^-1
    return {
        "volume_m3": volume_m3,
        "k_per_min": round(k, 4),
        "k_per_h": round(k * 60, 3),
        "cadr_m3_h": round(eta * q_eff_m3_min * 60, 2),
        "t50_min": round(math.log(2) / k, 1),
        "t90_min": round(math.log(10) / k, 1),
        "reduction_pct": round((1 - math.exp(-k * minutes)) * 100, 1),
        "minutes": minutes,
    }


def decay_constant(times_min: np.ndarray, conc: np.ndarray, c_bg: float = 0.0) -> Optional[float]:
    """Slope of ln(C - C_bg) vs time (per min), following AHAM AC-1."""
    y = np.asarray(conc, float) - c_bg
    t = np.asarray(times_min, float)
    ok = y > 0
    if ok.sum() < 3:
        return None
    slope = np.polyfit(t[ok], np.log(y[ok]), 1)[0]
    return float(-slope)


# =====================================================================
# TIME-SERIES CONSTRUCTION (shared by train.py and app.py)
# =====================================================================
FEATURES = ["aqi", "pm25", "co_mg", "mq135", "gpy10", "temperature", "humidity", "pressure",
            "hour_sin", "hour_cos"]
STEP = "15min"
WINDOW = 96      # w = 24 h of 15-min steps
HORIZON = 48     # h = 12 h


def vectorised_aqi(pm25: pd.Series, co: pd.Series) -> pd.Series:
    sp = pm25.map(lambda v: sub_index("pm25", v)).astype(float)   # None -> NaN
    sc = co.map(lambda v: sub_index("co", v)).astype(float)
    return pd.concat([sp, sc], axis=1).max(axis=1, skipna=True)


def build_timeseries(records: pd.DataFrame) -> pd.DataFrame:
    """
    records: one row per cleaned reading with columns
      created_at, pm25 (corrected), co_mg, mq135, gpy10, temperature, humidity, pressure
    returns the 15-min feature frame used by every forecasting model, with the
    target series 'aqi' = hourly indicative AQI (1-h rolling means, Sec 4.5).
    Gaps are left as NaN (not interpolated), as stated in Sec 5.1.
    """
    df = records.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.drop_duplicates(subset="created_at").set_index("created_at").sort_index()
    cols = ["pm25", "co_mg", "mq135", "gpy10", "temperature", "humidity", "pressure"]
    if "aqi_device" not in df:
        df["aqi_device"] = np.nan
    q = df[cols + ["aqi_device"]].astype(float).resample(STEP).mean()
    pm1h = q["pm25"].rolling(4, min_periods=2).mean()
    co1h = q["co_mg"].rolling(4, min_periods=2).mean()
    dev1h = q["aqi_device"].rolling(4, min_periods=2).mean()
    # CPCB AQI from PM2.5/CO; device AQI only where neither is available (old firmware)
    q["aqi"] = vectorised_aqi(pm1h, co1h).fillna(dev1h)
    # a sensor that is not installed at all (every value missing) becomes a constant 0 feature
    active = q["aqi"].notna()
    for c in cols:
        if q.loc[active, c].isna().all():
            q.loc[active, c] = 0.0
    hour = q.index.hour + q.index.minute / 60.0
    q["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    q["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return q[FEATURES]


def tabular_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Lagged feature vector for Random Forest / XGBoost (Table 4)."""
    X = pd.DataFrame(index=ts.index)
    for lag in [0, 1, 2, 4, 8, 16, 24, 48, 72, 95]:
        X[f"aqi_lag{lag}"] = ts["aqi"].shift(lag)
    for win in [4, 16, 48, 96]:
        X[f"aqi_mean{win}"] = ts["aqi"].rolling(win, min_periods=max(2, win // 2)).mean()
    X["aqi_std16"] = ts["aqi"].rolling(16, min_periods=4).std()
    for c in ["pm25", "co_mg", "mq135", "gpy10", "temperature", "humidity", "pressure"]:
        X[c] = ts[c]
        X[f"{c}_lag4"] = ts[c].shift(4)
        X[f"{c}_mean16"] = ts[c].rolling(16, min_periods=4).mean()
    X["hour_sin"], X["hour_cos"] = ts["hour_sin"], ts["hour_cos"]
    # hour of the *target* time (t + 12 h) helps the direct strategy
    idx_t = ts.index + pd.Timedelta(minutes=15 * HORIZON)
    ht = idx_t.hour + idx_t.minute / 60.0
    X["tgt_hour_sin"] = np.sin(2 * np.pi * ht / 24)
    X["tgt_hour_cos"] = np.cos(2 * np.pi * ht / 24)
    return X
