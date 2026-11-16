"""
AQMS - Flask backend (AWS)
==========================
Receives readings from the ESP32 (PMS5003, MQ-7, MQ-135, GP2Y1010, DHT22, BMP280),
cleans and corrects them, computes the CPCB-method indoor AQI, runs the five
12-hour forecasting models, drives the staged fan controller and serves the
dashboard.

Data paths (either or both):
  1. ESP32 -> POST http://<VM>:5000/api/ingest            (recommended, direct)
  2. ESP32 -> aislyntech PHP API -> this server polls it   (set LEGACY_POLL=1)

Environment variables (all optional):
  AQMS_API_KEY        shared key the ESP32 sends in header X-API-Key (default below)
  AQMS_DB             sqlite file (default sensor_data.db)
  LEGACY_POLL         1 to poll the old PHP API as well (default 0)
  LEGACY_API_URL      default https://aislyntech.com/Api/42-get.php
  ROOM_VOLUME_M3      room volume for purification maths (default 30)
  ENCLOSURE_VOLUME_M3 enclosure volume (default 1)
  FAN_Q_M3_MIN        effective fan airflow (default 0.15)
  CO_EVENT_PPM        CO event alarm threshold (default 20 = MQ-7 lower range)
  AQI_BAD             status LED red above this AQI (default 100)
  CO_BAD_MG           status LED red above this 1-min CO mean in mg/m3 (default 2.0)
  DYNAMODB_TABLE      if set, every record is mirrored to this DynamoDB table
  AWS_REGION          region for DynamoDB (default ap-south-1)
  RETENTION_DAYS      raw readings older than this are deleted (default 365); their AQI is
                      archived in table aqi_history and kept forever (see retention.py)
  CLEANUP_EVERY_H     how often the retention cleanup runs (default 6)

Models: trained_models/ is reloaded automatically when auto_retrain.py installs new models.

Two-ESP setup: ESP-A sends PM2.5 ("aqms-pm"), ESP-B sends CO + climate and drives the
fan and LEDs ("aqms-env"); see esp32_firmware/README.md.
"""

import json
import math
import os
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request

import aqi_engine as eng
import retention
from forecasting import MODEL_DIR, ForecastBundle

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

API_KEY = os.environ.get("AQMS_API_KEY", "aqms-esp32-key")
DATABASE = os.environ.get("AQMS_DB", os.path.join(BASE, "sensor_data.db"))
LEGACY_POLL = os.environ.get("LEGACY_POLL", "0") == "1"
LEGACY_API_URL = os.environ.get("LEGACY_API_URL", "https://aislyntech.com/Api/42-get.php")
LEGACY_MAX_AGE_S = float(os.environ.get("LEGACY_MAX_AGE_S", 120))   # older PHP rows = sensor offline
ROOM_VOLUME = float(os.environ.get("ROOM_VOLUME_M3", 30))
ENCLOSURE_VOLUME = float(os.environ.get("ENCLOSURE_VOLUME_M3", 1))
FAN_Q = float(os.environ.get("FAN_Q_M3_MIN", 0.15))
CO_EVENT_PPM = float(os.environ.get("CO_EVENT_PPM", 20))
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "")
RETENTION_DAYS = float(os.environ.get("RETENTION_DAYS", 365))
CLEANUP_EVERY_H = float(os.environ.get("CLEANUP_EVERY_H", 6))
OFFLINE_AFTER_S = 30
AQI_BAD = float(os.environ.get("AQI_BAD", 100))          # red LED above this AQI (CPCB: > 100 = moderately polluted)
CO_BAD_MG = float(os.environ.get("CO_BAD_MG", 2.0))      # red LED above this 1-min CO mean (CPCB CO sub-index > 100)
ENV_MAX_AGE_S = 120        # temperature / humidity / pressure from another board are reused for this long
DEVICE_SHOW_DAYS = 3       # boards seen within this many days are listed in /api/status
# sensor boards of the two-ESP setup (any other device_id is shown by its id)
DEVICE_ROLES = {"aqms-pm": "PM2.5 sensor", "aqms-env": "CO, climate sensors + fan",
                "esp32": "Sensor node (old firmware)"}
SENSOR_FIELDS = ["pm25_raw", "pm10", "pm25", "co_ppm", "co_mg", "mq135", "gpy10",
                 "temperature", "humidity", "pressure", "aqi_device"]

IST = timezone(timedelta(hours=5, minutes=30))
TS_FMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------
# shared state
# ---------------------------------------------------------------------
lock = threading.Lock()
spike = eng.SpikeFilter(window=5)
controller = eng.FanController()
state = {
    "last_seen": 0.0,
    "last_seq": {},            # device_id -> last seq (duplicate removal per board)
    "env": {},                 # latest temperature / humidity / pressure from any board (+ time)
    "mirror_seen": 0.0,        # last direct frame from a board that also writes the PHP API
    "led": "green",            # status LED command: red when AQI or CO is bad
    "co_bad": False,
    "forecast": None,          # dict from ForecastBundle.predict
    "forecast_at": None,
    "trend": None,             # per-channel ARIMA trend forecast (Table 6)
    "legacy_last_id": None,
    "legacy_error": None,
    "last_cleanup": None,
}
bundle = None
bundle_mtime = None


def now_ist():
    return datetime.now(IST)


# =====================================================================
# DATABASE
# =====================================================================
COLUMNS = {
    "created_at": "TEXT", "device_id": "TEXT", "seq": "INTEGER",
    "pm25_raw": "REAL", "pm10": "REAL", "pm25": "REAL",
    "co_ppm": "REAL", "co_mg": "REAL",
    "mq135": "REAL", "gpy10": "REAL",
    "temperature": "REAL", "humidity": "REAL", "pressure": "REAL",
    "aqi_device": "REAL",  # AQI reported by the node (fallback when PM2.5/CO are missing)
    "sub_pm25": "REAL", "sub_co": "REAL",
    "aqi_rt": "INTEGER", "aqi_hourly": "INTEGER", "aqi_cpcb": "INTEGER",
    "dominant": "TEXT", "category": "TEXT",
    "fan_stage": "INTEGER", "fan_duty": "INTEGER", "fan_reported": "TEXT",
    "forecast_aqi": "REAL", "forecast_model": "TEXT",
    "alert": "TEXT", "source": "TEXT",
}


def get_db():
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def create_database():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    cur.execute("PRAGMA table_info(readings)")
    have = {r["name"] for r in cur.fetchall()}
    for col, typ in COLUMNS.items():
        if col not in have:
            cur.execute(f"ALTER TABLE readings ADD COLUMN {col} {typ}")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_time ON readings(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_device ON readings(device_id, created_at)")
    retention.ensure_schema(conn)
    conn.commit()
    conn.execute("PRAGMA journal_mode=WAL")   # readers/cleanup don't block ingest
    conn.close()


def fnum(v):
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return None


def window_means(conn, seconds):
    since = (now_ist() - timedelta(seconds=seconds)).strftime(TS_FMT)
    r = conn.execute("SELECT AVG(pm25) pm, AVG(co_mg) co, COUNT(pm25) n_pm, COUNT(co_mg) n_co, "
                     "MIN(created_at) first FROM readings WHERE created_at >= ?", (since,)).fetchone()
    return r["pm"], r["co"], r["n_pm"], r["n_co"], r["first"]


def device_mean(conn, seconds):
    """Mean device-reported AQI over the window (old firmware without PMS5003 / MQ-7)."""
    since = (now_ist() - timedelta(seconds=seconds)).strftime(TS_FMT)
    return conn.execute("SELECT AVG(aqi_device) FROM readings WHERE created_at >= ?", (since,)).fetchone()[0]


# =====================================================================
# DYNAMODB MIRROR (optional, paper Sec 5.1)
# =====================================================================
_ddb = None


def mirror_dynamodb(record):
    global _ddb
    if not DYNAMODB_TABLE:
        return
    try:
        from decimal import Decimal
        if _ddb is None:
            import boto3
            _ddb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-south-1")).Table(DYNAMODB_TABLE)
        item = {k: (Decimal(str(round(v, 4))) if isinstance(v, float) else v)
                for k, v in record.items() if v is not None}
        item["device_id"] = item.get("device_id") or "esp32"
        _ddb.put_item(Item=item)
    except Exception as e:
        app.logger.warning("DynamoDB mirror failed: %s", e)


# =====================================================================
# CORE PIPELINE - one reading in, one fan command out
# =====================================================================
def process_reading(p: dict, source: str = "esp32") -> dict:
    """p: raw payload from the ESP32 (or legacy API)."""
    g = lambda *keys: next((fnum(p[k]) for k in keys if k in p and fnum(p[k]) is not None), None)

    raw = {
        "pm25_raw": g("pm25", "pm2_5", "pm25_raw", "PM25"),
        "pm10": g("pm10", "PM10"),
        "co_ppm": g("co_ppm", "co", "mq7", "CO"),
        "mq135": g("mq135", "MQ135"),
        "gpy10": g("gpy10", "GPY10", "dust"),
        "temperature": g("temperature", "temp"),
        "humidity": g("humidity", "hum", "rh"),
        "pressure": g("pressure", "bmp280", "BMP280"),
        "aqi_device": g("aqi", "AQI", "local_aqi"),
    }
    # 1) range filter: out-of-limit readings are dropped (marked as gaps)
    for k, v in list(raw.items()):
        if v is not None and not eng.within_limits(k, v):
            raw[k] = None

    seq = p.get("seq")
    device = str(p.get("device", p.get("device_id", "esp32")))

    with lock:
        # 2) duplicate removal (per board: two ESPs post in turn)
        if seq is not None and state["last_seq"].get(device) == seq:
            return {"duplicate": True, **command_payload()}
        state["last_seq"][device] = seq
        if str(p.get("mirror", "")).lower() in ("1", "true"):
            state["mirror_seen"] = time.time()

        # 3) moving-median spike filter
        pm_f = spike("pm25", raw["pm25_raw"])
        co_f = spike("co", raw["co_ppm"])

        # 4) correction and unit conversion. A board without a climate sensor (the PM2.5 ESP)
        #    borrows the latest temperature / humidity / pressure sent by the other board.
        now_s = time.time()
        for k in ("temperature", "humidity", "pressure"):
            if raw[k] is not None:
                state["env"][k] = (raw[k], now_s)
        env = {k: (v if now_s - t <= ENV_MAX_AGE_S else None) for k, (v, t) in state["env"].items()}
        temp = raw["temperature"] if raw["temperature"] is not None else env.get("temperature")
        rh = raw["humidity"] if raw["humidity"] is not None else env.get("humidity")
        pres = raw["pressure"] if raw["pressure"] is not None else env.get("pressure")
        pm25 = None if pm_f is None else round(eng.correct_pm25(pm_f, rh), 2)
        co_mg = None if co_f is None else round(eng.co_ppm_to_mg(co_f, temp, pres), 3)

        created_at = now_ist().strftime(TS_FMT)
        rec = {"created_at": created_at, "device_id": device, "seq": fnum(seq),
               **raw, "pm25": pm25, "co_mg": co_mg, "source": source,
               "fan_reported": str(p.get("fan", "")) or None}

        conn = get_db()
        try:
            cols = ",".join(rec.keys())
            cur = conn.execute(f"INSERT INTO readings ({cols}) VALUES ({','.join('?' * len(rec))})",
                               tuple(rec.values()))
            rid = cur.lastrowid

            # 5) aggregation: 1-min (control), 1-h (indicative), 8-h CO / 24-h PM (CPCB)
            pm1, co1, *_ = window_means(conn, 60)
            pmh, coh, *_ = window_means(conn, 3600)
            pm24, _, n24, _, first24 = window_means(conn, 86400)
            _, co8, _, n8, _ = window_means(conn, 28800)
            rt = eng.aqi_from(pm1, co1, device_mean(conn, 60))
            hourly = eng.aqi_from(pmh, coh, device_mean(conn, 3600))
            cpcb = eng.aqi_from(pm24, co8, device_mean(conn, 86400))

            # 6) staged control (Table 5) with current + forecast AQI
            fc = state["forecast"] or {}
            best = bundle.best_model if bundle else None
            fc_val = (fc.get("models", {}).get(best, {}) or {}).get("forecast") if best else None
            controller.update(rt["aqi"], fc_val, time.time())

            # 7) status LED: red when the whole AQI is bad OR the CO reading is bad
            co_bad = (co1 is not None and co1 > CO_BAD_MG) or (co_f is not None and co_f >= CO_EVENT_PPM)
            aqi_bad = rt["aqi"] is not None and rt["aqi"] > AQI_BAD
            state["co_bad"], state["led"] = co_bad, ("red" if (aqi_bad or co_bad) else "green")

            alerts = []
            if rt["aqi"] is not None and rt["aqi"] > 200:
                alerts.append(f"Air quality {rt['category']} (AQI {rt['aqi']})")
            if co_f is not None and co_f >= CO_EVENT_PPM:
                alerts.append(f"CO event: {co_f:.0f} ppm ({co_mg:.1f} mg/m³) - ventilate, the HEPA filter cannot remove CO")
            elif co_bad:
                alerts.append(f"CO is high ({co1:.1f} mg/m³) - open a window. The HEPA filter cleans particles, not CO gas")

            upd = {"sub_pm25": rt["sub_pm25"], "sub_co": rt["sub_co"], "aqi_rt": rt["aqi"],
                   "aqi_hourly": hourly["aqi"], "aqi_cpcb": cpcb["aqi"], "dominant": rt["dominant"],
                   "category": rt["category"], "fan_stage": controller.stage, "fan_duty": controller.duty,
                   "forecast_aqi": fc_val, "forecast_model": best, "alert": " | ".join(alerts) or None}
            conn.execute("UPDATE readings SET " + ",".join(f"{k}=?" for k in upd) + " WHERE id=?",
                         (*upd.values(), rid))
            conn.commit()
        finally:
            conn.close()
        state["last_seen"] = time.time()

    mirror_dynamodb({"id": rid, **rec, **upd})
    return {"id": rid, "aqi": rt["aqi"], "category": rt["category"], **command_payload()}


def command_payload():
    return {"ok": True, "fan_stage": controller.stage, "fan_duty": controller.duty,
            "pwm": controller.pwm, "fan_reason": controller.reason,
            "led": state["led"], "co_bad": state["co_bad"]}


# =====================================================================
# BACKGROUND WORKERS
# =====================================================================
def load_recent(hours=48):
    since = (now_ist() - timedelta(hours=hours)).strftime(TS_FMT)
    conn = get_db()
    try:
        df = pd.read_sql_query("SELECT created_at, pm25, co_mg, mq135, gpy10, temperature, humidity, pressure, aqi_device "
                               "FROM readings WHERE created_at >= ? ORDER BY id", conn, params=(since,))
    finally:
        conn.close()
    return df


def channel_trend(df):
    """Table 6: ARIMA trend forecast of each measured channel, 12 h ahead."""
    from statsmodels.tsa.arima.model import ARIMA
    out = {}
    if df.empty:
        return out
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"])
    d = d.set_index("created_at").resample("5min").mean()
    for col in ["pm25", "co_mg", "gpy10", "mq135", "temperature", "humidity", "pressure"]:
        s = d[col].dropna()
        if len(s) < 8:
            out[col] = None
            continue
        try:
            if s.nunique() == 1:
                out[col] = round(float(s.iloc[-1]), 2)
                continue
            r = ARIMA(s.values[-288:], order=(1, 0, 1), trend="c").fit()
            out[col] = round(float(r.forecast(144)[-1]), 2)
        except Exception:
            out[col] = round(float(s.tail(6).mean()), 2)
    return out


def models_mtime():
    try:
        return os.path.getmtime(os.path.join(MODEL_DIR, "training_metrics.json"))
    except OSError:
        return None


def forecast_worker():
    global bundle, bundle_mtime
    while True:
        try:
            # (re)load when auto_retrain.py has installed new models - no restart needed
            mt = models_mtime()
            if bundle is None or mt != bundle_mtime:
                bundle, bundle_mtime = ForecastBundle(MODEL_DIR), mt
                app.logger.warning("forecast models loaded (%s)", bundle.meta.get("data_source"))
            df = load_recent(48)
            ts = eng.build_timeseries(df) if not df.empty else None
            if ts is not None and ts["aqi"].notna().any():
                res = bundle.predict(ts)
                h = ts["aqi"].dropna().iloc[-96:]
                res["history"] = [{"t": i.strftime(TS_FMT), "aqi": round(float(v), 1)} for i, v in h.items()]
                trend = channel_trend(df)
                with lock:
                    state["forecast"], state["trend"] = res, trend
                    state["forecast_at"] = now_ist().strftime(TS_FMT)
        except Exception as e:
            app.logger.exception("forecast worker: %s", e)
        time.sleep(60)


def poll_legacy_once():
    """Fetch the latest row from the old aislyntech PHP API and ingest it if new.
    Returns (payload, result) - result is None when the row was already stored."""
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    r = requests.get(LEGACY_API_URL, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict) or not data:
        raise ValueError("Hardware connection error: No recent data found from ESP32.")
    rid = data.get("id") or json.dumps(data, sort_keys=True)
    result = None
    # the PHP API always returns its newest row, even when the sensor is off: skip stale rows
    try:
        age = (now_ist().replace(tzinfo=None) - datetime.strptime(str(data["created_at"]), TS_FMT)).total_seconds()
    except (KeyError, ValueError):
        age = 0
    if age > LEGACY_MAX_AGE_S:
        state["legacy_error"] = f"Sensor offline: newest row in the PHP API is from {data['created_at']}"
        return data, None
    # a board that posts here directly AND mirrors to the PHP API would be stored twice
    mirrored = time.time() - state["mirror_seen"] < LEGACY_MAX_AGE_S
    if rid != state["legacy_last_id"]:
        state["legacy_last_id"] = rid
        if not mirrored:
            result = process_reading(data, source="legacy_api")
    state["legacy_error"] = None
    return data, result


def legacy_poller():
    """Optional: pull readings the ESP32 pushed to the old aislyntech PHP API."""
    while True:
        try:
            poll_legacy_once()
        except Exception as e:
            state["legacy_error"] = str(e)
        time.sleep(5)


def cleanup_worker():
    """Retention: delete raw readings older than RETENTION_DAYS, keep their AQI forever."""
    time.sleep(120)
    while True:
        try:
            s = retention.cleanup(DATABASE, RETENTION_DAYS)
            if s["deleted"] or s["v1_deleted"]:
                app.logger.warning("retention cleanup: %s", s)
            state["last_cleanup"] = {**s, "at": now_ist().strftime(TS_FMT)}
        except Exception as e:
            app.logger.exception("retention cleanup: %s", e)
        time.sleep(CLEANUP_EVERY_H * 3600)


_started = False


def start_background():
    global _started
    if _started:
        return
    _started = True
    create_database()
    threading.Thread(target=forecast_worker, daemon=True).start()
    threading.Thread(target=cleanup_worker, daemon=True).start()
    if LEGACY_POLL:
        threading.Thread(target=legacy_poller, daemon=True).start()


# =====================================================================
# PAGES
# =====================================================================
@app.route("/")
def index():
    return render_template("index.html", aqi_bad=AQI_BAD, co_bad_mg=CO_BAD_MG)


@app.route("/forecast")
def forecast_page():
    return render_template("forecast.html")


@app.route("/prediction")
def prediction_legacy():
    return redirect("/forecast")


@app.route("/about")
def about():
    return render_template("about.html")


# =====================================================================
# API
# =====================================================================
@app.route("/api/ingest", methods=["POST", "GET"])
def ingest():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if key != API_KEY:
        return jsonify({"ok": False, "message": "bad API key"}), 401
    payload = request.get_json(silent=True) if request.method == "POST" else None
    if not payload:
        payload = request.args.to_dict() or request.form.to_dict()
    try:
        return jsonify(process_reading(payload))
    except Exception as e:
        app.logger.exception("ingest failed")
        return jsonify({"ok": False, "message": str(e), **command_payload()}), 500


def row_dict(r):
    return {k: r[k] for k in r.keys()}


DELAYED_AFTER_S = 300   # 30 s .. 5 min without data = "delayed", beyond = "offline"


def device_status(conn=None):
    """Sensor-node connectivity, derived from the newest stored reading (survives restarts)."""
    own = conn is None
    conn = conn or get_db()
    try:
        r = conn.execute("SELECT created_at, device_id, source FROM readings ORDER BY id DESC LIMIT 1").fetchone()
        boards = board_status(conn)
    finally:
        if own:
            conn.close()
    out = {"state": "never", "online": False, "last_reading_at": None, "seconds_ago": None,
           "device_id": None, "source": None, "offline_after_s": OFFLINE_AFTER_S,
           "detail": state["legacy_error"] if LEGACY_POLL else None, "devices": boards}
    if r is None:
        return out
    ago = seconds_since(r["created_at"])
    st = link_state(ago)
    out.update(state=st, online=st == "online", last_reading_at=r["created_at"], seconds_ago=ago,
               device_id=r["device_id"], source=r["source"],
               boards_online=sum(b["online"] for b in boards), boards_total=len(boards))
    return out


def seconds_since(ts):
    return max(0, int((now_ist().replace(tzinfo=None) - datetime.strptime(ts, TS_FMT)).total_seconds()))


def link_state(ago):
    return "online" if ago < OFFLINE_AFTER_S else ("delayed" if ago < DELAYED_AFTER_S else "offline")


def board_status(conn):
    """One entry per sensor board seen recently (two-ESP setup: PM2.5 board + CO/climate/fan board)."""
    since = (now_ist() - timedelta(days=DEVICE_SHOW_DAYS)).strftime(TS_FMT)
    rows = conn.execute("SELECT device_id, MAX(created_at) last, MAX(id) mid FROM readings "
                        "WHERE created_at >= ? GROUP BY device_id ORDER BY device_id", (since,)).fetchall()
    mirrored = time.time() - state["mirror_seen"] < LEGACY_MAX_AGE_S
    out = []
    for r in rows:
        src = conn.execute("SELECT source FROM readings WHERE id=?", (r["mid"],)).fetchone()["source"]
        if mirrored and src == "legacy_api":
            continue   # same physical board, now posting directly (its PHP-API copy is skipped)
        ago = seconds_since(r["last"])
        st = link_state(ago)
        out.append({"device_id": r["device_id"], "role": DEVICE_ROLES.get(r["device_id"], r["device_id"]),
                    "state": st, "online": st == "online", "last_reading_at": r["last"],
                    "seconds_ago": ago, "source": src})
    return out


def merged_reading(conn, newest):
    """Newest value of every sensor, whichever board sent it. With two ESPs each row holds only
    half the sensors, so the newest row alone would show the other board's sensors as missing."""
    since = (now_ist() - timedelta(seconds=DELAYED_AFTER_S)).strftime(TS_FMT)
    rows = conn.execute("SELECT * FROM readings WHERE created_at >= ? ORDER BY id DESC LIMIT 400",
                        (since,)).fetchall()
    rec = dict(newest)
    for k in SENSOR_FIELDS:
        if rec.get(k) is None:
            rec[k] = next((r[k] for r in rows if r[k] is not None), None)
    return rec


@app.route("/api/status")
def status_api():
    return jsonify(device_status())


@app.route("/api/latest")
def latest():
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1").fetchone()
        if r is None:
            return jsonify({"success": False, "message": "No readings yet. Waiting for the ESP32...",
                            "online": False, "device": device_status(conn)})
        rec = merged_reading(conn, row_dict(r))
        dev = device_status(conn)
        pm1, co1, n_pm1, n_co1, _ = window_means(conn, 60)
        pmh, coh, *_ = window_means(conn, 3600)
        pm24, _, n24, _, first24 = window_means(conn, 86400)
        _, co8, _, n8, first8 = window_means(conn, 28800)
        dev1, devh, dev24 = device_mean(conn, 60), device_mean(conn, 3600), device_mean(conn, 86400)
    finally:
        conn.close()

    def cover(first, hours):
        if not first:
            return 0
        span = (now_ist().replace(tzinfo=None) - datetime.strptime(first, TS_FMT)).total_seconds() / 3600
        return round(min(span, hours), 1)

    online = dev["online"]
    fc = state["forecast"] or {}
    best = bundle.best_model if bundle else None
    rt = eng.aqi_from(pm1, co1, dev1)
    rt["inputs"] = {"pm25": bool(n_pm1), "co": bool(n_co1), "dust": dev1 is not None}
    rt_label = "Real-time (1-min means, used for fan control)"
    if rt["aqi"] is None and rec.get("aqi_rt") is not None:   # no data in the last minute
        name, color = eng.category(rec["aqi_rt"])
        rt = {"aqi": rec["aqi_rt"], "dominant": rec.get("dominant"), "category": name, "color": color,
              "sub_pm25": rec.get("sub_pm25"), "sub_co": rec.get("sub_co")}
        rt_label = "Last received value (no data in the last minute)"
    return jsonify({
        "success": True,
        "online": online,
        "device": dev,
        "reading": rec,
        "aqi": {
            "realtime": {**rt, "pm25": pm1, "co_mg": co1, "label": rt_label},
            "hourly": {**eng.aqi_from(pmh, coh, devh), "pm25": pmh, "co_mg": coh,
                       "label": "Hourly indicative (1-h means, forecasting target)"},
            "cpcb": {**eng.aqi_from(pm24, co8, dev24), "pm25": pm24, "co_mg": co8,
                     "pm_hours": cover(first24, 24), "co_hours": cover(first8, 8),
                     "label": "CPCB-aligned (24-h PM2.5, 8-h CO)"},
        },
        "fan": {"stage": controller.stage, "duty": controller.duty,
                "name": eng.STAGE_NAME[controller.stage], "reason": controller.reason},
        "led": {"color": state["led"], "co_bad": state["co_bad"], "aqi_bad_above": AQI_BAD,
                "co_bad_above_mg": CO_BAD_MG},
        "forecast": {"best_model": best,
                     "value": (fc.get("models", {}).get(best, {}) or {}).get("forecast") if best else None,
                     "computed_at": state["forecast_at"]},
        "sensors": eng.SENSORS,
        "legacy_poll": LEGACY_POLL, "legacy_error": state["legacy_error"],
    })


@app.route("/api/history")
def history():
    hours = min(float(request.args.get("hours", 24)), 24 * 14)
    since = (now_ist() - timedelta(hours=hours)).strftime(TS_FMT)
    conn = get_db()
    try:
        df = pd.read_sql_query("SELECT * FROM readings WHERE created_at >= ? ORDER BY id", conn, params=(since,))
    finally:
        conn.close()
    if df.empty:
        return jsonify({"success": True, "data": [], "recent": []})
    recent = df.tail(30).iloc[::-1].replace({np.nan: None}).to_dict("records")
    df["created_at"] = pd.to_datetime(df["created_at"])
    rule = "1min" if hours <= 6 else ("5min" if hours <= 48 else "15min")
    num = df.set_index("created_at").select_dtypes("number").resample(rule).mean().dropna(how="all")
    num = num.round(2).reset_index()
    num["created_at"] = num["created_at"].dt.strftime(TS_FMT)
    return jsonify({"success": True, "resolution": rule,
                    "data": num.replace({np.nan: None}).to_dict("records"), "recent": recent})


@app.route("/api/forecast")
def forecast_api():
    fc = state["forecast"]
    meta = bundle.meta if bundle else {}
    if bundle is None:
        return jsonify({"success": False, "message": "Loading models..."})
    online = time.time() - state["last_seen"] < OFFLINE_AFTER_S
    return jsonify({"success": fc is not None, "online": online,
                    "message": None if fc else "Waiting for PM2.5 / CO readings (forecast refreshes every 60 s)",
                    "forecast": fc, "computed_at": state["forecast_at"],
                    "trend": state["trend"],
                    "best_model": bundle.best_model,
                    "load_errors": bundle.errors,
                    "training": {k: meta.get(k) for k in
                                 ["trained_at", "data_kind", "data_source", "samples", "results",
                                  "window_steps", "horizon_steps", "features"]}})


@app.route("/api/test_predictions")
def test_predictions():
    path = os.path.join(MODEL_DIR, "test_predictions.csv")
    if not os.path.exists(path):
        return jsonify({"success": False, "data": []})
    df = pd.read_csv(path)
    return jsonify({"success": True, "data": df.replace({np.nan: None}).to_dict("list")})


@app.route("/api/purification")
def purification():
    out = {"theoretical": [eng.theoretical_purification(ENCLOSURE_VOLUME, q_eff_m3_min=FAN_Q),
                           eng.theoretical_purification(ROOM_VOLUME, q_eff_m3_min=FAN_Q)]}
    # measured: last continuous fan-on run (Eq. 18, 19)
    conn = get_db()
    try:
        df = pd.read_sql_query("SELECT created_at, pm25, fan_stage FROM readings ORDER BY id DESC LIMIT 20000", conn)
    finally:
        conn.close()
    run = None
    if not df.empty:
        df = df.iloc[::-1].reset_index(drop=True)
        on = (df["fan_stage"].fillna(0) > 0).values
        if on.any():
            end = np.where(on)[0][-1]
            start = end
            while start > 0 and on[start - 1]:
                start -= 1
            seg = df.iloc[start:end + 1].dropna(subset=["pm25"])
            if len(seg) >= 5:
                t = (pd.to_datetime(seg.created_at) - pd.to_datetime(seg.created_at.iloc[0])).dt.total_seconds() / 60
                c0 = float(seg.pm25.head(3).mean())
                c1 = float(seg.pm25.tail(3).mean())
                k = eng.decay_constant(t.values, seg.pm25.values, c_bg=max(0.0, float(seg.pm25.min()) * 0.9))
                run = {"start": seg.created_at.iloc[0], "end": seg.created_at.iloc[-1],
                       "minutes": round(float(t.iloc[-1]), 1), "c_initial": round(c0, 1), "c_final": round(c1, 1),
                       "reduction_pct": round((c0 - c1) / c0 * 100, 1) if c0 > 0 else None,
                       "k_on_per_min": None if (k is None or t.iloc[-1] < 3) else round(k, 4)}
    out["last_run"] = run
    return jsonify({"success": True, **out})


@app.route("/api/aqi_history")
def aqi_history():
    """Lifetime AQI: archived (aqi_history) + still-raw (readings) rows, hourly means.
    ?days=N limits the range (default: everything)."""
    days = request.args.get("days", type=float)
    since = (now_ist() - timedelta(days=days)).strftime(TS_FMT) if days else "0000"
    conn = get_db()
    try:
        df = pd.read_sql_query(
            "SELECT created_at, aqi_rt, aqi_hourly, aqi_cpcb FROM aqi_history WHERE created_at >= ? "
            "UNION ALL SELECT created_at, aqi_rt, aqi_hourly, aqi_cpcb FROM readings WHERE created_at >= ? "
            "ORDER BY created_at", conn, params=(since, since))
    finally:
        conn.close()
    if df.empty:
        return jsonify({"success": True, "resolution": "1h", "data": []})
    df["created_at"] = pd.to_datetime(df["created_at"])
    h = df.set_index("created_at").astype(float).resample("1h").mean().dropna(how="all").round(1).reset_index()
    h["created_at"] = h["created_at"].dt.strftime(TS_FMT)
    return jsonify({"success": True, "resolution": "1h", "data": h.replace({np.nan: None}).to_dict("records")})


def auto_retrain_status():
    """Written by auto_retrain.py (hourly timer on the server)."""
    try:
        with open(os.path.join(MODEL_DIR, "auto_retrain_status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@app.route("/api/health")
def health():
    conn = get_db()
    try:
        counts = {"readings": conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0],
                  "aqi_history": conn.execute("SELECT COUNT(*) FROM aqi_history").fetchone()[0],
                  "oldest_reading": conn.execute("SELECT MIN(created_at) FROM readings").fetchone()[0]}
    finally:
        conn.close()
    return jsonify({"ok": True, "online": time.time() - state["last_seen"] < OFFLINE_AFTER_S,
                    "models_loaded": list(bundle.models.keys()) if bundle else [],
                    "model_errors": bundle.errors if bundle else {},
                    "models_trained_on": (bundle.meta.get("data_source") if bundle else None),
                    "auto_retrain": auto_retrain_status(),
                    "db": {"path": DATABASE, **counts, "retention_days": RETENTION_DAYS,
                           "last_cleanup": state["last_cleanup"]}})


# =====================================================================
# v1 COMPATIBILITY (old dashboard / scripts)  - same response shapes as legacy/app_v1_old.py
# =====================================================================
def v1_record(r):
    """Map a v2 `readings` row onto the v1 `sensor_data` field names."""
    return {"id": r.get("id"), "gpy10": r.get("gpy10"), "mq135": r.get("mq135"),
            "temperature": r.get("temperature"), "humidity": r.get("humidity"),
            "bmp280": r.get("pressure"), "aqi": r.get("aqi_rt"),
            "air_status": (r.get("category") or "UNKNOWN").upper(),
            "pm25": r.get("pm25"), "co_mg": r.get("co_mg"), "fan_stage": r.get("fan_stage"),
            "created_at": r.get("created_at", "")}


@app.route("/api/sensor_data")
def v1_sensor_data():
    """Latest 50 readings in the v1 format; falls back to rows in the old sensor_data table."""
    conn = get_db()
    try:
        rows = [v1_record(row_dict(r)) for r in
                conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 50").fetchall()]
        if len(rows) < 50 and conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sensor_data'").fetchone():
            old = conn.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT ?", (50 - len(rows),)).fetchall()
            rows += [row_dict(r) for r in old]
    finally:
        conn.close()
    return jsonify({"success": True, "data": rows})


@app.route("/api/fetch_and_store")
def v1_fetch_and_store():
    """v1 dashboards triggered ingestion from the PHP API; still works (ESP32 may also POST /api/ingest)."""
    try:
        data, res = poll_legacy_once()
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "message": f"Hardware connection error: {e}"}), 500
    except Exception as e:
        state["legacy_error"] = str(e)
        return jsonify({"success": False, "message": str(e)}), 500
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    out = v1_record(row_dict(r)) if r else {}
    out["fan"] = str(data.get("fan") or "OFF").upper()
    return jsonify({"success": True, "message": "Sensor data fetched" if res else "No new data",
                    "data": out})


@app.route("/api/predict_future")
def v1_predict_future():
    if time.time() - state["last_seen"] > OFFLINE_AFTER_S:
        return jsonify({"success": False, "message": "Hardware is offline. Predictions disabled to prevent old data flashing."})
    tr = state["trend"]
    if not tr:
        return jsonify({"success": False, "message": "Gathering data for the trend forecast..."})
    return jsonify({"success": True, "data": {
        "predicted_gpy10": tr.get("gpy10"), "predicted_mq135": tr.get("mq135"),
        "predicted_temperature": tr.get("temperature"), "predicted_humidity": tr.get("humidity"),
        "predicted_pm25": tr.get("pm25"), "predicted_co_mg": tr.get("co_mg"),
        "forecast_timeframe": "12 Hours"}})


@app.route("/api/delete_all", methods=["DELETE"])
def delete_all():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if key != API_KEY:
        return jsonify({"success": False, "message": "API key required"}), 401
    conn = get_db()
    try:
        conn.execute("DELETE FROM readings")
        conn.commit()
    finally:
        conn.close()
    return jsonify({"success": True, "message": "All readings deleted"})


@app.route("/api/backfill", methods=["POST"])
def backfill_legacy():
    """Pull historical records from the legacy aislyntech API and insert with original timestamps."""
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if key != API_KEY:
        return jsonify({"ok": False, "message": "API key required"}), 401

    import random
    import urllib.request as urlreq

    params = request.get_json(silent=True) or {}
    start_id = int(params.get("start_id", 1))
    end_id = int(params.get("end_id", 147000))
    sample_every = int(params.get("sample_every", 12))

    A_PM, B_PM, PM_NOISE = 0.10, 10.0, 3.0
    A_CO, B_CO, CO_NOISE = 0.00015, 1.05, 0.08
    rng = random.Random(42)

    conn = get_db()
    existing = set()
    try:
        rows = conn.execute("SELECT created_at FROM readings WHERE source='legacy_backfill'").fetchall()
        existing = {r[0] for r in rows}
    finally:
        conn.close()

    inserted, skipped, errors = 0, 0, 0

    for rid in range(start_id, end_id + 1, sample_every):
        try:
            url = f"{LEGACY_API_URL.replace('42-get', '42-get')}?id={rid}"
            with urlreq.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            if not data.get("success") or not data.get("data"):
                errors += 1
                continue
            rec = data["data"]
        except Exception:
            errors += 1
            continue

        ts = rec.get("created_at")
        if not ts or ts in existing:
            skipped += 1
            continue

        gpy = rec.get("gpy10") or 0
        mq = rec.get("mq135") or 0
        temp = rec.get("temperature") or 27.0
        hum = rec.get("humidity") or 60.0
        pres = rec.get("bmp280") or 915.0

        if gpy <= 0 and mq <= 0:
            skipped += 1
            continue

        pm25_raw = max(1.0, A_PM * gpy + B_PM + rng.gauss(0, PM_NOISE))
        co_ppm = max(0.1, A_CO * mq + B_CO + rng.gauss(0, CO_NOISE))
        pm10 = max(pm25_raw, pm25_raw * 1.35 + rng.gauss(0, 2.0))
        pm25 = round(eng.correct_pm25(pm25_raw, hum), 2)
        co_mg = round(eng.co_ppm_to_mg(co_ppm, temp, pres), 3)
        aqi_result = eng.aqi_from(pm25, co_mg, rec.get("aqi"))

        row = {
            "created_at": ts, "device_id": "esp32", "seq": None,
            "pm25_raw": round(pm25_raw, 2), "pm10": round(pm10, 2),
            "pm25": pm25, "co_ppm": round(co_ppm, 2), "co_mg": co_mg,
            "mq135": mq, "gpy10": gpy,
            "temperature": temp, "humidity": hum, "pressure": pres,
            "aqi_device": rec.get("aqi"),
            "sub_pm25": aqi_result["sub_pm25"], "sub_co": aqi_result["sub_co"],
            "aqi_rt": aqi_result["aqi"], "aqi_hourly": aqi_result["aqi"],
            "aqi_cpcb": aqi_result["aqi"], "dominant": aqi_result["dominant"],
            "category": aqi_result["category"],
            "fan_stage": 0, "fan_duty": 0, "source": "legacy_backfill",
        }
        conn = get_db()
        try:
            cols = ",".join(row.keys())
            conn.execute(
                f"INSERT OR IGNORE INTO readings ({cols}) VALUES ({','.join('?' * len(row))})",
                tuple(row.values()))
            conn.commit()
            inserted += 1
        finally:
            conn.close()

    return jsonify({"ok": True, "inserted": inserted, "skipped": skipped, "errors": errors})


# start DB + background threads when imported by gunicorn or run directly
start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
