"""
auto_retrain.py  -  retrains the forecast models by itself once enough real data exists
=========================================================================================
The server runs this every hour (systemd timer deploy/aqms-retrain.timer).

  ready    : the database holds >= MIN_DAYS of complete real readings (PM2.5/CO + climate)
             AND >= MIN_WINDOWS usable samples. A sample needs 24 h of inputs without gaps
             longer than 1 h plus the AQI 12 h later, so boards that are switched off at
             night break samples - leave both ESPs running.
  due      : ready AND (the models were trained on synthetic data
                        OR the last real training is >= RETRAIN_EVERY_DAYS old)
  training : train.py --source db into trained_models_new/ (low CPU priority)
  install  : the new files replace trained_models/; the old set is kept in
             trained_models_prev/. The app picks the new models up within 60 s (no restart).

Progress is written to trained_models/auto_retrain_status.json and shown in /api/health.

Usage
  python auto_retrain.py              # check, train if due (what the timer runs)
  python auto_retrain.py --dry-run    # only report how much usable data there is
  python auto_retrain.py --force      # train now if there is enough data, even if not due

Environment (optional): AQMS_DB, AUTO_RETRAIN_MIN_DAYS (7), AUTO_RETRAIN_EVERY_DAYS (7),
AUTO_RETRAIN_QUICK (0; 1 = smaller hyper-parameter search)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from aqi_engine import FEATURES, WINDOW, build_timeseries
from forecasting import MODEL_DIR, fill_inputs, valid_origins

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_WINDOWS = 100                      # enough for a reliable train/val/test split
MIN_DAYS = float(os.environ.get("AUTO_RETRAIN_MIN_DAYS", 7))
RETRAIN_EVERY_DAYS = float(os.environ.get("AUTO_RETRAIN_EVERY_DAYS", 7))
QUICK = os.environ.get("AUTO_RETRAIN_QUICK", "0") == "1"
RETRY_AFTER_H = 6                      # after a failed training, wait before trying again
TRAIN_TIMEOUT_S = 3 * 3600

STAGING = MODEL_DIR + "_new"
PREVIOUS = MODEL_DIR + "_prev"
STATUS_FILE = os.path.join(MODEL_DIR, "auto_retrain_status.json")
# training_metrics.json goes last: the app reloads when that file changes
MODEL_FILES = ["random_forest.joblib", "xgboost.json", "lstm.keras", "gru.keras",
               "test_predictions.csv", "model_comparison.png", "training_metrics.json"]
TS_FMT = "%Y-%m-%d %H:%M:%S"


def now():
    return datetime.now().strftime(TS_FMT)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def data_readiness(db_path):
    """How much of the database the models can actually learn from."""
    from train import load_records
    try:
        records, _, _ = load_records("db", None, db_path)
    except SystemExit:
        return {"usable_days": 0.0, "windows": 0, "readings": 0}
    ts = build_timeseries(records)
    complete = fill_inputs(ts)[FEATURES].notna().all(axis=1)
    return {"usable_days": round(float(complete.sum()) / WINDOW, 2),   # 96 steps of 15 min = 1 day
            "windows": int(len(valid_origins(ts))),
            "readings": int(len(records)),
            "newest_reading": str(records["created_at"].iloc[-1])}


def decide(ready_info, meta, status, force):
    """Returns (train_now, reason)."""
    days, windows = ready_info["usable_days"], ready_info["windows"]
    if days < MIN_DAYS or windows < MIN_WINDOWS:
        return False, (f"collecting data: {days:.1f} of {MIN_DAYS:g} days usable, "
                       f"{windows} of {MIN_WINDOWS} samples")
    last_fail = status.get("last_failure_at")
    if last_fail and not force:
        age_h = (datetime.now() - datetime.strptime(last_fail, TS_FMT)).total_seconds() / 3600
        if age_h < RETRY_AFTER_H:
            return False, f"last training failed {age_h:.1f} h ago; retrying after {RETRY_AFTER_H} h"
    if force:
        return True, "forced"
    if meta.get("data_kind") != "real":
        return True, "enough real data - replacing the synthetic bootstrap models"
    trained = datetime.strptime(meta["trained_at"], TS_FMT)
    age_d = (datetime.now() - trained).total_seconds() / 86400
    if age_d >= RETRAIN_EVERY_DAYS:
        return True, f"models are {age_d:.1f} days old - retraining with the newest data"
    return False, f"models up to date (trained on real data {age_d:.1f} days ago)"


def train(db_path):
    shutil.rmtree(STAGING, ignore_errors=True)
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--source", "db", "--db", db_path, "--out", STAGING]
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        cmd.append("--no-dl")
    if QUICK:
        cmd.append("--quick")
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE, timeout=TRAIN_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"train.py exited with code {r.returncode}")
    meta = read_json(os.path.join(STAGING, "training_metrics.json"))
    if meta.get("data_kind") != "real":
        raise RuntimeError("train.py did not produce models from real data")
    for f in ("random_forest.joblib", "xgboost.json"):
        if not os.path.exists(os.path.join(STAGING, f)):
            raise RuntimeError(f"{f} missing after training")
    return meta


def install():
    """Swap the freshly trained files in, keeping the previous set for rollback."""
    shutil.rmtree(PREVIOUS, ignore_errors=True)
    shutil.copytree(MODEL_DIR, PREVIOUS)
    for f in MODEL_FILES:
        new, cur = os.path.join(STAGING, f), os.path.join(MODEL_DIR, f)
        if os.path.exists(new):
            os.replace(new, cur)
        elif os.path.exists(cur):          # e.g. no TensorFlow this time -> no stale LSTM/GRU
            os.remove(cur)
    shutil.rmtree(STAGING, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("AQMS_DB", "sensor_data.db"))
    ap.add_argument("--dry-run", action="store_true", help="report only, never train")
    ap.add_argument("--force", action="store_true", help="train now if there is enough data")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    status = read_json(STATUS_FILE)
    meta = read_json(os.path.join(MODEL_DIR, "training_metrics.json"))
    info = data_readiness(args.db)
    go, reason = decide(info, meta, status, args.force)
    status.update({"checked_at": now(), **info, "needs_days": MIN_DAYS, "needs_windows": MIN_WINDOWS,
                   "retrain_every_days": RETRAIN_EVERY_DAYS, "models_data_kind": meta.get("data_kind"),
                   "models_trained_at": meta.get("trained_at"), "decision": reason})
    print(f"[{now()}] {reason}", flush=True)

    if go and not args.dry_run:
        status.update({"training_started_at": now(), "decision": "training: " + reason})
        write_status(status, args.dry_run)
        t0 = time.time()
        try:
            new = train(args.db)
            install()
            status.update({"last_success_at": now(), "last_failure_at": None, "last_error": None,
                           "models_data_kind": new["data_kind"], "models_trained_at": new["trained_at"],
                           "best_model": new["best_model"], "training_minutes": round((time.time() - t0) / 60, 1),
                           "decision": f"installed new models ({new['data_source']}); best: {new['best_model']}"})
            print(f"[{now()}] installed new models, best: {new['best_model']}", flush=True)
        except Exception as e:
            status.update({"last_failure_at": now(), "last_error": str(e)[:500],
                           "decision": "training failed - current models kept"})
            print(f"[{now()}] training failed, current models kept: {e}", flush=True)
            write_status(status, args.dry_run)
            sys.exit(1)
    write_status(status, args.dry_run)


def write_status(status, dry_run):
    if dry_run:
        print(json.dumps(status, indent=2))
        return
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, STATUS_FILE)


if __name__ == "__main__":
    main()
