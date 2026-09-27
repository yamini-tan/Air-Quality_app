"""
train.py  -  trains and compares the five 12-hour AQI forecasting models
=========================================================================
Protocol (paper Sec 4.6, Table 4):
  * target  : hourly indicative CPCB AQI 12 h ahead (h = 48 steps of 15 min)
  * inputs  : 24-h window (w = 96) of AQI, PM2.5, CO, gas index, dust, T, RH, P,
              sin/cos hour of day
  * split   : chronological 70 / 15 / 15 % train / validation / test, no shuffling
  * scalers : fitted on the training part only
  * tuning  : hyper-parameters chosen on the validation set (ARIMA: AIC)
  * scoring : MAE, RMSE, R2 computed ONCE on the test set, MAE gain vs persistence

Usage
  python train.py                       # auto: real DB data if >= 7 days, else synthetic
  python train.py --source db           # force real readings from sensor_data.db
  python train.py --source csv --csv my_export.csv
  python train.py --source synthetic    # bootstrap models (NOT for the paper)
  python train.py --quick               # smaller search (for small VMs)
"""

import argparse
import itertools
import json
import os
import sqlite3
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from aqi_engine import FEATURES, HORIZON, WINDOW, build_timeseries, tabular_features
from forecasting import MODEL_DIR, JsonScaler, fill_inputs, sequence_tensor, valid_origins

SEED = 42
np.random.seed(SEED)
MIN_REAL_DAYS = 7


# =====================================================================
# DATA
# =====================================================================
def load_records(source, csv_path, db_path):
    note = ""
    if source in ("auto", "db") and os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT created_at, pm25, co_mg, mq135, gpy10, temperature, humidity, pressure, aqi_device "
                "FROM readings WHERE pm25 IS NOT NULL OR co_mg IS NOT NULL OR aqi_device IS NOT NULL "
                "ORDER BY id", conn)
        except Exception:
            df = pd.DataFrame()
        conn.close()
        if not df.empty:
            span = (pd.to_datetime(df.created_at).max() - pd.to_datetime(df.created_at).min()).total_seconds() / 86400
        else:
            span = 0
        if source == "db" or span >= MIN_REAL_DAYS:
            if df.empty:
                raise SystemExit("No readings in the database yet.")
            return df, "real", f"sensor_data.db ({len(df)} readings, {span:.1f} days)"
        note = f"DB has only {span:.1f} days of PM2.5/CO data (< {MIN_REAL_DAYS}); "
    if source == "csv":
        df = pd.read_csv(csv_path)
        return df, "real", f"{csv_path} ({len(df)} rows)"
    from synthetic_data import generate
    df = generate(days=30)
    return df, "synthetic", note + "SYNTHETIC bootstrap data (30 days, 1-min) - retrain on real data before reporting results"


def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "db", "csv", "synthetic"])
    ap.add_argument("--csv", default="dataset.csv")
    ap.add_argument("--db", default=os.environ.get("AQMS_DB", "sensor_data.db"))
    ap.add_argument("--quick", action="store_true", help="reduced search space")
    ap.add_argument("--no-dl", action="store_true", help="skip LSTM/GRU (no TensorFlow)")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60 + "\n  AQMS 12-HOUR AQI FORECAST - MODEL TRAINING\n" + "=" * 60)
    records, kind, source_note = load_records(args.source, args.csv, args.db)
    print("Data:", source_note)

    ts = build_timeseries(records)
    origins = valid_origins(ts)
    if len(origins) < 200:
        raise SystemExit(f"Only {len(origins)} usable 24-h windows; collect more data.")
    y_all = ts["aqi"].shift(-HORIZON).values          # target A(t+h) (never filled)
    ts = fill_inputs(ts)                               # inputs bridge gaps <= 1 h
    n = len(origins)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    o_tr, o_va, o_te = origins[:n_tr], origins[n_tr:n_tr + n_va], origins[n_tr + n_va:]
    y_tr, y_va, y_te = y_all[o_tr], y_all[o_va], y_all[o_te]
    train_end = o_tr[-1]
    print(f"15-min steps: {len(ts)} | samples: train {len(o_tr)}, val {len(o_va)}, test {len(o_te)}")

    results, preds_te, configs, times = {}, {}, {}, {}
    A = ts["aqi"].values

    # ---------------- Persistence ----------------
    preds_te["Persistence"] = A[o_te]
    results["Persistence"] = metrics(y_te, preds_te["Persistence"])
    configs["Persistence"] = "A(t+h) = A(t)"
    times["Persistence"] = None

    # ---------------- ARIMA ----------------
    from statsmodels.tsa.arima.model import ARIMA
    print("\n[ARIMA] AIC search p,q in 0..3, d in 0,1")
    series_tr = A[:train_end + 1]
    best = None
    grid = list(itertools.product(range(4), [0, 1], range(4)))
    if args.quick:
        grid = [(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 2)]
    for p, d, q in grid:
        try:
            r = ARIMA(series_tr, order=(p, d, q)).fit()
            if best is None or r.aic < best[0]:
                best = (r.aic, (p, d, q))
        except Exception:
            pass
    order = best[1]
    t0 = time.time()
    arima_res = ARIMA(series_tr, order=order).fit()
    times["ARIMA"] = time.time() - t0
    params = arima_res.params.tolist()
    full = ARIMA(A, order=order).filter(arima_res.params)
    p_te = []
    for o in o_te:
        fc = full.get_prediction(start=o + 1, end=o + HORIZON, dynamic=True).predicted_mean
        p_te.append(float(np.asarray(fc)[-1]))
    preds_te["ARIMA"] = np.clip(np.array(p_te), 0, 500)
    results["ARIMA"] = metrics(y_te, preds_te["ARIMA"])
    configs["ARIMA"] = f"order (p,d,q) = {order}, AIC = {best[0]:.1f}"
    print("  best", configs["ARIMA"], "->", results["ARIMA"])

    # ---------------- Tabular features ----------------
    X_tab = tabular_features(ts)
    tab_cols = list(X_tab.columns)
    fill = X_tab.iloc[:train_end + 1].mean().to_dict()
    X_tab = X_tab.fillna(fill)
    Xtr, Xva, Xte = X_tab.values[o_tr], X_tab.values[o_va], X_tab.values[o_te]

    # ---------------- Random Forest ----------------
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    print("\n[Random Forest] trees {100,300,500} x depth {10,20,full}")
    rf_grid = list(itertools.product([100, 300, 500], [10, 20, None]))
    if args.quick:
        rf_grid = [(100, 10), (300, 20)]
    best = None
    for trees, depth in rf_grid:
        m = RandomForestRegressor(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=SEED,
                                  min_samples_leaf=2).fit(Xtr, y_tr)
        mae = mean_absolute_error(y_va, m.predict(Xva))
        if best is None or mae < best[0]:
            best = (mae, trees, depth)
    _, trees, depth = best
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=trees, max_depth=depth, n_jobs=-1, random_state=SEED,
                               min_samples_leaf=2).fit(np.vstack([Xtr, Xva]), np.concatenate([y_tr, y_va]))
    times["Random Forest"] = time.time() - t0
    preds_te["Random Forest"] = rf.predict(Xte)
    results["Random Forest"] = metrics(y_te, preds_te["Random Forest"])
    configs["Random Forest"] = f"trees = {trees}, depth = {depth or 'full'}"
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"), compress=3)
    print("  best", configs["Random Forest"], "->", results["Random Forest"])

    # ---------------- XGBoost ----------------
    import xgboost as xgb
    print("\n[XGBoost] eta {0.03,0.1} x depth {3,5,7} x trees {200,500}")
    xg_grid = list(itertools.product([0.03, 0.1], [3, 5, 7], [200, 500]))
    if args.quick:
        xg_grid = [(0.1, 3, 200), (0.03, 5, 500)]
    best = None
    for eta, depth, trees in xg_grid:
        m = xgb.XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=eta, subsample=0.9,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=-1).fit(Xtr, y_tr)
        mae = mean_absolute_error(y_va, m.predict(Xva))
        if best is None or mae < best[0]:
            best = (mae, eta, depth, trees)
    _, eta, depth, trees = best
    t0 = time.time()
    xg = xgb.XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=eta, subsample=0.9,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=-1).fit(
        np.vstack([Xtr, Xva]), np.concatenate([y_tr, y_va]))
    times["XGBoost"] = time.time() - t0
    preds_te["XGBoost"] = xg.predict(Xte)
    results["XGBoost"] = metrics(y_te, preds_te["XGBoost"])
    configs["XGBoost"] = f"eta = {eta}, depth = {depth}, trees = {trees}"
    xg.save_model(os.path.join(MODEL_DIR, "xgboost.json"))
    print("  best", configs["XGBoost"], "->", results["XGBoost"])

    # ---------------- Sequence data (scalers on train only) ----------------
    feat_scaler = JsonScaler().fit(ts[FEATURES].values[:train_end + 1])
    tgt_scaler = JsonScaler().fit(y_tr.reshape(-1, 1))
    V = feat_scaler.transform(ts[FEATURES].values)
    Str, Sva, Ste = sequence_tensor(V, o_tr), sequence_tensor(V, o_va), sequence_tensor(V, o_te)
    ytr_s = tgt_scaler.transform(y_tr.reshape(-1, 1)).ravel()
    yva_s = tgt_scaler.transform(y_va.reshape(-1, 1)).ravel()
    histories = {}

    for fn in ("lstm.keras", "gru.keras"):          # remove stale recurrent models
        if os.path.exists(os.path.join(MODEL_DIR, fn)):
            os.remove(os.path.join(MODEL_DIR, fn))
    if not args.no_dl:
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.random.set_seed(SEED)

            def build(cell, layers, units):
                Layer = keras.layers.LSTM if cell == "LSTM" else keras.layers.GRU
                m = keras.Sequential([keras.Input((WINDOW, len(FEATURES)))])
                for i in range(layers):
                    m.add(Layer(units, return_sequences=(i < layers - 1)))
                    m.add(keras.layers.Dropout(0.2))
                m.add(keras.layers.Dense(1))
                m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
                return m

            dl_grid = list(itertools.product([1, 2], [32, 64]))
            if args.quick:
                dl_grid = [(1, 32)]
            for cell in ["LSTM", "GRU"]:
                print(f"\n[{cell}] layers {{1,2}} x units {{32,64}}, dropout 0.2, Adam 1e-3, batch 32, patience 10")
                best = None
                for layers, units in dl_grid:
                    tf.random.set_seed(SEED)
                    m = build(cell, layers, units)
                    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                    t0 = time.time()
                    h = m.fit(Str, ytr_s, validation_data=(Sva, yva_s), epochs=100, batch_size=32,
                              callbacks=[es], verbose=0)
                    dt = time.time() - t0
                    pv = tgt_scaler.inverse(m.predict(Sva, verbose=0)).ravel()
                    mae = mean_absolute_error(y_va, pv)
                    print(f"  layers={layers} units={units}: val MAE {mae:.2f} ({len(h.history['loss'])} epochs, {dt:.0f}s)")
                    if best is None or mae < best[0]:
                        best = (mae, layers, units, m, dt, h.history)
                _, layers, units, m, dt, hist = best
                times[cell] = dt
                histories[cell] = hist
                preds_te[cell] = np.clip(tgt_scaler.inverse(m.predict(Ste, verbose=0)).ravel(), 0, 500)
                results[cell] = metrics(y_te, preds_te[cell])
                configs[cell] = f"layers = {layers}, units = {units}, dropout = 0.2"
                m.save(os.path.join(MODEL_DIR, f"{cell.lower()}.keras"))
                print("  best", configs[cell], "->", results[cell])
        except ImportError:
            print("\nTensorFlow not installed - LSTM/GRU skipped.")

    # ---------------- Summary ----------------
    pers = results["Persistence"]["mae"]
    table = []
    for name in ["Persistence", "ARIMA", "Random Forest", "XGBoost", "LSTM", "GRU"]:
        if name not in results:
            continue
        r = results[name]
        table.append({"model": name, "mae": round(r["mae"], 2), "rmse": round(r["rmse"], 2),
                      "r2": round(r["r2"], 3),
                      "mae_gain_pct": None if name == "Persistence" else round((pers - r["mae"]) / pers * 100, 1),
                      "train_time_s": None if times.get(name) is None else round(times[name], 1),
                      "config": configs[name]})
    ml = [t for t in table if t["model"] != "Persistence"]
    best_model = min(ml, key=lambda t: t["mae"])["model"]

    print("\n" + "=" * 78)
    print(f"{'Model':15s}{'MAE':>8s}{'RMSE':>8s}{'R2':>8s}{'MAE gain':>11s}{'Train time':>12s}")
    for t in table:
        g = "-" if t["mae_gain_pct"] is None else f"{t['mae_gain_pct']:.1f} %"
        tt = "-" if t["train_time_s"] is None else f"{t['train_time_s']:.1f} s"
        print(f"{t['model']:15s}{t['mae']:8.2f}{t['rmse']:8.2f}{t['r2']:8.3f}{g:>11s}{tt:>12s}")
    print("Best model:", best_model)

    meta = {
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_kind": kind,
        "data_source": source_note,
        "steps_15min": int(len(ts)),
        "samples": {"train": int(len(o_tr)), "validation": int(len(o_va)), "test": int(len(o_te))},
        "window_steps": WINDOW, "horizon_steps": HORIZON, "features": FEATURES,
        "results": table, "best_model": best_model,
        "artifacts": {
            "feature_scaler": feat_scaler.to_dict(),
            "target_scaler": tgt_scaler.to_dict(),
            "tabular_columns": tab_cols,
            "tabular_fill": {k: float(v) for k, v in fill.items()},
            "arima": {"order": list(order), "params": params},
        },
    }
    with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # test predictions for the dashboard chart
    te_idx = ts.index[o_te] + pd.Timedelta(minutes=15 * HORIZON)
    out = pd.DataFrame({"target_time": te_idx, "actual": y_te})
    for k, v in preds_te.items():
        out[k] = np.round(v, 2)
    out.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)

    # Fig 4(c)-style comparison chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [t["model"] for t in table]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [t["mae"] for t in table], 0.4, label="MAE", color="#2563eb")
    ax.bar(x + 0.2, [t["rmse"] for t in table], 0.4, label="RMSE", color="#94a3b8")
    ax.set_xticks(x, names)
    ax.set_ylabel("AQI units")
    ax.set_title("12-hour AQI forecast error (test set)" + (" - SYNTHETIC DATA" if kind == "synthetic" else ""))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150)
    print("\nSaved models and metrics to", MODEL_DIR)


if __name__ == "__main__":
    main()
