"""
forecasting.py
==============
Shared model code for the 12-hour AQI forecast (Sec 4.6, Table 4).

  Persistence   A^(t+h) = A(t)                      (baseline)
  ARIMA         (p,d,q) chosen by AIC on the AQI series
  Random Forest lagged feature vector
  XGBoost       lagged feature vector
  LSTM          96 x F sequence
  GRU           96 x F sequence (same architecture as LSTM)

Used by train.py (fit + evaluate) and by app.py (live inference).
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from aqi_engine import FEATURES, HORIZON, WINDOW, tabular_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
MODEL_NAMES = ["Persistence", "ARIMA", "Random Forest", "XGBoost", "LSTM", "GRU"]
ML_MODELS = MODEL_NAMES[1:]  # the five forecasting models


# ---------------------------------------------------------------------
# Scaler stored as JSON (version independent, no pickle)
# ---------------------------------------------------------------------
class JsonScaler:
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, float)
        self.std = None if std is None else np.asarray(std, float)

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std < 1e-9] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean) / self.std

    def inverse(self, X):
        return np.asarray(X, float) * self.std + self.mean

    def to_dict(self):
        return {"mean": np.atleast_1d(self.mean).tolist(), "std": np.atleast_1d(self.std).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


# ---------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------
def fill_inputs(ts: pd.DataFrame) -> pd.DataFrame:
    """Model INPUTS may bridge short gaps (<= 1 h); targets are never filled."""
    return ts.ffill(limit=4)


def valid_origins(ts: pd.DataFrame, require_target: bool = True) -> np.ndarray:
    """Positions t with a complete 96-step input window (and a real target at t+48)."""
    ok_row = fill_inputs(ts)[FEATURES].notna().all(axis=1).values.astype(int)
    full_window = pd.Series(ok_row).rolling(WINDOW).sum().values == WINDOW
    idx = np.where(full_window)[0]
    if require_target:
        tgt = ts["aqi"].shift(-HORIZON).notna().values
        idx = idx[tgt[idx]]
    return idx


def sequence_tensor(ts_values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.stack([ts_values[o - WINDOW + 1:o + 1] for o in origins]).astype("float32")


# ---------------------------------------------------------------------
# ARIMA helpers
# ---------------------------------------------------------------------
def arima_forecast_path(series: np.ndarray, order, params, steps: int = HORIZON) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    warnings.filterwarnings("ignore")
    mod = ARIMA(series, order=tuple(order))
    res = mod.filter(np.asarray(params))
    return np.asarray(res.forecast(steps))


# ---------------------------------------------------------------------
# Live inference bundle
# ---------------------------------------------------------------------
class ForecastBundle:
    """Loads everything train.py saved and predicts A(t+12h) with every model."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.dir = model_dir
        self.meta = {}
        self.models = {}
        self.errors = {}
        self.loaded = False
        self.load()

    def load(self):
        meta_path = os.path.join(self.dir, "training_metrics.json")
        if not os.path.exists(meta_path):
            self.errors["all"] = "No trained models found. Run: python train.py"
            return
        with open(meta_path) as f:
            self.meta = json.load(f)
        cfg = self.meta.get("artifacts", {})
        self.feat_scaler = JsonScaler.from_dict(cfg["feature_scaler"])
        self.tgt_scaler = JsonScaler.from_dict(cfg["target_scaler"])
        self.tab_columns = cfg["tabular_columns"]
        self.tab_fill = cfg.get("tabular_fill", {})

        # ARIMA
        try:
            self.models["ARIMA"] = cfg["arima"]
        except Exception as e:
            self.errors["ARIMA"] = str(e)
        # Random Forest
        try:
            import joblib
            self.models["Random Forest"] = joblib.load(os.path.join(self.dir, "random_forest.joblib"))
        except Exception as e:
            self.errors["Random Forest"] = f"{type(e).__name__}: {e}"
        # XGBoost
        try:
            import xgboost as xgb
            m = xgb.XGBRegressor()
            m.load_model(os.path.join(self.dir, "xgboost.json"))
            self.models["XGBoost"] = m
        except Exception as e:
            self.errors["XGBoost"] = f"{type(e).__name__}: {e}"
        # LSTM / GRU (TensorFlow optional)
        for name, fn in [("LSTM", "lstm.keras"), ("GRU", "gru.keras")]:
            try:
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
                from tensorflow import keras
                self.models[name] = keras.models.load_model(os.path.join(self.dir, fn), compile=False)
            except ImportError:
                self.errors[name] = "TensorFlow not installed on this server (pip install tensorflow-cpu)"
            except Exception as e:
                self.errors[name] = f"{type(e).__name__}: {e}"
        self.loaded = True

    @property
    def best_model(self) -> Optional[str]:
        return self.meta.get("best_model")

    def predict(self, ts: pd.DataFrame) -> dict:
        """
        ts: 15-min feature frame from aqi_engine.build_timeseries (most recent last).
        If fewer than 96 complete steps are available the window is padded with the
        earliest available step and the result is flagged 'warming_up'.
        """
        out = {"models": {}, "warming_up": False, "history_steps": 0}
        if not self.loaded:
            out["error"] = self.errors.get("all", "models not loaded")
            return out
        ts = ts[FEATURES].copy()
        ts = ts.ffill().dropna()
        if ts.empty:
            out["error"] = "No readings with PM2.5 / CO yet"
            return out
        n = len(ts)
        out["history_steps"] = int(min(n, WINDOW))
        if n < WINDOW:
            out["warming_up"] = True
            pad = pd.DataFrame([ts.iloc[0].values] * (WINDOW - n), columns=ts.columns,
                               index=pd.date_range(end=ts.index[0] - pd.Timedelta("15min"),
                                                   periods=WINDOW - n, freq="15min"))
            ts = pd.concat([pad, ts])
        ts = ts.iloc[-WINDOW * 2:]   # enough history for lag features
        a_now = float(ts["aqi"].iloc[-1])
        out["aqi_now"] = round(a_now, 1)
        out["origin_time"] = str(ts.index[-1])
        out["target_time"] = str(ts.index[-1] + pd.Timedelta(minutes=15 * HORIZON))
        out["models"]["Persistence"] = {"forecast": round(a_now, 1)}

        # ARIMA -> full 48-step path
        if "ARIMA" in self.models:
            try:
                a = self.models["ARIMA"]
                hist = ts["aqi"].values[-WINDOW * 2:]
                path = arima_forecast_path(hist, a["order"], a["params"])
                path = np.clip(path, 0, 500)
                out["models"]["ARIMA"] = {"forecast": round(float(path[-1]), 1),
                                          "path": [round(float(v), 1) for v in path]}
            except Exception as e:
                out["models"]["ARIMA"] = {"error": str(e)}

        # tree models
        X = tabular_features(ts).iloc[[-1]]
        X = X[self.tab_columns].fillna(pd.Series(self.tab_fill)).fillna(0)
        for name in ["Random Forest", "XGBoost"]:
            if name in self.models:
                try:
                    y = float(self.models[name].predict(X.values)[0])
                    out["models"][name] = {"forecast": round(float(np.clip(y, 0, 500)), 1)}
                except Exception as e:
                    out["models"][name] = {"error": str(e)}

        # recurrent models
        seq = self.feat_scaler.transform(ts[FEATURES].values[-WINDOW:])[None, ...].astype("float32")
        for name in ["LSTM", "GRU"]:
            if name in self.models:
                try:
                    y = self.models[name](seq, training=False).numpy().ravel()[0]
                    y = float(self.tgt_scaler.inverse([[y]]).ravel()[0])
                    out["models"][name] = {"forecast": round(float(np.clip(y, 0, 500)), 1)}
                except Exception as e:
                    out["models"][name] = {"error": str(e)}

        for name, err in self.errors.items():
            if name in MODEL_NAMES and name not in out["models"]:
                out["models"][name] = {"error": err}
        return out
