import os
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# 1. SETTINGS
# ============================================================
DATASET_PATH = "dataset.csv"
OUTPUT_DIR = "trained_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Machine Learning Settings
TIME_STEPS = 12
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Future Forecast Settings (100 steps * 3 seconds = 5 Minutes)
FORECAST_STEPS = 100

print("\n==============================================")
print("     AIR QUALITY ML TRAINING SYSTEM")
print("==============================================")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nERROR: {DATASET_PATH} not found.\n"
        "Put dataset.csv in the same folder as this Python file."
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", list(df.columns))

# CHECK REQUIRED COLUMNS
required_columns = [
    "gpy10",
    "mq135",
    "temperature",
    "humidity",
    "bmp280",
    "air_status"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "\nMissing columns in dataset.csv: "
        + ", ".join(missing)
        + "\n\nRequired columns are:\n"
        + ", ".join(required_columns)
    )

# ============================================================
# DATA CLEANING
# ============================================================
print("\n----------------------------------------------")
print("STEP 1: DATA CLEANING")
print("----------------------------------------------")

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicates_removed = before - len(df)

print("Duplicate rows removed:", duplicates_removed)

# Convert numeric columns
numeric_columns = [
    "gpy10",
    "mq135",
    "temperature",
    "humidity",
    "bmp280"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove missing values
missing_before = df.isnull().sum().sum()
df = df.dropna().reset_index(drop=True)
missing_removed = missing_before - df.isnull().sum().sum()

print("Rows after cleaning:", len(df))
print("Missing/invalid values removed:", missing_removed)

# Clean status text
df["air_status"] = (
    df["air_status"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\nAir status distribution:")
print(df["air_status"].value_counts())

print("\n----------------------------------------------")
print("STEP 2: CREATE AIR QUALITY SCORE")
print("----------------------------------------------")

gpy10_clip = df["gpy10"].clip(
    lower=df["gpy10"].quantile(0.01),
    upper=df["gpy10"].quantile(0.99)
)

mq135_clip = df["mq135"].clip(
    lower=df["mq135"].quantile(0.01),
    upper=df["mq135"].quantile(0.99)
)

gpy10_norm = (
    (gpy10_clip - gpy10_clip.min())
    / (gpy10_clip.max() - gpy10_clip.min() + 1e-8)
)

mq135_norm = (
    (mq135_clip - mq135_clip.min())
    / (mq135_clip.max() - mq135_clip.min() + 1e-8)
)

df["AQI_Proxy"] = (
    (0.60 * gpy10_norm + 0.40 * mq135_norm) * 100
)

df["AQI_Proxy"] = df["AQI_Proxy"].clip(0, 100)

print("AQI_Proxy created.")
print(df["AQI_Proxy"].describe())

print("\n----------------------------------------------")
print("STEP 3: FILTERING AND SMOOTHING")
print("----------------------------------------------")

filter_columns = [
    "gpy10",
    "mq135",
    "temperature",
    "humidity",
    "bmp280",
    "AQI_Proxy"
]

for col in filter_columns:
    df[col + "_filtered"] = (
        df[col]
        .rolling(window=3, min_periods=1)
        .mean()
    )

ml_columns = [
    "gpy10_filtered",
    "mq135_filtered",
    "temperature_filtered",
    "humidity_filtered",
    "bmp280_filtered"
]

target_column = "AQI_Proxy_filtered"

print("Moving-average filtering completed.")

print("\n----------------------------------------------")
print("STEP 4: NORMALIZATION")
print("----------------------------------------------")

feature_scaler = MinMaxScaler()

df[ml_columns] = feature_scaler.fit_transform(
    df[ml_columns]
)

target_scaler = MinMaxScaler()

df[[target_column]] = target_scaler.fit_transform(
    df[[target_column]]
)

# Save scalers
with open(
    os.path.join(OUTPUT_DIR, "feature_scaler.pkl"),
    "wb"
) as f:
    pickle.dump(feature_scaler, f)

with open(
    os.path.join(OUTPUT_DIR, "target_scaler.pkl"),
    "wb"
) as f:
    pickle.dump(target_scaler, f)

print("Normalization completed.")
print("Scalers saved.")

print("\n----------------------------------------------")
print("STEP 5: DATASET SPLITTING")
print("----------------------------------------------")

n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("Total:", n)
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\n----------------------------------------------")
print("STEP 6: CREATE LSTM TIME-SERIES SEQUENCES")
print("----------------------------------------------")

feature_values = df[ml_columns].values
target_values = df[target_column].values

def create_sequences(features, targets, time_steps):
    X = []
    y = []

    for i in range(time_steps, len(features)):
        X.append(features[i - time_steps:i])
        y.append(targets[i])

    return np.array(X), np.array(y)

X_all, y_all = create_sequences(
    feature_values,
    target_values,
    TIME_STEPS
)

sequence_target_indices = np.arange(
    TIME_STEPS,
    len(df)
)

train_mask = sequence_target_indices < train_end
val_mask = (
    (sequence_target_indices >= train_end)
    & (sequence_target_indices < val_end)
)
test_mask = sequence_target_indices >= val_end

X_train = X_all[train_mask]
y_train = y_all[train_mask]

X_val = X_all[val_mask]
y_val = y_all[val_mask]

X_test = X_all[test_mask]
y_test = y_all[test_mask]

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print("\n----------------------------------------------")
print("STEP 7: BUILD LSTM MODEL")
print("----------------------------------------------")

lstm_model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(TIME_STEPS, len(ml_columns))
    ),
    Dropout(0.20),
    LSTM(
        32,
        return_sequences=False
    ),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1)
])

lstm_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

lstm_model.summary()

print("\n----------------------------------------------")
print("STEP 8: TRAIN LSTM")
print("----------------------------------------------")

lstm_model_path = os.path.join(
    OUTPUT_DIR,
    "lstm_aqi_forecasting.keras"
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        lstm_model_path,
        monitor="val_loss",
        save_best_only=True
    )
]

history = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

print("\nLSTM training completed.")
print("Model saved:", lstm_model_path)

print("\n----------------------------------------------")
print("STEP 9: LSTM TEST EVALUATION")
print("----------------------------------------------")

lstm_pred_scaled = lstm_model.predict(
    X_test,
    verbose=0
)

# Convert predictions back to original scale
lstm_pred = target_scaler.inverse_transform(
    lstm_pred_scaled
).flatten()

y_test_original = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

lstm_mae = mean_absolute_error(
    y_test_original,
    lstm_pred
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        lstm_pred
    )
)

print("LSTM MAE :", round(lstm_mae, 4))
print("LSTM RMSE:", round(lstm_rmse, 4))

print("\n----------------------------------------------")
print("STEP 10: TRAIN ARIMA")
print("----------------------------------------------")

all_aqi_original = target_scaler.inverse_transform(
    df[[target_column]]
).flatten()

train_aqi = all_aqi_original[:train_end]
val_aqi = all_aqi_original[train_end:val_end]
test_aqi = all_aqi_original[val_end:]

try:
    arima_model = ARIMA(
        train_aqi,
        order=(1, 1, 1)
    )

    arima_fitted = arima_model.fit()

    # Forecast the validation + test period
    forecast_length = len(val_aqi) + len(test_aqi)

    arima_forecast = arima_fitted.forecast(
        steps=forecast_length
    )

    arima_test_forecast = np.array(
        arima_forecast[-len(test_aqi):]
    )

    arima_mae = mean_absolute_error(
        test_aqi,
        arima_test_forecast
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            test_aqi,
            arima_test_forecast
        )
    )

    print("ARIMA order: (1, 1, 1)")
    print("ARIMA MAE :", round(arima_mae, 4))
    print("ARIMA RMSE:", round(arima_rmse, 4))

    # Save ARIMA model
    with open(
        os.path.join(OUTPUT_DIR, "arima_model.pkl"),
        "wb"
    ) as f:
        pickle.dump(arima_fitted, f)

    print("ARIMA model saved.")

except Exception as e:
    print("\nARIMA training failed:")
    print(e)
    arima_test_forecast = None
    arima_mae = None
    arima_rmse = None

# FUTURE STEP FORECAST
print("\n----------------------------------------------")
print(f"STEP 11: {FORECAST_STEPS}-STEP FUTURE FORECAST (5 MINS)")
print("----------------------------------------------")

# LSTM recursive forecast
last_sequence = feature_values[-TIME_STEPS:].copy()

future_lstm_scaled = []

for _ in range(FORECAST_STEPS):
    current_input = last_sequence.reshape(
        1,
        TIME_STEPS,
        len(ml_columns)
    )

    next_prediction = lstm_model.predict(
        current_input,
        verbose=0
    )[0][0]

    future_lstm_scaled.append(next_prediction)

    next_features = last_sequence[-1].copy()

    last_sequence = np.vstack([
        last_sequence[1:],
        next_features
    ])

future_lstm = target_scaler.inverse_transform(
    np.array(future_lstm_scaled).reshape(-1, 1)
).flatten()

print(f"\nLSTM {FORECAST_STEPS}-step forecast completed.")

# ARIMA 100-step forecast
try:
    arima_future = arima_fitted.forecast(
        steps=FORECAST_STEPS
    )
    print(f"ARIMA {FORECAST_STEPS}-step forecast completed.")

except Exception:
    arima_future = None

print("\n----------------------------------------------")
print("STEP 12: ANOMALY DETECTION")
print("----------------------------------------------")

def z_score_anomaly(series, threshold=3.0):
    mean = series.mean()
    std = series.std()

    if std == 0:
        return np.zeros(len(series), dtype=int)

    z = np.abs((series - mean) / std)

    return (z > threshold).astype(int)

anomaly_columns = [
    "gpy10",
    "mq135",
    "temperature",
    "humidity",
    "bmp280"
]

df["anomaly"] = 0

for col in anomaly_columns:
    df["anomaly"] = np.maximum(
        df["anomaly"],
        z_score_anomaly(df[col])
    )

print("Total anomalies detected:", int(df["anomaly"].sum()))

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "processed_dataset_with_anomalies.csv"
    ),
    index=False
)

# SAVE LSTM TRAINING GRAPH
print("\n----------------------------------------------")
print("STEP 13: SAVE GRAPHS")
print("----------------------------------------------")

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "lstm_training_loss.png"
    ),
    dpi=300
)

plt.close()

# LSTM ACTUAL VS PREDICTED GRAPH
plt.figure(figsize=(12, 5))

plt.plot(
    y_test_original,
    label="Actual AQI Proxy"
)

plt.plot(
    lstm_pred,
    label="LSTM Predicted AQI Proxy"
)

plt.title("LSTM AQI Forecasting - Actual vs Predicted")
plt.xlabel("Test Sample")
plt.ylabel("AQI Proxy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "lstm_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()

# ARIMA ACTUAL VS PREDICTED GRAPH
if arima_test_forecast is not None:
    plt.figure(figsize=(12, 5))

    plt.plot(
        test_aqi,
        label="Actual AQI Proxy"
    )

    plt.plot(
        arima_test_forecast,
        label="ARIMA Predicted AQI Proxy"
    )

    plt.title("ARIMA AQI Forecasting - Actual vs Predicted")
    plt.xlabel("Test Sample")
    plt.ylabel("AQI Proxy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "arima_actual_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()

# SENSOR TREND GRAPH
plt.figure(figsize=(12, 5))

plt.plot(
    df["gpy10"],
    label="GPY10"
)

plt.plot(
    df["mq135"],
    label="MQ135"
)

plt.title("Sensor Pollution Trend")
plt.xlabel("Sample")
plt.ylabel("Sensor Value")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sensor_pollution_trend.png"
    ),
    dpi=300
)

plt.close()

# AQI PROXY TREND GRAPH
original_aqi = target_scaler.inverse_transform(
    df[[target_column]]
).flatten()

plt.figure(figsize=(12, 5))

plt.plot(
    original_aqi,
    label="AQI Proxy"
)

plt.title("Air Quality / AQI Proxy Trend")
plt.xlabel("Sample")
plt.ylabel("AQI Proxy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "aqi_proxy_trend.png"
    ),
    dpi=300
)

plt.close()

# SAVE TEST PREDICTIONS
prediction_df = pd.DataFrame({
    "actual_aqi_proxy": y_test_original,
    "lstm_predicted_aqi_proxy": lstm_pred
})

if arima_test_forecast is not None:
    prediction_df["arima_predicted_aqi_proxy"] = arima_test_forecast

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_predictions.csv"
    ),
    index=False
)

# SAVE METRICS
metrics = {
    "dataset_rows": int(len(df)),
    "training_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "testing_rows": int(len(test_df)),
    "time_steps": TIME_STEPS,
    "forecast_steps": FORECAST_STEPS,
    "lstm_mae": float(lstm_mae),
    "lstm_rmse": float(lstm_rmse),
    "arima_mae": (
        None if arima_mae is None
        else float(arima_mae)
    ),
    "arima_rmse": (
        None if arima_rmse is None
        else float(arima_rmse)
    ),
    "anomalies_detected": int(df["anomaly"].sum()),
    "note": (
        "AQI_Proxy is derived from available GPY10 and MQ135 "
        "sensor values. It is not an official CPCB AQI because "
        "the dataset does not contain numeric PM2.5 concentration."
    )
}

with open(
    os.path.join(OUTPUT_DIR, "training_metrics.json"),
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

# SAVE FORECAST
future_df = pd.DataFrame({
    "step": np.arange(1, FORECAST_STEPS + 1),
    "lstm_forecast": future_lstm
})

if arima_future is not None:
    future_df["arima_forecast"] = np.array(
        arima_future
    )

future_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        f"{FORECAST_STEPS}_step_forecast.csv"
    ),
    index=False
)

# FINAL RESULT
print("\n==============================================")
print("           TRAINING COMPLETED")
print("==============================================")

print("\nDataset:")
print("  Total rows      :", len(df))
print("  Training rows   :", len(train_df))
print("  Validation rows :", len(val_df))
print("  Testing rows    :", len(test_df))

print("\nLSTM:")
print("  MAE  :", round(lstm_mae, 4))
print("  RMSE :", round(lstm_rmse, 4))

if arima_mae is not None:
    print("\nARIMA (1,1,1):")
    print("  MAE  :", round(arima_mae, 4))
    print("  RMSE :", round(arima_rmse, 4))

print("\nAnomaly Detection:")
print("  Anomalies:", int(df["anomaly"].sum()))

print("\nSaved files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(" ", filename)

print("\nAll training outputs are inside:")
print(os.path.abspath(OUTPUT_DIR))

print("\nIMPORTANT:")
print("The report requires CPCB-compliant AQI.")
print("Your current dataset has no numeric PM2.5 column.")
print("Therefore this training code uses AQI_Proxy.")
print("For actual CPCB AQI, add real PM2.5 data and replace")
print("the AQI_Proxy calculation with the CPCB AQI calculation.")