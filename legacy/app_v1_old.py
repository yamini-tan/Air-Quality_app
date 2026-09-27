from flask import Flask, render_template, jsonify
import requests
import sqlite3
from datetime import datetime, timezone, timedelta
import time
import warnings

# For ARIMA Forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    raise ImportError("Please install statsmodels and pandas: pip install statsmodels pandas")

app = Flask(__name__)

# CONFIGURATION
API_URL = "https://aislyntech.com/Api/42-get.php"
DATABASE = "sensor_data.db"

# GLOBAL HARDWARE STATES
last_hardware_ping = 0
last_known_values = None
last_value_change_time = time.time()

# DATABASE CONNECTION
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# CREATE / UPDATE DATABASE SCHEMA
def create_database():
    conn = get_db()
    cursor = conn.cursor()

    # Create base table if it doesn't exist at all
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    """)
    
    # Dictionary of all expected columns to auto-repair old databases
    expected_columns = {
        "gpy10": "REAL DEFAULT 0",
        "mq135": "REAL DEFAULT 0",
        "temperature": "REAL DEFAULT 0",
        "humidity": "REAL DEFAULT 0",
        "bmp280": "REAL DEFAULT 0",
        "aqi": "INTEGER DEFAULT 0",
        "gpy10_status": "TEXT DEFAULT 'UNKNOWN'",
        "mq135_status": "TEXT DEFAULT 'UNKNOWN'",
        "temperature_status": "TEXT DEFAULT 'UNKNOWN'",
        "humidity_status": "TEXT DEFAULT 'UNKNOWN'",
        "pressure_status": "TEXT DEFAULT 'UNKNOWN'",
        "air_status": "TEXT DEFAULT 'UNKNOWN'",
        "created_at": "TEXT"
    }

    # Fetch existing columns
    cursor.execute("PRAGMA table_info(sensor_data)")
    existing_columns = [col["name"] for col in cursor.fetchall()]
    
    # Safely add any missing columns without crashing
    for col_name, col_type in expected_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE sensor_data ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


# SAFE FLOAT / INT
def safe_float(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return float(value)
    except Exception:
        return default

def safe_int(value, default=0):
    try:
        return int(safe_float(value, default))
    except Exception:
        return default


# STATUS CLASSIFICATIONS
def classify_gpy10(value):
    value = safe_float(value)
    if 400 <= value <= 484:
        return "GOOD"
    elif 485 <= value <= 499:
        return "POOR"
    elif value >= 500:
        return "VERY POOR"
    return "LOW"

def classify_mq135(value):
    value = safe_float(value)
    if value <= 2350:
        return "GOOD"
    elif 2351 <= value <= 2390:
        return "MODERATE"
    elif value >= 2391:
        return "POOR"
    return "UNKNOWN"

def classify_temperature(value):
    value = safe_float(value)
    if value < 0:
        return "VERY_LOW"
    elif value > 30:
        return "HIGH"
    return "NORMAL"

def classify_humidity(value):
    value = safe_float(value)
    if value < 30:
        return "LOW"
    elif value > 60:
        return "HIGH"
    return "NORMAL"

def classify_pressure(value):
    value = safe_float(value)
    if 0 <= value <= 900:
        return "LOW"
    elif 901 <= value <= 1020:
        return "NORMAL"
    elif value >= 1021:
        return "HIGH"
    return "UNKNOWN"

def calculate_air_status(gpy10_status, mq135_status, temperature_status):
    if gpy10_status == "VERY POOR":
        return "VERY POOR"
    if gpy10_status == "POOR" or mq135_status == "POOR":
        return "POOR"
    if mq135_status == "MODERATE":
        return "MODERATE"
    if temperature_status == "HIGH":
        return "HIGH TEMPERATURE"
    return "GOOD"


# EXTRACT SENSOR DATA
def extract_sensor_data(api_response):
    data = api_response
    
    root_id = data.get("id") if isinstance(data, dict) else None

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Hardware connection error: No recent data found from ESP32.")
        data = data[0]

    if not isinstance(data, dict) or not data:
        raise ValueError("Hardware connection error: Invalid or missing data payload from API.")

    gpy10 = data.get("gpy10", data.get("GPY10", data.get("gpy10_value", 0)))
    mq135 = data.get("mq135", data.get("MQ135", data.get("mq135_value", 0)))
    temperature = data.get("temperature", data.get("Temperature", data.get("temp", 0)))
    humidity = data.get("humidity", data.get("Humidity", data.get("hum", 0)))
    bmp280 = data.get("bmp280", data.get("BMP280", data.get("pressure", data.get("bmp280_pressure", 0))))
    aqi = data.get("aqi", data.get("AQI", 0))
    
    fan = data.get("fan")
    if fan is None:
        fan = "OFF"

    return {
        "id": data.get("id") or root_id, 
        "gpy10": safe_float(gpy10),
        "mq135": safe_float(mq135),
        "temperature": safe_float(temperature),
        "humidity": safe_float(humidity),
        "bmp280": safe_float(bmp280),
        "aqi": safe_int(aqi),
        "api_air_status": data.get("air_status"),
        "fan": str(fan).upper()
    }


# ARIMA PREDICTION LOGIC
def predict_with_arima(y_values, steps=7200):
    if len(y_values) < 10 or all(v == 0 for v in y_values[-5:]):
        return None 
        
    if len(set(y_values)) == 1:
        return y_values[-1]

    try:
        warnings.filterwarnings("ignore")
        model = ARIMA(y_values, order=(1, 1, 1))
        model_fit = model.fit()
        forecast_result = model_fit.forecast(steps=steps)
        predicted_val = list(forecast_result)[-1]
        return round(float(predicted_val), 2)
        
    except Exception:
        window_size = min(5, len(y_values))
        return round(sum(y_values[-window_size:]) / window_size, 2)


# DASHBOARD ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# FETCH AND STORE FROM ESP32
@app.route("/api/fetch_and_store")
def fetch_and_store():
    global last_hardware_ping, last_known_values, last_value_change_time
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json"
        }
        response = requests.get(API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        
        try:
            api_response = response.json()
        except Exception:
            raise ValueError(f"API returned invalid JSON. Server responded with: {response.text[:100]}...")

        sensor = extract_sensor_data(api_response)

        gpy10 = sensor["gpy10"]
        mq135 = sensor["mq135"]
        temperature = sensor["temperature"]
        humidity = sensor["humidity"]
        bmp280 = sensor["bmp280"]
        aqi = sensor["aqi"]
        fan = sensor["fan"] 

        # ---------------------------------------------------------
        # FREEZE DETECTOR: If values don't change for 15s, mark offline
        # ---------------------------------------------------------
        current_sensor_hash = (gpy10, mq135, temperature, humidity, bmp280, aqi)
        
        if last_known_values is None:
            last_known_values = current_sensor_hash
            last_value_change_time = time.time()
        elif last_known_values == current_sensor_hash:
            if time.time() - last_value_change_time > 15:
                raise ValueError(" (Hardware not connected) ")
        else:
            # Values changed, update the cache and timer
            last_known_values = current_sensor_hash
            last_value_change_time = time.time()

        # ---------------------------------------------------------
        # FORCE IST TIMESTAMP (UTC + 5:30)
        # ---------------------------------------------------------
        IST = timezone(timedelta(hours=5, minutes=30))
        created_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        # Classifications
        gpy10_status = classify_gpy10(gpy10)
        mq135_status = classify_mq135(mq135)
        temperature_status = classify_temperature(temperature)
        humidity_status = classify_humidity(humidity)
        pressure_status = classify_pressure(bmp280)
        air_status = calculate_air_status(gpy10_status, mq135_status, temperature_status)

        conn = get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sensor_data
                (
                    gpy10, mq135, temperature, humidity, bmp280, aqi,
                    gpy10_status, mq135_status, temperature_status, humidity_status, pressure_status, air_status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gpy10, mq135, temperature, humidity, bmp280, aqi,
                gpy10_status, mq135_status, temperature_status, humidity_status, pressure_status, air_status,
                created_at
            ))
            conn.commit()
            record_id = cursor.lastrowid
        finally:
            conn.close()

        last_hardware_ping = time.time()

        return jsonify({
            "success": True,
            "message": "Sensor data fetched",
            "data": {
                "id": record_id,
                "gpy10": gpy10,
                "mq135": mq135,
                "temperature": temperature,
                "humidity": humidity,
                "bmp280": bmp280,
                "aqi": aqi,
                "gpy10_status": gpy10_status,
                "mq135_status": mq135_status,
                "temperature_status": temperature_status,
                "humidity_status": humidity_status,
                "pressure_status": pressure_status,
                "air_status": air_status,
                "fan": fan,
                "created_at": created_at
            }
        })

    except requests.exceptions.RequestException as re:
        return jsonify({"success": False, "message": f"Hardware connection error: {str(re)}"}), 500
    except ValueError as ve:
        return jsonify({"success": False, "message": str(ve)}), 500
    except Exception as e:
        return jsonify({"success": False, "message": "Internal processing error", "error": str(e)}), 500


# GET STORED DATA 
@app.route("/api/sensor_data")
def sensor_data():
    try:
        conn = get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 50")
            rows = cursor.fetchall()

            records = []
            for row in rows:
                r = dict(row) 
                records.append({
                    "id": r.get("id"),
                    "gpy10": r.get("gpy10", 0),
                    "mq135": r.get("mq135", 0),
                    "temperature": r.get("temperature", 0),
                    "humidity": r.get("humidity", 0),
                    "bmp280": r.get("bmp280", 0),
                    "aqi": r.get("aqi", 0),
                    "gpy10_status": r.get("gpy10_status", "UNKNOWN"),
                    "mq135_status": r.get("mq135_status", "UNKNOWN"),
                    "temperature_status": r.get("temperature_status", "UNKNOWN"),
                    "humidity_status": r.get("humidity_status", "UNKNOWN"),
                    "pressure_status": r.get("pressure_status", "UNKNOWN"),
                    "air_status": r.get("air_status", "UNKNOWN"),
                    "created_at": r.get("created_at", "")
                })

            return jsonify({"success": True, "data": records})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ARIMA PREDICTION API
@app.route("/api/predict_future")
def predict_future():
    global last_hardware_ping
    try:
        if time.time() - last_hardware_ping > 15:
            return jsonify({"success": False, "message": "Hardware is offline. Predictions disabled to prevent old data flashing."})

        conn = get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT gpy10, mq135, temperature, humidity FROM sensor_data ORDER BY id DESC LIMIT 40")
            rows = cursor.fetchall()

            if len(rows) < 10:
                return jsonify({"success": False, "message": "Gathering data for ARIMA model... (needs 10 valid points)"})

            rows.reverse()

            gpy_vals = [r["gpy10"] for r in rows]
            mq_vals = [r["mq135"] for r in rows]
            temp_vals = [r["temperature"] for r in rows]
            hum_vals = [r["humidity"] for r in rows]

            FORECAST_STEPS = 7200 
            
            p_gpy10 = predict_with_arima(gpy_vals, steps=FORECAST_STEPS)
            p_mq135 = predict_with_arima(mq_vals, steps=FORECAST_STEPS)
            p_temp = predict_with_arima(temp_vals, steps=FORECAST_STEPS)
            p_hum = predict_with_arima(hum_vals, steps=FORECAST_STEPS)

            if p_gpy10 is None and p_mq135 is None:
                return jsonify({"success": False, "message": "Hardware connection error: No recent valid data to predict from."})

            predictions = {
                "predicted_gpy10": p_gpy10,
                "predicted_mq135": p_mq135,
                "predicted_temperature": p_temp,
                "predicted_humidity": p_hum,
                "forecast_timeframe": "6 Hours"
            }

            return jsonify({"success": True, "data": predictions})
        finally:
            conn.close()

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# DELETE ALL
@app.route("/api/delete_all", methods=["DELETE"])
def delete_all():
    try:
        conn = get_db()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sensor_data")
            conn.commit()
            return jsonify({"success": True, "message": "All sensor data deleted"})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    create_database()
    app.run(host="0.0.0.0", port=5000, debug=True)