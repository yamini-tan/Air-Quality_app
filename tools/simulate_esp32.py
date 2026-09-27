"""
Pretends to be the ESP32: POSTs one JSON frame every few seconds to /api/ingest,
exactly like the firmware does. Use it to test the cloud side before the hardware
is connected.

  python tools/simulate_esp32.py --url http://<EC2_IP>:5000/api/ingest --key aqms-esp32-key
  python tools/simulate_esp32.py --event          # adds a cooking spike to watch the fan react
"""
import argparse
import math
import random
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:5000/api/ingest")
ap.add_argument("--key", default="aqms-esp32-key")
ap.add_argument("--interval", type=float, default=5)
ap.add_argument("--count", type=int, default=0, help="0 = forever")
ap.add_argument("--event", action="store_true", help="simulate a cooking event")
a = ap.parse_args()

seq, i = 0, 0
while a.count == 0 or i < a.count:
    i += 1
    seq += 1
    ev = 0
    if a.event:
        ev = 260 * math.exp(-((i % 240) - 20) ** 2 / 400) if (i % 240) > 5 else 0
    pm = max(1, 38 + 6 * math.sin(i / 50) + random.gauss(0, 2) + ev)
    frame = {
        "device": "simulator", "seq": seq,
        "pm25": round(pm, 1), "pm10": round(pm * 1.35, 1),
        "co_ppm": round(max(0, 1.3 + random.gauss(0, 0.1) + ev / 40), 2),
        "mq135": int(1500 + 1.6 * pm + random.gauss(0, 10)),
        "gpy10": int(340 + 1.9 * pm + random.gauss(0, 12)),
        "temperature": round(28.1 + random.gauss(0, 0.1), 2),
        "humidity": round(68.6 + random.gauss(0, 0.4), 2),
        "pressure": round(912.7 + random.gauss(0, 0.05), 2),
        "fan": 0,
    }
    try:
        r = requests.post(a.url, json=frame, headers={"X-API-Key": a.key}, timeout=8)
        j = r.json()
        print(f"#{seq} PM2.5 {frame['pm25']:6.1f}  CO {frame['co_ppm']:5.2f} ppm -> {r.status_code} "
              f"AQI {j.get('aqi')} {j.get('category')} | fan stage {j.get('fan_stage')} ({j.get('fan_duty')}%)")
    except Exception as e:
        print("error:", e)
    time.sleep(a.interval)
