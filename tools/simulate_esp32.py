"""
Pretends to be the ESP32: POSTs one JSON frame every few seconds to /api/ingest,
exactly like the firmware does. Use it to test the cloud side before the hardware
is connected.

  python tools/simulate_esp32.py --url http://<EC2_IP>:5000/api/ingest --key aqms-esp32-key
  python tools/simulate_esp32.py --event          # adds a cooking spike to watch the fan react
  python tools/simulate_esp32.py --two-esp        # two boards: "aqms-pm" (PM2.5) + "aqms-env" (CO, climate, fan)
  python tools/simulate_esp32.py --two-esp --co-event   # CO rises -> red LED + "open a window" alert
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
ap.add_argument("--two-esp", action="store_true", help="split frames like the two-ESP hardware")
ap.add_argument("--co-event", action="store_true", help="simulate a gas-stove CO rise")
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
    if a.co_event:
        frame["co_ppm"] = round(frame["co_ppm"] + 3.5 * math.exp(-((i % 120) - 30) ** 2 / 300), 2)
    frames = [frame]
    if a.two_esp:   # ESP-A sends only PM, ESP-B everything else (and drives the fan)
        pm_keys = ("pm25", "pm10")
        frames = [{"device": "aqms-pm", "seq": seq, **{k: frame[k] for k in pm_keys}},
                  {"device": "aqms-env", "seq": seq, **{k: v for k, v in frame.items() if k not in pm_keys + ("device", "seq")}}]
    for f in frames:
        try:
            r = requests.post(a.url, json=f, headers={"X-API-Key": a.key}, timeout=8)
            j = r.json()
            print(f"#{seq} {f['device']:<9} PM2.5 {f.get('pm25', '-'):>6}  CO {f.get('co_ppm', '-'):>5} ppm -> {r.status_code} "
                  f"AQI {j.get('aqi')} {j.get('category')} | fan stage {j.get('fan_stage')} ({j.get('fan_duty')}%) | LED {j.get('led')}")
        except Exception as e:
            print("error:", e)
    time.sleep(a.interval)
