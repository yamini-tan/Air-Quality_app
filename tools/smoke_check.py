"""
Core functionality check (not unit tests).

Verifies only what a user would notice if a change broke the site: pages load
(no 404 / 500 / blank page) and every dashboard API answers with valid JSON and data.
Deliberately independent of the page design, so UI changes do not break it.
Standard library only.

  # PR check: fresh app + fake ESP32 frames (writes test data - never use on production)
  python tools/smoke_check.py --url http://127.0.0.1:5000 --ingest

  # after deploy: read-only check of the live server
  python tools/smoke_check.py --url http://<EC2_IP>:5000
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

results = []


def req(base, path, method="GET", body=None, headers=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", **(headers or {})} if data else (headers or {})
    r = urllib.request.Request(base + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8", "replace")


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail and not ok else ""))


def page(base, path, must_contain):
    code, ctype, text = req(base, path)
    check(f"page {path} loads (200, HTML)", code == 200 and "text/html" in ctype, f"HTTP {code} {ctype}")
    check(f"page {path} is not blank", len(text) > 2000 and all(m in text for m in must_contain),
          f"{len(text)} bytes, expected markers {must_contain}")


def api(base, path, expect=None):
    """GET a JSON API; `expect(j)` returns (ok, detail) for content checks."""
    code, ctype, text = req(base, path)
    try:
        j = json.loads(text)
    except ValueError:
        check(f"api {path} returns JSON (200)", False, f"HTTP {code}, not JSON: {text[:80]!r}")
        return None
    check(f"api {path} returns JSON (200)", code == 200, f"HTTP {code}")
    if expect:
        ok, detail = expect(j)
        check(f"api {path} has data", ok, detail)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--key", default="aqms-esp32-key")
    ap.add_argument("--ingest", action="store_true", help="send fake ESP32 frames first (NOT for production)")
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for the server to come up")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    print(f"Core functionality check against {base}")
    # wait until the server answers AND the models finished loading (background thread after a restart)
    t0, up = time.time(), False
    while True:
        try:
            code, _, text = req(base, "/api/health", timeout=5)
            up = code == 200
            if up and json.loads(text).get("models_loaded"):
                break
        except Exception:
            pass
        if time.time() - t0 > a.wait:
            if not up:
                print("  [FAIL] server did not answer /api/health")
                return 1
            break   # up but models not loaded: the health check below reports it
        time.sleep(2)

    if a.ingest:
        print("- ingest (fake ESP32)")
        for i in range(3):
            frame = {"device": "smoke-check", "seq": 900000 + i, "pm25": 40 + i, "pm10": 55, "co_ppm": 1.2,
                     "mq135": 1500, "gpy10": 420, "temperature": 28.0, "humidity": 60.0, "pressure": 912.5}
            code, _, text = req(base, "/api/ingest", "POST", frame, {"X-API-Key": a.key})
            try:
                ok = json.loads(text).get("ok") is True
            except ValueError:
                ok = False
            check(f"POST /api/ingest frame {i + 1} accepted", code == 200 and ok, f"HTTP {code} {text[:120]}")
        code, _, _ = req(base, "/api/ingest?key=" + a.key + "&gpy10=450&mq135=960&temperature=30&humidity=55"
                               "&bmp280=913&aqi=45&air_status=GOOD&fan=OFF")
        check("GET /api/ingest (current firmware field names) accepted", code == 200, f"HTTP {code}")
        code, _, _ = req(base, "/api/ingest", "POST", {"pm25": 1}, {"X-API-Key": "wrong-key"})
        check("wrong API key rejected (401)", code == 401, f"HTTP {code}")

    print("- pages")
    page(base, "/", ["AirMonitor", 'id="device"', "/api/latest"])
    page(base, "/forecast", ["AirMonitor", "/api/forecast"])
    page(base, "/about", ["AirMonitor"])
    code, _, _ = req(base, "/prediction")
    check("old /prediction link still works (redirect)", code == 200, f"HTTP {code}")

    print("- APIs")
    api(base, "/api/health", lambda j: (j.get("ok") is True and len(j.get("models_loaded", [])) >= 3,
                                        f"ok={j.get('ok')} models={j.get('models_loaded')} errors={j.get('model_errors')}"))
    api(base, "/api/status", lambda j: (j.get("state") in ("online", "delayed", "offline", "never"), f"state={j.get('state')}"))
    latest = api(base, "/api/latest", lambda j: ("success" in j and ("device" in j), f"keys={list(j)[:8]}"))
    api(base, "/api/history?hours=24", lambda j: (j.get("success") is True and "data" in j, str(j)[:120]))
    api(base, "/api/forecast", lambda j: ("forecast" in j or "message" in j, str(j)[:120]))
    api(base, "/api/aqi_history?days=30", lambda j: (j.get("success") is True, str(j)[:120]))
    api(base, "/api/purification", lambda j: (j.get("success") is True and len(j.get("theoretical", [])) == 2, str(j)[:120]))
    api(base, "/api/test_predictions", lambda j: ("data" in j, str(j)[:120]))
    api(base, "/api/sensor_data", lambda j: (j.get("success") is True and isinstance(j.get("data"), list), str(j)[:120]))
    api(base, "/api/predict_future", lambda j: ("success" in j, str(j)[:120]))

    if a.ingest:
        print("- data reaches the dashboard")
        check("latest reading is shown", bool(latest and latest.get("success") and latest.get("reading")),
              str(latest)[:160])
        rt = ((latest or {}).get("aqi") or {}).get("realtime") or {}
        check("real-time AQI computed", rt.get("aqi") is not None, f"realtime={rt}")
        api(base, "/api/status", lambda j: (j.get("state") == "online", f"state={j.get('state')}"))
        api(base, "/api/history?hours=1", lambda j: (len(j.get("recent", [])) > 0, f"recent={len(j.get('recent', []))}"))

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for name, _, detail in failed:
        print(f"FAILED: {name}  {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
