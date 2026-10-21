# ESP32 firmware

| Sketch | Board | What it does |
|---|---|---|
| `AQMS_PM25/` | **ESP-A** (new) | PMS5003 PM2.5 / PM10 → server every 5 s. No fan. |
| `AQMS_ENV_FAN/` | **ESP-B** | MQ-7 (CO), GP2Y1010 dust, MQ-135, DHT22, BMP280 → server every 5 s; drives the HEPA fan and the red/green LEDs from the server's reply; also keeps sending to the old aislyntech API. |
| `AQMS_ESP32/` | single board | Older all-in-one design (every sensor on one ESP). Not used in the two-ESP setup. |

## How the two boards work together

- Both boards POST to `http://13.205.111.95:5000/api/ingest`. The server stamps each reading with **its own clock** and averages each sensor over the last minute, so the two boards' readings line up without syncing their clocks.
- **Whole AQI** = the bigger of the PM2.5 sub-index and the CO sub-index (CPCB). If the PM2.5 board is off, the dust sensor stands in for PM2.5.
- **Fan** (decided by the server, applied by ESP-B):

  | AQI | Fan |
  |---|---|
  | 130 or more | 100 % |
  | 101 or more (or 12-h forecast ≥ 130) | 50 % |
  | below 90 for 10 min | off |

  If ESP-B gets no reply for 30 s, it runs the same rule on its own dust + CO AQI.
- **LEDs**: **red** when the AQI is above 100 **or** CO is above 2 mg/m³ (1-minute mean), otherwise green. High CO turns the LED red at once on the board itself.
- HEPA traps particles (PM2.5, dust), not CO gas. When CO is high, the dashboard says "open a window".

## Flashing

1. Arduino IDE → board **ESP32 Dev Module**. ESP-B needs the libraries *DHT sensor library*, *Adafruit Unified Sensor*, *Adafruit BMP280 Library* and *Adafruit BusIO*.
2. In each sketch folder copy `secrets.example.h` to `secrets.h` and fill in your Wi-Fi details and the API key. `secrets.h` is git-ignored.
3. Upload `AQMS_PM25` to ESP-A and `AQMS_ENV_FAN` to ESP-B.
4. Open the Serial Monitor at 115200 to see the readings and the server's reply.

## New wiring

**ESP-A – PMS5003**

| PMS5003 | ESP32 |
|---|---|
| VCC (pin 1) | 5 V / VIN |
| GND (pin 2) | GND |
| TX (pin 5) | GPIO 16 |
| RX (pin 4) | GPIO 17 |

**ESP-B – MQ-7** (all other pins as before: GPY10 23/35, MQ-135 34, DHT22 4, BMP280 21/22, L298N 25/26/27, LEDs 18 green / 19 red)

| MQ-7 module | ESP32 |
|---|---|
| VCC | 5 V |
| GND | GND |
| AO | 10 kΩ → **GPIO 32**, and GPIO 32 → 20 kΩ → GND |

The divider matters: the MQ-7 output can reach 5 V, and an ESP32 pin takes at most 3.3 V.

## MQ-7 calibration

- After power-on the MQ-7 heats for 3 minutes; CO is reported only after that.
- On the very first boot it then measures its clean-air resistance R0 for 20 s and saves it. **Do the first boot in clean air** (no stove or smoke nearby).
- To redo it later: Serial Monitor → type `CAL` → Enter, in clean air.
- Low-cost MQ-7 modules give an estimate, not a lab value. Check the CO trend on the dashboard, not single readings.

## Test without hardware

```
python tools/simulate_esp32.py --two-esp              # both boards, normal air
python tools/simulate_esp32.py --two-esp --co-event   # CO rises -> red LED + "open a window"
```
