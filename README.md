# AQMS v2: PM2.5 + CO, CPCB AQI, 5 forecasting models

This version implements the revised paper (AQMS_Revised_Paper_final_v5).

| Paper section | File |
|---|---|
| 4.3 MQ-7 CO, ppm → mg/m³ with measured T and P (Eq. 9–11) | `esp32_firmware/AQMS_ESP32/AQMS_ESP32.ino`, `aqi_engine.py` |
| 4.4 PMS5003 humidity correction (Eq. 12) | `aqi_engine.py` |
| 4.5 CPCB sub-index + max operator (Eq. 13–14, Table 3): real-time, hourly and CPCB-aligned AQI | `aqi_engine.py`, `app.py` |
| 4.6 Persistence, ARIMA, Random Forest, XGBoost, LSTM, GRU (Table 4), 70/15/15 split, MAE/RMSE/R² | `train.py`, `forecasting.py` |
| 4.7 Staged fan control with hysteresis (Table 5), purification R%, k, t50, t90, CADR (Eq. 18–19, Table 9) | `aqi_engine.py`, `app.py` |
| 5.1 Filtering, duplicates, moving median, dashboard, alerts, Table 6 channel trend forecast | `app.py`, `templates/` |
| 5.2 Pin map (Table 7) | firmware |

```
air-quality-aws/
├── app.py                  Flask server: /api/ingest, dashboard, forecasts, fan control
├── aqi_engine.py           CPCB AQI, corrections, filters, controller, purification maths
├── forecasting.py          model loading + live 12-h inference for all 6 models
├── train.py                trains/compares the 5 models (+ persistence)
├── synthetic_data.py       bootstrap data until real PM2.5/CO data exists
├── trained_models/         saved models + training_metrics.json + test predictions
├── templates/              index.html (live), forecast.html (models), about.html
├── esp32_firmware/AQMS_ESP32/AQMS_ESP32.ino
├── tools/simulate_esp32.py fake ESP32, useful for testing the cloud
├── deploy/                 setup_ec2.sh, update_existing_vm.sh, aqms.service
└── legacy/                 your old app.py / train.py / dataset for reference
```

## Data path

```
ESP32 --(Wi-Fi, HTTP POST JSON every 5 s)--> http://<EC2_IP>:5000/api/ingest
      <-- {"fan_stage":1,"pwm":128,...} ----  (fan command from current + forecast AQI)
```
The ESP32 now talks to your EC2 server directly. The old aislyntech PHP API is not
needed. If you want to keep it, set `USE_LEGACY_API = true` in the firmware and
`LEGACY_POLL=1` in `deploy/aqms.service`. The PHP script only stores the
columns it already knows, so PM2.5 and CO arrive only through the direct path.

---

## 1. Wiring (paper Table 7)

| Device | ESP32 pin | Notes |
|---|---|---|
| PMS5003 TX → | GPIO16 (RX2) | PMS5003 needs 5 V power; its TX is 3.3 V logic |
| PMS5003 RX ← | GPIO17 (TX2) | optional |
| MQ-7 AOUT | GPIO35 | through 10 kΩ / 20 kΩ divider (5 V → 3.3 V) |
| MQ-135 AOUT | GPIO34 | through 10 kΩ / 20 kΩ divider |
| GP2Y1010 Vo | GPIO36 (VP) | through divider |
| GP2Y1010 LED | GPIO32 | 150 Ω + 220 µF; LED is active LOW |
| DHT22 data | GPIO4 | 10 kΩ pull-up |
| BMP280 SDA / SCL | GPIO21 / GPIO22 | I²C 0x76 |
| L298N ENA / IN1 / IN2 | GPIO26 / GPIO27 / GPIO14 | fan PWM |
| Status LEDs (optional) | GPIO18 green, GPIO19 red | |

These pins differ from your old sketch: the old sketch used GPY10 on 35/23 and L298N on 25/26/27.
If you don't want to rewire, change the `#define`s at the top of the sketch.

**MQ-7:** give it a 48-hour burn-in once. On first boot, after the 3-minute warm-up, the
firmware measures R0 in clean air and saves it to flash. Make sure the node is in
clean air the first time. To recalibrate later, type `CAL` in the Serial Monitor.
Check the load resistor on your MQ-7 module (usually 10 kΩ, some boards 1 kΩ) and set
`MQ_RL_KOHM` to match.

## 2. Flash the ESP32

1. Install the Arduino IDE 2.x. Open **File → Preferences → Additional boards manager URLs** and add
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`.
2. **Tools → Board → Boards Manager**: install **esp32 by Espressif** (3.x; compile-tested on 3.0.7).
3. **Library Manager**: install **DHT sensor library** (Adafruit), **Adafruit Unified Sensor**,
   **Adafruit BMP280 Library** and **Adafruit BusIO**.
4. Open `esp32_firmware/AQMS_ESP32/AQMS_ESP32.ino` and edit the top block:
   - `WIFI_SSID`, `WIFI_PASSWORD`
   - `SERVER_URL = "http://<your EC2 public IP>:5000/api/ingest"`
   - `API_KEY`: must equal `AQMS_API_KEY` in `deploy/aqms.service`
5. Select **Tools → Board → ESP32 Dev Module** and the COM port, then click **Upload**.
   If it hangs at "Connecting…", hold the **BOOT** button until upload starts.
6. Open the Serial Monitor at 115200. Every 5 s you should see the readings and
   `Server 200 -> fan stage 0, pwm 0`.

## 3a. Update the VM that is already running

Run these commands on your PC. The key is your `.pem` file, and `<IP>` is the EC2 public IP.

```bash
scp -i key.pem air-quality-aws-v2.zip ubuntu@<IP>:/home/ubuntu/
ssh -i key.pem ubuntu@<IP>
unzip -o -q air-quality-aws-v2.zip -d /tmp/x && cp /tmp/x/air-quality-aws/deploy/update_existing_vm.sh ~/
bash ~/update_existing_vm.sh
```
The script:
- stops the old app (systemd service, or a `python app.py` started with nohup/screen)
- backs up the old folder, database included
- copies the new code over the old folder but keeps your `sensor_data.db`. Old rows stay in the `sensor_data` table and new rows go into `readings`.
- installs the requirements, retrains the models on the server and starts the `aqms` systemd service

It assumes an **Ubuntu** AMI with the project in `/home/ubuntu/air-quality-aws`. If your folder or user is different
(Amazon Linux uses `ec2-user` and `dnf` instead of `apt`), edit the paths in the two scripts and in `aqms.service`.

## 3b. Start a new VM from scratch

1. EC2 console → **Launch instance**: Ubuntu 24.04. Use **t3.small (2 GB) or larger** so that
   TensorFlow can run LSTM/GRU. A t2.micro runs the other 3 models only.
2. Security group inbound rules: **SSH 22** (your IP) and **Custom TCP 5000** (0.0.0.0/0).
3. Attach an **Elastic IP** so the address in the ESP32 firmware does not change when the VM restarts.
4. Upload and run:
   ```bash
   scp -i key.pem air-quality-aws-v2.zip ubuntu@<IP>:/home/ubuntu/
   ssh -i key.pem ubuntu@<IP>
   sudo apt-get install -y unzip && unzip -q air-quality-aws-v2.zip
   cd air-quality-aws && bash deploy/setup_ec2.sh
   ```
   Training takes about 5–15 min on a small instance.
5. Open `http://<IP>:5000` in a browser.

## 4. Check it works

```bash
curl http://<IP>:5000/api/health            # models_loaded should list all 5
# no hardware yet? pretend to be the ESP32:
python3 tools/simulate_esp32.py --url http://<IP>:5000/api/ingest --event
sudo journalctl -u aqms -f                  # server logs
sudo systemctl restart aqms                 # after changing code or settings
```

## 5. Retrain on your real data (important for the paper)

The shipped models are trained on **synthetic bootstrap data**. The dashboard shows a
yellow banner while this is the case. **Do not report the synthetic test scores in the paper.**
After the node has logged at least 7 days of PM2.5/CO:
```bash
cd ~/air-quality-aws && AQMS_DB=/data/sensor_data.db ./venv/bin/python train.py --source db && sudo systemctl restart aqms
```
The ML forecast page then shows Table 8 computed from your own data.

## Data storage and retention

- The database lives on a separate EBS volume: `AQMS_DB=/data/sensor_data.db` (set in `deploy/aqms.service`).
- Raw readings are kept for `RETENTION_DAYS` (10). Before deletion their AQI values are copied to the
  `aqi_history` table, which is **never** cleaned. Lifetime AQI: `GET /api/aqi_history?days=N`.
- The cleanup runs inside the app every 6 h; run it by hand with
  `AQMS_DB=/data/sensor_data.db ./venv/bin/python retention.py --dry-run`.
- Old firmware (GP2Y1010/MQ-135 only, via the aislyntech PHP API): with no PM2.5/CO, the device's own
  `aqi` is used as the AQI (`dominant = "Dust (device AQI)"`), so the dashboard, forecasting and training still work.

## Troubleshooting

| Symptom (ESP32 Serial Monitor) | Fix |
|---|---|
| `Server error -1: connection refused` | Security group is missing TCP 5000, the IP in `SERVER_URL` is wrong, or the service is down (`systemctl status aqms`) |
| `Server error 401` | `API_KEY` in firmware ≠ `AQMS_API_KEY` in the service file |
| `PM2.5 --` | PMS5003 TX not on GPIO16, sensor not on 5 V, or loose wiring |
| `CO --  [warming up]` | normal for the first 3 minutes |
| LSTM/GRU cards say "TensorFlow not installed" | instance < 2 GB RAM, or run `./venv/bin/pip install -r requirements-dl.txt` then retrain |
| Dashboard says "ESP32 offline" | no frame for 30 s. Check Wi-Fi and the Serial Monitor |

## Settings (environment variables in `deploy/aqms.service`)
`AQMS_API_KEY`, `LEGACY_POLL`, `LEGACY_API_URL`, `ROOM_VOLUME_M3` (30), `ENCLOSURE_VOLUME_M3` (1),
`FAN_Q_M3_MIN` (0.15), `CO_EVENT_PPM` (20), `DYNAMODB_TABLE` (optional mirror to DynamoDB, needs
`boto3` and an IAM role with `dynamodb:PutItem`), `AWS_REGION`.
