## Summary

<!-- One or two sentences: what does this PR do and why? -->

## What changed

<!-- List the main changes. Mention files/areas touched. -->
-
-

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] UI / design change
- [ ] Model / forecasting change
- [ ] Config / deployment / CI change
- [ ] Docs only

## How I tested it

<!-- e.g. ran the app locally with fake data:
     python tools/seed_fake_data.py --db test.db --days 14 --cleanup
     AQMS_DB=test.db python app.py   +   python tools/simulate_esp32.py
     python tools/smoke_check.py --url http://127.0.0.1:5000 --ingest -->

## Screenshots (for UI changes)

<!-- Before / after, desktop and phone if the layout changed. -->

## Checklist

- [ ] The **Core functionality check** on this PR is green
- [ ] Dashboard (`/`), forecast (`/forecast`) and system (`/about`) pages open without errors
- [ ] Sensor data still shows on the dashboard (ingest -> latest -> charts)
- [ ] No secrets added in plain text beyond what is already in the repo (API keys, AWS keys, `.pem` files)
- [ ] No database files (`*.db`) or `venv/` committed
- [ ] `requirements.txt` updated if I added a Python package
- [ ] Old ESP32 firmware still works (fields `gpy10`, `mq135`, `temperature`, `humidity`, `bmp280`, `aqi` via the aislyntech API)
- [ ] README / PLAN.md updated if behaviour or setup changed

> Merging into `main` deploys to the EC2 automatically. If the checks fail after merge, the deploy is skipped and the server keeps the previous version.
