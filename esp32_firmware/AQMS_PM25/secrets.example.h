// Copy this file to secrets.h (same folder) and fill in your values.
// secrets.h is ignored by git, so the Wi-Fi password never goes to GitHub.
#pragma once

const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// AWS server (Elastic IP of air-quality-v3)
const char* SERVER_URL = "http://13.205.111.95:5000/api/ingest";
// must match AQMS_API_KEY on the server (deploy/aqms.service or /etc/aqms.env)
const char* API_KEY    = "aqms-esp32-key";
