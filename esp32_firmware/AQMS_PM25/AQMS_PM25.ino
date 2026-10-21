/*
 * =====================================================================
 *  AQMS - ESP-A : PM2.5 board (two-ESP setup)
 *
 *  Sensor : PMS5003 (PM1.0 / PM2.5 / PM10, laser particle counter)
 *  Every 5 s this board averages the PMS5003 frames it received and
 *  POSTs PM2.5 + PM10 to the AWS server (/api/ingest). The server puts them
 *  together with the CO / climate board's readings (same 1-minute window,
 *  timed by the server clock) and works out the whole AQI. This board does
 *  not drive the fan; ESP-B does.
 *
 *  Wiring (PMS5003 8-pin cable):
 *    PMS VCC (pin 1)  -> 5 V  (VIN)       the fan inside needs 5 V
 *    PMS GND (pin 2)  -> GND
 *    PMS TX  (pin 5)  -> GPIO 16 (RX2)    3.3 V logic, no divider needed
 *    PMS RX  (pin 4)  -> GPIO 17 (TX2)
 *    SET / RESET      -> leave open (internal pull-up = running)
 *  Onboard blue LED (GPIO 2): on = last send to the server worked.
 *
 *  Board: "ESP32 Dev Module". No extra libraries needed.
 *  Wi-Fi name/password, server URL and API key live in secrets.h
 *  (copy secrets.example.h to secrets.h). secrets.h is not stored in git.
 * =====================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "secrets.h"     // WIFI_SSID, WIFI_PASSWORD, SERVER_URL, API_KEY

const char* DEVICE_ID = "aqms-pm";      // the dashboard shows this board as "PM2.5 sensor"

#define PMS_RX_PIN   16     // ESP32 RX2 <- PMS5003 TX
#define PMS_TX_PIN   17     // ESP32 TX2 -> PMS5003 RX
#define STATUS_LED   2

const unsigned long SEND_INTERVAL_MS = 5000;
const unsigned long PMS_STALE_MS     = 10000;   // no frame this long -> sensor problem

unsigned long lastSendMs = 0, lastFrameMs = 0;
uint32_t seqNo = 0;
// running sums of the frames received since the last send
uint32_t sum25 = 0, sum10 = 0, nFrames = 0;

// =====================================================
// PMS5003 PARSER (32-byte frame: 0x42 0x4D header, checksum at the end)
// =====================================================
void readPMS() {
  static uint8_t buf[32];
  static int idx = 0;
  while (Serial2.available()) {
    uint8_t c = Serial2.read();
    if (idx == 0 && c != 0x42) continue;
    if (idx == 1 && c != 0x4D) { idx = 0; continue; }
    buf[idx++] = c;
    if (idx == 32) {
      idx = 0;
      uint16_t sum = 0;
      for (int i = 0; i < 30; i++) sum += buf[i];
      if (sum != ((buf[30] << 8) | buf[31])) continue;      // corrupted frame
      // "atmospheric environment" values (bytes 10-15), ug/m3
      sum25 += (buf[12] << 8) | buf[13];
      sum10 += (buf[14] << 8) | buf[15];
      nFrames++;
      lastFrameMs = millis();
    }
  }
}

// =====================================================
// WIFI (does not block; keeps reading the sensor while reconnecting)
// =====================================================
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  static unsigned long lastTry = 0;
  if (lastTry != 0 && millis() - lastTry < 15000) return;
  lastTry = millis();
  Serial.println("WiFi: connecting...");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

bool sendToServer(const String& json) {
  if (WiFi.status() != WL_CONNECTED) { Serial.println("WiFi not connected - reading kept, not sent"); return false; }
  HTTPClient http;
  http.begin(SERVER_URL);
  http.setConnectTimeout(4000);
  http.setTimeout(6000);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", API_KEY);
  int code = http.POST(json);
  if (code == 200) {
    String body = http.getString();
    int p = body.indexOf("\"aqi\":");
    Serial.printf("Server OK -> whole AQI %s\n", p < 0 ? "?" : body.substring(p + 6, body.indexOf(',', p)).c_str());
  } else {
    Serial.printf("Server error %d %s\n", code, code < 0 ? http.errorToString(code).c_str() : "");
  }
  http.end();
  return code == 200;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("======================================");
  Serial.println("  AIR QUALITY - PM2.5 board (PMS5003)");
  Serial.println("======================================");
  Serial2.begin(9600, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.println("PMS5003 needs ~30 s after power-on for stable readings.");
}

void loop() {
  readPMS();                                  // keep draining the UART (frames arrive ~1/s)
  if (millis() - lastSendMs < SEND_INTERVAL_MS) return;
  lastSendMs = millis();
  ensureWiFi();

  if (nFrames == 0 || millis() - lastFrameMs > PMS_STALE_MS) {
    Serial.println("PMS5003: no data - check 5 V power and the TX -> GPIO 16 wire");
    digitalWrite(STATUS_LED, LOW);
    return;                                   // send nothing: the dashboard then shows this board offline
  }
  float pm25 = sum25 / (float)nFrames, pm10 = sum10 / (float)nFrames;
  sum25 = sum10 = nFrames = 0;

  seqNo++;
  String j = "{\"device\":\"" + String(DEVICE_ID) + "\",\"seq\":" + String(seqNo) +
             ",\"pm25\":" + String(pm25, 1) + ",\"pm10\":" + String(pm10, 1) + "}";
  Serial.printf("#%lu  PM2.5 %.1f ug/m3 | PM10 %.1f ug/m3\n", (unsigned long)seqNo, pm25, pm10);
  digitalWrite(STATUS_LED, sendToServer(j) ? HIGH : LOW);
}
