/*
 * =====================================================================
 *  AQMS - ESP-B : CO + climate sensors + HEPA fan + status LEDs
 *  (two-ESP setup; the PM2.5 sensor is on ESP-A, sketch AQMS_PM25)
 *
 *  Sensors : MQ-7 (CO), GP2Y1010 (dust), MQ-135 (gas index),
 *            DHT22 (temperature, humidity), BMP280 (pressure)
 *  Outputs : L298N -> fan with HEPA filter, green + red LED
 *
 *  Every 5 s this board:
 *    1. reads all its sensors
 *    2. POSTs them to the AWS server (/api/ingest). The server combines them
 *       with the PM2.5 board's readings, works out the WHOLE AQI
 *       (PM2.5 + CO) and replies with the fan speed and the LED colour.
 *    3. sets the fan and LEDs from that reply
 *    4. also sends to the old aislyntech API (keeps the old system working)
 *
 *  LEDs : RED when the AQI is bad (> 100) OR the CO reading is bad
 *         (> 2 mg/m3, 1-min mean), otherwise GREEN. High CO turns the LED
 *         red at once, without waiting for the server.
 *  Fan  : follows the server (staged 0 / 50 / 100 %, with a 10-min calm-down).
 *         No reply for 30 s (Wi-Fi or server down) -> same rule run locally
 *         on this board's own AQI (dust + CO), so the fan keeps working.
 *
 *  Wiring that is NEW compared with the old sketch:
 *    MQ-7 AO -> 10k -> GPIO 32 -> 20k -> GND   (divider: MQ-7 output is up to 5 V,
 *    MQ-7 VCC -> 5 V, GND -> GND                ESP32 pins take max 3.3 V)
 *    GPIO 32 is on ADC1 (ADC2 pins stop working while Wi-Fi is on).
 *  Everything else uses the same pins as before.
 *
 *  Libraries (Arduino IDE -> Library Manager):
 *    "DHT sensor library" (Adafruit) + "Adafruit Unified Sensor"
 *    "Adafruit BMP280 Library" + "Adafruit BusIO"
 *  Board: "ESP32 Dev Module"
 *
 *  Wi-Fi name/password, server URL and API key live in secrets.h
 *  (copy secrets.example.h to secrets.h). secrets.h is not stored in git.
 *
 *  Serial Monitor (115200): type CAL in clean air to re-calibrate the MQ-7.
 * =====================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Preferences.h>
#include <Adafruit_BMP280.h>
#include "DHT.h"
#include "secrets.h"     // WIFI_SSID, WIFI_PASSWORD, SERVER_URL, API_KEY

const char* DEVICE_ID = "aqms-env";     // the dashboard shows this board as "CO, climate sensors + fan"
const char* LEGACY_API_URL = "https://aislyntech.com/Api/42-update.php";
const bool  USE_LEGACY_API = true;      // keep feeding the old system

// =====================================================
// PIN DEFINITIONS (same as the old sketch + MQ-7)
// =====================================================
#define GPY10_LED_PIN     23
#define GPY10_ANALOG_PIN  35
#define MQ135_PIN         34
#define MQ7_PIN           32     // NEW: via 10k/20k divider
#define DHT_PIN           4
#define DHT_TYPE          DHT22
#define SDA_PIN           21
#define SCL_PIN           22
#define ENA_PIN           25
#define IN1_PIN           26
#define IN2_PIN           27
#define GREEN_LED_PIN     18
#define RED_LED_PIN       19

// =====================================================
// LIMITS
// =====================================================
const int   AQI_BAD    = 100;    // LED red above this AQI  (same as the server)
const float CO_BAD_MG  = 2.0;    // LED red above this CO, mg/m3 (CPCB CO sub-index > 100)

// GPY10 dust ranges (unchanged from the old sketch)
#define GPY_GOOD_MAX       500
#define GPY_MODERATE_MAX   1000
#define GPY_POOR_MAX       1800
#define GPY_VERY_POOR_MAX  2500

// =====================================================
// MQ-7 CALIBRATION
// =====================================================
const float MQ_VC         = 5.0;     // module supply (V)
const float MQ_RL_KOHM    = 10.0;    // load resistor on the module (most modules: 10k; check yours)
const float DIVIDER_RATIO = 1.5;     // 10k/20k divider: Vmodule = Vpin * 30/20
const float MQ7_A         = 99.042;  // CO ppm = A * (Rs/R0)^B  (datasheet curve)
const float MQ7_B         = -1.518;
const float MQ7_CLEAN_AIR_RATIO = 27.5;          // Rs/R0 in clean air (datasheet)
const unsigned long MQ7_WARMUP_MS = 180000UL;    // 3 min heater warm-up before CO is used
// R0 is measured once in clean air after the first warm-up and saved in flash.

// =====================================================
// TIMING / FAN
// =====================================================
const unsigned long SAMPLE_INTERVAL_MS = 5000;
const unsigned long SERVER_TIMEOUT_MS  = 30000;  // no server reply this long -> local control
const int PWM_STAGE[3] = {0, 128, 255};          // stage 0 / 1 / 2 = 0 / 50 / 100 %

// =====================================================
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;
Preferences prefs;
bool bmpOK = false;

float mq7R0 = 0;                          // kOhm
unsigned long bootMs = 0, lastSampleMs = 0, lastServerOkMs = 0, belowSinceMs = 0;
uint32_t seqNo = 0;
int fanStage = 0, fanPwm = 0;
bool serverRed = false;                   // LED colour from the server's last reply
bool serverControl = false;

const int AVG_N = 12;                     // 12 x 5 s = 1-min mean (same window as the server)
float coBuf[AVG_N]; int coIdx = 0, coCount = 0;

// =====================================================
// SENSORS
// =====================================================
int readGPY10() {                         // same timing as the old sketch, mean of 5 pulses
  long acc = 0;
  for (int i = 0; i < 5; i++) {
    digitalWrite(GPY10_LED_PIN, LOW);
    delayMicroseconds(280);
    acc += analogRead(GPY10_ANALOG_PIN);
    delayMicroseconds(40);
    digitalWrite(GPY10_LED_PIN, HIGH);
    delayMicroseconds(9680);
  }
  return acc / 5;
}

float mq7Rs() {                           // Rs = RL (Vc - Vout) / Vout
  uint32_t acc = 0;
  for (int i = 0; i < 32; i++) { acc += analogReadMilliVolts(MQ7_PIN); delayMicroseconds(200); }
  float vout = acc / 32.0 / 1000.0 * DIVIDER_RATIO;
  vout = constrain(vout, 0.01, MQ_VC - 0.01);
  return MQ_RL_KOHM * (MQ_VC - vout) / vout;
}

void calibrateMQ7() {
  Serial.println("Calibrating MQ-7 in clean air (20 s)... keep the sensor away from smoke and gas");
  float acc = 0;
  for (int i = 0; i < 40; i++) { acc += mq7Rs(); delay(500); }
  mq7R0 = (acc / 40.0) / MQ7_CLEAN_AIR_RATIO;
  prefs.putFloat("r0", mq7R0);
  Serial.printf("MQ-7 R0 = %.3f kOhm (saved)\n", mq7R0);
}

float pushAvg(float v) {
  coBuf[coIdx] = v; coIdx = (coIdx + 1) % AVG_N; if (coCount < AVG_N) coCount++;
  float s = 0; for (int k = 0; k < coCount; k++) s += coBuf[k];
  return s / coCount;
}

// =====================================================
// LOCAL AQI (used for the LEDs at once and when the server is unreachable)
// =====================================================
int dustAQI(int g) {                      // unchanged mapping from the old sketch
  int a;
  if (g <= GPY_GOOD_MAX)           a = map(g, 0, GPY_GOOD_MAX, 0, 50);
  else if (g <= GPY_MODERATE_MAX)  a = map(g, GPY_GOOD_MAX + 1, GPY_MODERATE_MAX, 51, 100);
  else if (g <= GPY_POOR_MAX)      a = map(g, GPY_MODERATE_MAX + 1, GPY_POOR_MAX, 101, 200);
  else if (g <= GPY_VERY_POOR_MAX) a = map(g, GPY_POOR_MAX + 1, GPY_VERY_POOR_MAX, 201, 300);
  else                             a = map(g, GPY_VERY_POOR_MAX + 1, 4095, 301, 500);
  return constrain(a, 0, 500);
}

float coSubIndex(float mg) {              // CPCB CO breakpoints (mg/m3)
  const float bp[6][4] = {{0,1.0,0,50},{1.1,2.0,51,100},{2.1,10,101,200},{10.1,17,201,300},{17.1,34,301,400},{34.1,50,401,500}};
  if (mg < 0) mg = 0;
  for (int i = 0; i < 6; i++)
    if (mg <= bp[i][1]) return constrain((bp[i][3]-bp[i][2])/(bp[i][1]-bp[i][0])*(mg-bp[i][0])+bp[i][2], bp[i][2], bp[i][3]);
  return 500;
}

const char* airStatus(int aqi) {          // same words the old API received
  if (aqi <= 50) return "GOOD";
  if (aqi <= 100) return "MODERATE";
  if (aqi <= 200) return "POOR";
  if (aqi <= 300) return "VERY POOR";
  return "SEVERE";
}

void localFanControl(int aqi) {           // same rule as the server, without the forecast
  if (aqi >= 130) fanStage = 2;
  else if (aqi >= 101) fanStage = 1;
  else if (fanStage == 2) fanStage = 1;
  if (aqi < 90) {
    if (belowSinceMs == 0) belowSinceMs = millis();
    if (fanStage > 0 && millis() - belowSinceMs >= 600000UL) fanStage = 0;   // 10 min calm-down
  } else belowSinceMs = 0;
  fanPwm = PWM_STAGE[fanStage];
}

void setLeds(bool red) {
  digitalWrite(RED_LED_PIN, red ? HIGH : LOW);
  digitalWrite(GREEN_LED_PIN, red ? LOW : HIGH);
}

// =====================================================
// WIFI (does not block: the fan keeps working without Wi-Fi)
// =====================================================
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  static unsigned long lastTry = 0;
  if (lastTry != 0 && millis() - lastTry < 15000) return;
  lastTry = millis();
  Serial.print("WiFi: connecting");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 8000) { delay(250); Serial.print("."); }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) { Serial.print("WiFi connected, IP "); Serial.println(WiFi.localIP()); }
  else Serial.println("WiFi not connected - fan runs on local control");
}

// =====================================================
// SERVER (new AWS dashboard)
// =====================================================
int jsonInt(const String& body, const char* key, int def) {
  String k = String("\"") + key + "\":";
  int p = body.indexOf(k);
  if (p < 0) return def;
  return body.substring(p + k.length()).toInt();
}

void jnum(String& s, const char* k, float v, int d) {
  s += "\""; s += k; s += "\":";
  if (isnan(v)) s += "null"; else s += String(v, d);
  s += ",";
}

bool sendToServer(const String& json) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.begin(SERVER_URL);
  http.setConnectTimeout(4000);
  http.setTimeout(6000);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", API_KEY);
  int code = http.POST(json);
  bool ok = false;
  if (code == 200) {
    String body = http.getString();
    int st  = jsonInt(body, "fan_stage", -1);
    int pwm = jsonInt(body, "pwm", -1);
    if (st >= 0 && pwm >= 0) { fanStage = st; fanPwm = pwm; }
    serverRed = body.indexOf("\"led\":\"red\"") >= 0 || body.indexOf("\"led\": \"red\"") >= 0;
    Serial.printf("Server OK -> fan stage %d (pwm %d), LED %s\n", fanStage, fanPwm, serverRed ? "RED" : "GREEN");
    ok = true;
  } else {
    Serial.printf("Server error %d %s\n", code, code < 0 ? http.errorToString(code).c_str() : "");
  }
  http.end();
  return ok;
}

// =====================================================
// OLD API 42 (unchanged fields, keeps the old system working)
// =====================================================
void sendToLegacyAPI(int gpy10, int mq135, float t, float h, float p, int aqi) {
  if (!USE_LEGACY_API || WiFi.status() != WL_CONNECTED) return;
  if (isnan(t) || isnan(h)) { Serial.println("DHT22 error - old API not sent."); return; }
  String url = LEGACY_API_URL;
  url += "?gpy10=" + String(gpy10);
  url += "&mq135=" + String(mq135);
  url += "&temperature=" + String(t, 2);
  url += "&humidity=" + String(h, 2);
  url += "&bmp280=" + String(isnan(p) ? 0 : p, 2);
  url += "&aqi=" + String(aqi);
  String st = airStatus(aqi); st.replace(" ", "%20");
  url += "&air_status=" + st;
  url += "&fan="; url += (fanStage == 0 ? "OFF" : (fanStage == 1 ? "LOW" : "HIGH"));
  HTTPClient http;
  http.begin(url);
  http.setTimeout(10000);
  int code = http.GET();
  Serial.printf("Old API response: %d\n", code);
  http.end();
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("======================================");
  Serial.println("  AIR QUALITY - CO + climate + fan board");
  Serial.println("======================================");
  bootMs = millis();

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(GPY10_LED_PIN, OUTPUT); digitalWrite(GPY10_LED_PIN, HIGH);
  pinMode(MQ135_PIN, INPUT);
  pinMode(MQ7_PIN, INPUT);
  dht.begin();

  Wire.begin(SDA_PIN, SCL_PIN);
  if (bmp.begin(0x76))      { bmpOK = true; Serial.println("BMP280 detected at 0x76"); }
  else if (bmp.begin(0x77)) { bmpOK = true; Serial.println("BMP280 detected at 0x77"); }
  else Serial.println("BMP280 NOT detected!");

  pinMode(ENA_PIN, OUTPUT); pinMode(IN1_PIN, OUTPUT); pinMode(IN2_PIN, OUTPUT);
  digitalWrite(IN1_PIN, HIGH); digitalWrite(IN2_PIN, LOW);
  analogWrite(ENA_PIN, 0);

  pinMode(GREEN_LED_PIN, OUTPUT); pinMode(RED_LED_PIN, OUTPUT);
  setLeds(false);

  prefs.begin("aqms", false);
  mq7R0 = prefs.getFloat("r0", 0);
  Serial.printf("MQ-7 R0 from flash: %.3f kOhm %s\n", mq7R0, mq7R0 > 0 ? "" : "(will calibrate after warm-up)");

  WiFi.mode(WIFI_STA);
  ensureWiFi();
  Serial.println("System ready. MQ-7 warming up for 3 min...");
}

// =====================================================
// MAIN LOOP
// =====================================================
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n'); cmd.trim();
    if (cmd.equalsIgnoreCase("CAL")) calibrateMQ7();
  }
  if (lastSampleMs != 0 && millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();
  ensureWiFi();

  // ---------- read sensors ----------
  int gpy10 = readGPY10();
  int mq135 = analogRead(MQ135_PIN);
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  float p = bmpOK ? bmp.readPressure() / 100.0 : NAN;

  bool warm = millis() - bootMs >= MQ7_WARMUP_MS;
  if (warm && mq7R0 <= 0) calibrateMQ7();
  float rs = mq7Rs();
  float coPpm = NAN, coMg = NAN;
  if (warm && mq7R0 > 0) {
    coPpm = pushAvg(MQ7_A * pow(rs / mq7R0, MQ7_B));                  // 1-min mean, ppm
    float tK = (isnan(t) ? 25.0 : t) + 273.15;
    float pPa = (isnan(p) ? 1013.25 : p) * 100.0;
    coMg = coPpm * 28.01 * pPa / (8.314 * tK) * 1e-3;                 // ppm -> mg/m3
  }

  // ---------- local AQI: dust + CO (the server adds PM2.5) ----------
  int aqiDust = dustAQI(gpy10);
  int aqiLocal = aqiDust;
  if (!isnan(coMg)) aqiLocal = max(aqiLocal, (int)round(coSubIndex(coMg)));
  bool coBad = !isnan(coMg) && coMg > CO_BAD_MG;

  // ---------- send, then fan + LEDs ----------
  seqNo++;
  String j = "{";
  j += "\"device\":\""; j += DEVICE_ID; j += "\",";
  j += "\"seq\":"; j += seqNo; j += ",";
  j += "\"mirror\":"; j += (USE_LEGACY_API ? 1 : 0); j += ",";     // tells the server we also write the old API
  jnum(j, "co_ppm", coPpm, 2);
  jnum(j, "mq7_rs_kohm", rs, 3);
  jnum(j, "mq135", mq135, 0);
  jnum(j, "gpy10", gpy10, 0);
  jnum(j, "temperature", t, 2);
  jnum(j, "humidity", h, 2);
  jnum(j, "pressure", p, 2);
  jnum(j, "aqi", aqiDust, 0);                                         // dust AQI (stand-in if PM2.5 board is off)
  j += "\"fan\":"; j += fanStage; j += "}";

  if (sendToServer(j)) { lastServerOkMs = millis(); serverControl = true; }
  bool serverAlive = lastServerOkMs != 0 && millis() - lastServerOkMs <= SERVER_TIMEOUT_MS;
  if (!serverAlive) {
    if (serverControl) Serial.println("Server unreachable -> LOCAL control");
    serverControl = false;
    localFanControl(aqiLocal);
  }
  analogWrite(ENA_PIN, fanPwm);
  bool red = coBad || (serverAlive ? serverRed : aqiLocal > AQI_BAD);
  setLeds(red);

  // ---------- serial monitor ----------
  Serial.println();
  Serial.println("======================================");
  Serial.printf("#%lu  GPY10 %d (dust AQI %d) | MQ-135 %d\n", (unsigned long)seqNo, gpy10, aqiDust, mq135);
  if (isnan(coPpm)) Serial.printf("CO     : %s (Rs %.2f k)\n", warm ? "not calibrated" : "warming up", rs);
  else Serial.printf("CO     : %.2f ppm = %.2f mg/m3 %s\n", coPpm, coMg, coBad ? "<-- BAD" : "");
  if (isnan(t) || isnan(h)) Serial.println("DHT22  : ERROR");
  else Serial.printf("Temp   : %.2f C | Humidity %.2f %%\n", t, h);
  if (bmpOK) Serial.printf("Press. : %.2f hPa\n", p); else Serial.println("Press. : ERROR");
  Serial.printf("Local AQI %d (%s) | fan stage %d (%s) | LED %s\n", aqiLocal, airStatus(aqiLocal), fanStage,
                serverAlive ? "server" : "local", red ? "RED" : "GREEN");
  Serial.println("======================================");

  sendToLegacyAPI(gpy10, mq135, t, h, p, aqiLocal);
}
