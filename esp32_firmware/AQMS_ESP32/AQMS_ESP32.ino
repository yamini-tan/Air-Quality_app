/*
 * =====================================================================
 *  AQMS - ESP32 firmware (multi-pollutant version)
 *  Sensors : PMS5003 (PM2.5/PM10), MQ-7 (CO), MQ-135 (gas index),
 *            GP2Y1010AU0F (dust), DHT22 (T, RH), BMP280 (P)
 *  Actuator: L298N -> purifier fan (G4 + H13 filters)
 *
 *  Every 5 s the node reads all sensors and POSTs one JSON frame to the
 *  Flask server on AWS (/api/ingest). The server replies with the fan
 *  command computed from the current + 12-h forecast AQI (paper Table 5),
 *  which is applied to the fan PWM. If the server is unreachable for 30 s
 *  the node falls back to local staged control on its own AQI.
 *
 *  Libraries (Arduino IDE -> Library Manager):
 *    - "DHT sensor library" by Adafruit  (+ "Adafruit Unified Sensor")
 *    - "Adafruit BMP280 Library"         (+ "Adafruit BusIO")
 *  Board: "ESP32 Dev Module" (esp32 by Espressif, core 2.x or 3.x)
 *  PMS5003 is read with the built-in parser below (no library needed).
 * =====================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <Preferences.h>
#include <Adafruit_BMP280.h>
#include "DHT.h"

// =====================================================================
//  >>> EDIT THESE <<<
// =====================================================================
const char* WIFI_SSID     = "Aislyntech";
const char* WIFI_PASSWORD = "Aislyntech88*";

// Your EC2 public IP (or domain). Port 5000 = Flask/gunicorn, port 80 if you use nginx.
const char* SERVER_URL = "http://YOUR_EC2_PUBLIC_IP:5000/api/ingest";
const char* API_KEY    = "aqms-esp32-key";          // must match AQMS_API_KEY on the server
const char* DEVICE_ID  = "aqms-node-1";

// Optional: also push to the old aislyntech PHP API (keeps the old path alive)
const bool  USE_LEGACY_API = false;
const char* LEGACY_API_URL = "https://aislyntech.com/Api/42-update.php";

// =====================================================================
//  PIN MAP  (paper Table 7)
// =====================================================================
#define PMS_RX_PIN        16   // ESP32 RX2  <- PMS5003 TX
#define PMS_TX_PIN        17   // ESP32 TX2  -> PMS5003 RX
#define MQ7_PIN           35   // analog, via 10k/20k divider
#define MQ135_PIN         34   // analog, via 10k/20k divider
#define GPY10_ANALOG_PIN  36   // analog (VP), via divider
#define GPY10_LED_PIN     32   // LED drive (150R + 220uF), ACTIVE LOW
#define DHT_PIN           4
#define DHT_TYPE          DHT22
#define SDA_PIN           21
#define SCL_PIN           22
#define ENA_PIN           26   // L298N PWM
#define IN1_PIN           27
#define IN2_PIN           14
#define GREEN_LED_PIN     18   // optional status LEDs
#define RED_LED_PIN       19
/*  If your board is still wired like the OLD version, use instead:
      GPY10_LED_PIN 23, GPY10_ANALOG_PIN 35 (then move MQ-7 to 39/VN),
      ENA 25, IN1 26, IN2 27                                            */

// =====================================================================
//  MQ-7 CALIBRATION  (paper Sec 4.3, Eq. 9-10)
// =====================================================================
const float MQ_VC          = 5.0;      // module supply (V)
const float MQ_RL_KOHM     = 10.0;     // load resistor on the module (check: 10k or 1k)
const float DIVIDER_RATIO  = 1.5;      // 10k/20k divider: Vmodule = Vadc * (10+20)/20
const float MQ7_A          = 99.042;   // C_ppm = a * (Rs/R0)^b  (datasheet curve fit)
const float MQ7_B          = -1.518;
const float MQ7_CLEAN_AIR_RATIO = 27.5;   // Rs/R0 in clean air (datasheet)
const unsigned long MQ7_WARMUP_MS = 180000UL;  // 3 min warm-up before CO is reported
// R0 is auto-calibrated in clean air on first boot (after warm-up) and saved in flash.
// Type CAL in the Serial Monitor (115200) to re-calibrate in clean air.

// =====================================================================
//  TIMING / CONTROL
// =====================================================================
const unsigned long SAMPLE_INTERVAL_MS = 5000;   // paper: poll every 5 s
const unsigned long SERVER_TIMEOUT_MS  = 30000;  // fall back to local control after this
const int PWM_STAGE[3] = {0, 128, 255};          // 0 %, 50 %, 100 %

// =====================================================================
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;
Preferences prefs;
bool bmpOK = false;

float mq7R0 = 0;                 // kOhm
unsigned long bootMs = 0, lastSampleMs = 0, lastServerOkMs = 0;
uint32_t seqNo = 0;
int fanPwm = 0, fanStage = 0;
bool serverControl = false;
unsigned long belowSinceMs = 0;

// PMS5003 latest frame
uint16_t pm25Raw = 0, pm10Raw = 0;
bool pmsValid = false;
unsigned long pmsLastMs = 0;

// 1-min rolling averages (12 x 5 s)
const int AVG_N = 12;
float coBuf[AVG_N]; int coIdx = 0, coCount = 0;
float pmBuf[AVG_N]; int pmIdx = 0, pmCount = 0;

// =====================================================================
//  PMS5003 PARSER  (32-byte frame, 0x42 0x4D header)
// =====================================================================
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
      uint16_t chk = (buf[30] << 8) | buf[31];
      if (sum == chk) {
        // atmospheric-environment values (bytes 10-15)
        pm25Raw = (buf[12] << 8) | buf[13];
        pm10Raw = (buf[14] << 8) | buf[15];
        pmsValid = true;
        pmsLastMs = millis();
      }
    }
  }
}

// =====================================================================
//  ANALOG HELPERS
// =====================================================================
float readMilliVolts(int pin, int samples = 32) {
  uint32_t acc = 0;
  for (int i = 0; i < samples; i++) { acc += analogReadMilliVolts(pin); delayMicroseconds(200); }
  return acc / (float)samples;
}

int readGPY10() {                       // mean of 10 pulses, sampled 0.28 ms after LED on
  long acc = 0;
  for (int i = 0; i < 10; i++) {
    digitalWrite(GPY10_LED_PIN, LOW);   // LED ON (active low)
    delayMicroseconds(280);
    acc += analogRead(GPY10_ANALOG_PIN);
    delayMicroseconds(40);
    digitalWrite(GPY10_LED_PIN, HIGH);  // LED OFF
    delayMicroseconds(9680);
  }
  return acc / 10;
}

float mq7Rs() {                         // Eq. (9): Rs = RL (Vc - Vout) / Vout
  float vout = readMilliVolts(MQ7_PIN) / 1000.0 * DIVIDER_RATIO;
  if (vout < 0.01) vout = 0.01;
  if (vout > MQ_VC - 0.01) vout = MQ_VC - 0.01;
  return MQ_RL_KOHM * (MQ_VC - vout) / vout;
}

void calibrateMQ7() {
  Serial.println("Calibrating MQ-7 R0 in clean air (20 s)...");
  float acc = 0;
  for (int i = 0; i < 40; i++) { acc += mq7Rs(); delay(500); }
  mq7R0 = (acc / 40.0) / MQ7_CLEAN_AIR_RATIO;
  prefs.putFloat("r0", mq7R0);
  Serial.printf("MQ-7 R0 = %.3f kOhm (saved)\n", mq7R0);
}

float pushAvg(float* b, int& i, int& n, float v) {
  b[i] = v; i = (i + 1) % AVG_N; if (n < AVG_N) n++;
  float s = 0; for (int k = 0; k < n; k++) s += b[k];
  return s / n;
}

// =====================================================================
//  LOCAL AQI (CPCB, used only for serial print + fallback control)
// =====================================================================
float subIndex(const float bp[][4], int n, float c) {
  if (c < 0) c = 0;
  for (int i = 0; i < n; i++) {
    if (c <= bp[i][1]) {
      float v = (bp[i][3] - bp[i][2]) / (bp[i][1] - bp[i][0]) * (c - bp[i][0]) + bp[i][2];
      return constrain(v, bp[i][2], bp[i][3]);
    }
  }
  return 500;
}
const float BP_PM[6][4] = {{0,30,0,50},{31,60,51,100},{61,90,101,200},{91,120,201,300},{121,250,301,400},{251,380,401,500}};
const float BP_CO[6][4] = {{0,1.0,0,50},{1.1,2.0,51,100},{2.1,10,101,200},{10.1,17,201,300},{17.1,34,301,400},{34.1,50,401,500}};

const char* category(int a) {
  if (a <= 50) return "GOOD";
  if (a <= 100) return "SATISFACTORY";
  if (a <= 200) return "MODERATELY POLLUTED";
  if (a <= 300) return "POOR";
  if (a <= 400) return "VERY POOR";
  return "SEVERE";
}

void localControl(int aqi) {             // Table 5 without forecast
  if (aqi >= 130) fanStage = 2;
  else if (aqi >= 101) fanStage = 1;
  else if (fanStage == 2) fanStage = 1;
  if (aqi < 90) {
    if (belowSinceMs == 0) belowSinceMs = millis();
    if (fanStage > 0 && millis() - belowSinceMs >= 600000UL) fanStage = 0;
  } else belowSinceMs = 0;
  fanPwm = PWM_STAGE[fanStage];
}

void applyFan() {
  analogWrite(ENA_PIN, fanPwm);
  digitalWrite(GREEN_LED_PIN, fanStage == 0 ? HIGH : LOW);
  digitalWrite(RED_LED_PIN,   fanStage > 0 ? HIGH : LOW);
}

// =====================================================================
//  JSON helpers (no library)
// =====================================================================
void jnum(String& s, const char* k, float v, int d = 2, bool comma = true) {
  s += "\""; s += k; s += "\":";
  if (isnan(v)) s += "null"; else s += String(v, d);
  if (comma) s += ",";
}
int jsonInt(const String& body, const char* key, int def) {
  String k = String("\"") + key + "\":";
  int p = body.indexOf(k);
  if (p < 0) return def;
  return body.substring(p + k.length()).toInt();
}

// =====================================================================
//  WIFI
// =====================================================================
void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  static unsigned long lastTry = 0;
  if (millis() - lastTry < 10000 && lastTry != 0) return;
  lastTry = millis();
  Serial.println("WiFi: (re)connecting...");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 8000) { delay(250); Serial.print("."); }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) { Serial.print("WiFi OK, IP "); Serial.println(WiFi.localIP()); }
  else Serial.println("WiFi not connected - running local control");
}

// =====================================================================
//  SEND TO SERVER
// =====================================================================
bool sendToServer(const String& json) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  WiFiClient plain;
  WiFiClientSecure secure;
  bool ok = false;
  String url = SERVER_URL;
  if (url.startsWith("https")) { secure.setInsecure(); http.begin(secure, url); }
  else http.begin(plain, url);
  http.setConnectTimeout(4000);
  http.setTimeout(6000);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", API_KEY);
  int code = http.POST(json);
  if (code == 200) {
    String body = http.getString();
    int pwm = jsonInt(body, "pwm", -1);
    int st  = jsonInt(body, "fan_stage", -1);
    if (pwm >= 0 && st >= 0) { fanPwm = pwm; fanStage = st; }
    Serial.printf("Server 200 -> fan stage %d, pwm %d\n", fanStage, fanPwm);
    ok = true;
  } else {
    Serial.printf("Server error %d: %s\n", code, code < 0 ? http.errorToString(code).c_str() : http.getString().c_str());
  }
  http.end();
  return ok;
}

void sendLegacy(float pm25, float pm10, float co, int gpy, int mq, float t, float h, float p, int aqi) {
  if (!USE_LEGACY_API || WiFi.status() != WL_CONNECTED) return;
  WiFiClientSecure c; c.setInsecure();
  HTTPClient http;
  String u = String(LEGACY_API_URL) + "?gpy10=" + gpy + "&mq135=" + mq +
             "&temperature=" + String(t, 2) + "&humidity=" + String(h, 2) + "&bmp280=" + String(p, 2) +
             "&pm25=" + String(pm25, 1) + "&pm10=" + String(pm10, 1) + "&co=" + String(co, 2) +
             "&aqi=" + aqi + "&air_status=" + String(category(aqi)).substring(0, 4) +
             "&fan=" + (fanStage == 0 ? "OFF" : (fanStage == 1 ? "LOW" : "HIGH"));
  u.replace(" ", "%20");
  http.begin(c, u);
  http.setTimeout(6000);
  int code = http.GET();
  Serial.printf("Legacy API: %d\n", code);
  http.end();
}

// =====================================================================
//  SETUP
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n======================================");
  Serial.println("  AQMS multi-pollutant node (PM2.5 + CO)");
  Serial.println("======================================");
  bootMs = millis();

  Serial2.begin(9600, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(GPY10_LED_PIN, OUTPUT); digitalWrite(GPY10_LED_PIN, HIGH);
  pinMode(MQ135_PIN, INPUT); pinMode(MQ7_PIN, INPUT); pinMode(GPY10_ANALOG_PIN, INPUT);
  dht.begin();
  Wire.begin(SDA_PIN, SCL_PIN);
  if (bmp.begin(0x76)) { bmpOK = true; Serial.println("BMP280 at 0x76"); }
  else if (bmp.begin(0x77)) { bmpOK = true; Serial.println("BMP280 at 0x77"); }
  else Serial.println("BMP280 NOT detected!");

  pinMode(ENA_PIN, OUTPUT); pinMode(IN1_PIN, OUTPUT); pinMode(IN2_PIN, OUTPUT);
  digitalWrite(IN1_PIN, HIGH); digitalWrite(IN2_PIN, LOW);
  analogWrite(ENA_PIN, 0);
  pinMode(GREEN_LED_PIN, OUTPUT); pinMode(RED_LED_PIN, OUTPUT);

  prefs.begin("aqms", false);
  mq7R0 = prefs.getFloat("r0", 0);
  Serial.printf("MQ-7 R0 from flash: %.3f kOhm %s\n", mq7R0, mq7R0 > 0 ? "" : "(will auto-calibrate after warm-up)");

  WiFi.mode(WIFI_STA);
  ensureWiFi();
  Serial.println("System ready. MQ-7 warming up for 3 min...");
}

// =====================================================================
//  LOOP
// =====================================================================
void loop() {
  readPMS();                                  // keep draining the UART

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n'); cmd.trim();
    if (cmd.equalsIgnoreCase("CAL")) calibrateMQ7();
  }

  if (millis() - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs = millis();
  ensureWiFi();

  // ---------- read sensors ----------
  bool warm = millis() - bootMs >= MQ7_WARMUP_MS;
  if (warm && mq7R0 <= 0) calibrateMQ7();

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  float p = bmpOK ? bmp.readPressure() / 100.0 : NAN;
  int mq135 = analogRead(MQ135_PIN);
  int gpy = readGPY10();

  float pm25 = NAN, pm10 = NAN, pmAvg = NAN;
  if (pmsValid && millis() - pmsLastMs < 10000) {
    pm25 = pm25Raw; pm10 = pm10Raw;
    pmAvg = pushAvg(pmBuf, pmIdx, pmCount, pm25);
  }

  float rs = mq7Rs();
  float coPpm = NAN, coAvg = NAN;
  if (warm && mq7R0 > 0) {
    coPpm = MQ7_A * pow(rs / mq7R0, MQ7_B);  // Eq. (10)
    coAvg = pushAvg(coBuf, coIdx, coCount, coPpm);   // 1-min mean
  }

  // local CPCB-method AQI (server computes the authoritative one)
  float tK = (isnan(t) ? 25.0 : t) + 273.15;
  float pPa = (isnan(p) ? 1013.25 : p) * 100.0;
  float coMg = isnan(coAvg) ? NAN : coAvg * 28.01 * pPa / (8.314 * tK) * 1e-3;       // Eq. (11)
  float pmCorr = isnan(pmAvg) ? NAN : max(0.0f, 0.524f * pmAvg - 0.0862f * (isnan(h) ? 50 : h) + 5.75f);  // Eq. (12)
  float sPm = isnan(pmCorr) ? 0 : subIndex(BP_PM, 6, pmCorr);
  float sCo = isnan(coMg) ? 0 : subIndex(BP_CO, 6, coMg);
  int aqi = (int)round(max(sPm, sCo));

  // ---------- JSON frame ----------
  seqNo++;
  String j = "{";
  j += "\"device\":\""; j += DEVICE_ID; j += "\",";
  j += "\"seq\":"; j += seqNo; j += ",";
  jnum(j, "pm25", pm25, 1);
  jnum(j, "pm10", pm10, 1);
  jnum(j, "co_ppm", coAvg, 2);
  jnum(j, "mq7_rs_kohm", rs, 3);
  jnum(j, "mq135", mq135, 0);
  jnum(j, "gpy10", gpy, 0);
  jnum(j, "temperature", t, 2);
  jnum(j, "humidity", h, 2);
  jnum(j, "pressure", p, 2);
  jnum(j, "local_aqi", aqi, 0);
  j += "\"fan\":"; j += fanStage; j += "}";

  // ---------- serial monitor ----------
  Serial.println("--------------------------------------");
  Serial.printf("#%lu  PM2.5 %s ug/m3 (corr %.1f) | PM10 %s\n", (unsigned long)seqNo,
                isnan(pm25) ? "--" : String(pm25, 0).c_str(), pmCorr, isnan(pm10) ? "--" : String(pm10, 0).c_str());
  Serial.printf("CO %s ppm = %s mg/m3 (Rs %.2fk, R0 %.2fk)%s\n",
                isnan(coAvg) ? "--" : String(coAvg, 1).c_str(), isnan(coMg) ? "--" : String(coMg, 2).c_str(),
                rs, mq7R0, warm ? "" : "  [warming up]");
  Serial.printf("MQ-135 %d | Dust %d | T %.1f C | RH %.1f %% | P %.1f hPa\n", mq135, gpy, t, h, p);
  Serial.printf("Local AQI %d (%s)  PM sub %.0f / CO sub %.0f\n", aqi, category(aqi), sPm, sCo);
  Serial.println(j);

  // ---------- send + control ----------
  if (sendToServer(j)) { lastServerOkMs = millis(); serverControl = true; }
  if (millis() - lastServerOkMs > SERVER_TIMEOUT_MS) {
    if (serverControl) Serial.println("Server unreachable -> LOCAL control");
    serverControl = false;
    localControl(aqi);
  }
  applyFan();
  sendLegacy(pm25, pm10, coAvg, gpy, mq135, t, h, p, aqi);
}
