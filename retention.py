"""
retention.py - data retention for the AQMS database
===================================================
Raw sensor readings are kept for RETENTION_DAYS (default 10: enough for the
7-day retraining window). Before a reading is deleted its AQI values are
copied into `aqi_history`, which is kept FOREVER.

Runs automatically inside app.py every CLEANUP_EVERY_H hours, or manually:
    python retention.py                 # uses AQMS_DB / RETENTION_DAYS env vars
    python retention.py --days 10 --dry-run
"""

import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
TS_FMT = "%Y-%m-%d %H:%M:%S"

AQI_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS aqi_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER,
    source_table TEXT NOT NULL DEFAULT 'readings',
    created_at TEXT NOT NULL,
    device_id TEXT,
    aqi_rt INTEGER, aqi_hourly INTEGER, aqi_cpcb INTEGER,
    sub_pm25 REAL, sub_co REAL,
    dominant TEXT, category TEXT,
    UNIQUE (source_table, reading_id)
)"""


def ensure_schema(conn):
    conn.execute(AQI_HISTORY_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_aqi_history_time ON aqi_history(created_at)")


def _has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def cleanup(db_path, days=10, dry_run=False):
    """Archive AQI of readings older than `days` into aqi_history, then delete those readings."""
    cutoff = (datetime.now(IST) - timedelta(days=days)).strftime(TS_FMT)
    conn = sqlite3.connect(db_path, timeout=30)
    stats = {"cutoff": cutoff, "archived": 0, "deleted": 0, "v1_archived": 0, "v1_deleted": 0}
    try:
        ensure_schema(conn)
        if _has_table(conn, "readings"):
            if dry_run:
                stats["deleted"] = conn.execute("SELECT COUNT(*) FROM readings WHERE created_at < ?",
                                                (cutoff,)).fetchone()[0]
            else:
                with conn:  # one transaction: archive + delete, never lose AQI
                    stats["archived"] = conn.execute(
                        "INSERT OR IGNORE INTO aqi_history (reading_id, source_table, created_at, device_id, "
                        "aqi_rt, aqi_hourly, aqi_cpcb, sub_pm25, sub_co, dominant, category) "
                        "SELECT id, 'readings', created_at, device_id, aqi_rt, aqi_hourly, aqi_cpcb, "
                        "sub_pm25, sub_co, dominant, category FROM readings "
                        "WHERE created_at < ? AND COALESCE(aqi_rt, aqi_hourly, aqi_cpcb) IS NOT NULL",
                        (cutoff,)).rowcount
                    stats["deleted"] = conn.execute("DELETE FROM readings WHERE created_at < ?",
                                                    (cutoff,)).rowcount
        # old v1 table (if the old database was migrated): keep its `aqi` column forever too
        if _has_table(conn, "sensor_data"):
            if dry_run:
                stats["v1_deleted"] = conn.execute("SELECT COUNT(*) FROM sensor_data WHERE created_at < ?",
                                                   (cutoff,)).fetchone()[0]
            else:
                with conn:
                    stats["v1_archived"] = conn.execute(
                        "INSERT OR IGNORE INTO aqi_history (reading_id, source_table, created_at, aqi_rt, category) "
                        "SELECT id, 'sensor_data', created_at, aqi, air_status FROM sensor_data "
                        "WHERE created_at < ? AND aqi IS NOT NULL", (cutoff,)).rowcount
                    stats["v1_deleted"] = conn.execute("DELETE FROM sensor_data WHERE created_at < ?",
                                                       (cutoff,)).rowcount
    finally:
        conn.close()
    return stats


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("AQMS_DB", os.path.join(here, "sensor_data.db")))
    ap.add_argument("--days", type=float, default=float(os.environ.get("RETENTION_DAYS", 10)))
    ap.add_argument("--dry-run", action="store_true", help="only count what would be deleted")
    a = ap.parse_args()
    print(cleanup(a.db, a.days, a.dry_run))
