"""
database_manager.py
====================
Sets up and manages the local SQLite telemetry store (telemetry.db).

Provides a small, thread-safe-enough data-access layer used by:
  - mqtt_subscriber.py  (writes incoming live sensor readings)
  - app.py              (reads the latest rows for inference/plotting)

Run this file directly to (re)initialize the database schema:
    python database_manager.py
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "telemetry.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL,
    machine_id              TEXT NOT NULL,
    cycle                   INTEGER NOT NULL,
    temperature             REAL NOT NULL,
    vibration                REAL NOT NULL,
    pressure                REAL NOT NULL,
    rotational_speed        REAL NOT NULL,
    operational_setting_1   REAL NOT NULL,
    operational_setting_2   REAL NOT NULL,
    operational_setting_3   REAL NOT NULL,
    predicted_rul           REAL,
    inserted_at              TEXT NOT NULL
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_telemetry_machine_time
ON telemetry (machine_id, timestamp);
"""


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Context-managed SQLite connection with sane defaults for concurrent access."""
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # allows concurrent reader (app) + writer (subscriber)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Creates the telemetry table and index if they do not already exist."""
    with get_connection(db_path) as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(INDEX_SQL)
    print(f"[database_manager] Database initialized at '{db_path}'.")


def insert_telemetry(record: dict, db_path: str = DB_PATH) -> None:
    """
    Inserts a single telemetry reading.

    Expected keys in `record`:
        machine_id, cycle, temperature, vibration, pressure,
        rotational_speed, operational_setting_1, operational_setting_2,
        operational_setting_3, timestamp (optional), predicted_rul (optional)
    """
    timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
    inserted_at = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO telemetry (
                timestamp, machine_id, cycle, temperature, vibration, pressure,
                rotational_speed, operational_setting_1, operational_setting_2,
                operational_setting_3, predicted_rul, inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                record["machine_id"],
                record["cycle"],
                record["temperature"],
                record["vibration"],
                record["pressure"],
                record["rotational_speed"],
                record["operational_setting_1"],
                record["operational_setting_2"],
                record["operational_setting_3"],
                record.get("predicted_rul"),
                inserted_at,
            ),
        )


def fetch_latest(n: int = 50, machine_id: Optional[str] = None, db_path: str = DB_PATH):
    """
    Returns the latest `n` telemetry rows as a list of dicts, ordered
    chronologically ascending (oldest -> newest) for correct plotting.
    """
    with get_connection(db_path) as conn:
        if machine_id:
            cursor = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM telemetry
                    WHERE machine_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (machine_id, n),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM telemetry
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (n,),
            )
        rows = [dict(row) for row in cursor.fetchall()]
    return rows


def update_predicted_rul(record_id: int, predicted_rul: float, db_path: str = DB_PATH) -> None:
    """Back-fills the predicted RUL for a given row id (optional audit trail)."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE telemetry SET predicted_rul = ? WHERE id = ?",
            (predicted_rul, record_id),
        )


def row_count(db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM telemetry")
        return cursor.fetchone()["cnt"]


if __name__ == "__main__":
    init_db()
    print(f"[database_manager] Current row count: {row_count()}")
