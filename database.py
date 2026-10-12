"""Small SQLite helpers. All statements that accept values use parameters."""
from pathlib import Path
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "predictions.db"
MESSAGES = {
    "LOW": "Current conditions indicate relatively low flood risk.",
    "MEDIUM": "Moderate flood risk detected. Monitor changing weather conditions.",
    "HIGH": "WARNING: High flood risk detected. Follow official local weather and emergency advisories.",
}


def init_db(path=DB_PATH):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            rainfall REAL NOT NULL, water_level REAL NOT NULL,
            temperature REAL NOT NULL, humidity REAL NOT NULL,
            risk TEXT NOT NULL, alert TEXT NOT NULL, source TEXT NOT NULL)""")


def save_prediction(values, risk, source="Manual measurements", path=DB_PATH):
    if risk not in MESSAGES:
        raise ValueError("Unknown risk category")
    init_db(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO predictions (timestamp,rainfall,water_level,temperature,humidity,risk,alert,source) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), *values, risk, MESSAGES[risk], source))
        return cursor.lastrowid


def get_history(path=DB_PATH):
    init_db(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", connection)


def clear_history(path=DB_PATH):
    init_db(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM predictions")
