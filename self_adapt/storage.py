"""SQLite persistence for independent experiment records."""
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "results" / "experiments.sqlite"


def _connect(db_path=None):
    path = Path(db_path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        drift_type TEXT NOT NULL, seed INTEGER NOT NULL, payload TEXT NOT NULL)""")
    conn.commit()
    return conn


def save_experiment(result, db_path=None):
    conn = _connect(db_path)
    try:
        conn.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?)", (
            result["experiment_id"], result["created_at"],
            result["config"].get("drift_type","Unknown"),
            int(result["config"].get("seed",0)),
            json.dumps(result, allow_nan=False)))
        conn.commit()
    finally:
        conn.close()


def load_experiments(db_path=None):
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT payload FROM experiments ORDER BY created_at DESC").fetchall()
        return [json.loads(r[0]) for r in rows]
    finally:
        conn.close()


def get_experiment(experiment_id, db_path=None):
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT payload FROM experiments WHERE experiment_id=?",(experiment_id,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()
