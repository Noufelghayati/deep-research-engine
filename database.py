import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "dossier.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_id TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            picture TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            last_login TEXT DEFAULT (datetime('now')),
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            subscription_status TEXT DEFAULT 'free',
            dossier_count INTEGER DEFAULT 0,
            billing_cycle_start TEXT
        )
    """)
    # Migrate existing tables: add columns if they don't exist
    cursor = conn.execute("PRAGMA table_info(users)")
    existing = {row[1] for row in cursor.fetchall()}
    migrations = {
        "stripe_customer_id": "TEXT",
        "stripe_subscription_id": "TEXT",
        "subscription_status": "TEXT DEFAULT 'free'",
        "dossier_count": "INTEGER DEFAULT 0",
        "billing_cycle_start": "TEXT",
    }
    for col, col_type in migrations.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
