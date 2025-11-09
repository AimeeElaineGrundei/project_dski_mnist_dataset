import sqlite3
from datetime import datetime

DB_PATH = "cnn_mnist.db"

def connect():
    """Creates connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_table():
    """Create table, if it doesn't exist."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cnn_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            input_data TEXT,
            predicted_label TEXT,
            true_label TEXT,
            correct INTEGER,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def insert_result(model_name, input_data, predicted_label, true_label, correct, confidence=None):
    """Inserts a new dataset result into the database."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cnn_results (timestamp, model_name, input_data, predicted_label, true_label, correct, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model_name,
        str(input_data),
        str(predicted_label),
        str(true_label),
        int(correct),
        confidence
    ))
    conn.commit()
    conn.close()

def fetch_all():
    """Fetches all results from the database."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cnn_results ORDER BY id DESC")
    results = cursor.fetchall()
    conn.close()
    return results

def fetch_by_model(model_name):
    """Fetches results for a specific model."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cnn_results WHERE model_name = ?", (model_name,))
    results = cursor.fetchall()
    conn.close()
    return results
