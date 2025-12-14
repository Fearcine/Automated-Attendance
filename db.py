import sqlite3
import pandas as pd
from datetime import datetime
import os

DB = "attendance.db"
EXCEL_FILE = "attendance.xlsx"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            embedding TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_person(name, embedding):
    emb_str = ",".join(map(str, embedding))
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO people (name, embedding) VALUES (?, ?)",
        (name, emb_str)
    )
    conn.commit()
    conn.close()

def load_people():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM people")
    rows = []
    for r in c.fetchall():
        emb = list(map(float, r[2].split(",")))
        rows.append((r[0], r[1], emb))
    conn.close()
    return rows

def mark_attendance(person_id, name):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO attendance (person_id, name) VALUES (?, ?)",
        (person_id, name)
    )
    conn.commit()
    conn.close()

def mark_excel_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        if ((df["Name"] == name) & (df["Date"] == today)).any():
            return
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    new_row = {
        "Name": name,
        "Date": today,
        "Time": time_now
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")


