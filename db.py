import sqlite3

DB = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS people (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT UNIQUE,
                   embedding TEXT
                 )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   person_id INTEGER,
                   name TEXT,
                   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                 )""")
    conn.commit()
    conn.close()

def save_person(name: str, embedding):
    # Ensure embedding is a list
    if not isinstance(embedding, (list, tuple)):
        embedding = [embedding]
    emb_str = ",".join(map(str, embedding))

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO people (name, embedding) VALUES (?, ?)",
        (name, emb_str),
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

def mark_attendance(person_id: int, name: str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO attendance (person_id, name) VALUES (?, ?)",
        (person_id, name),
    )
    conn.commit()
    conn.close()
