import json
import time
import sqlite3

def open_save_db(name):
    db_file = f"{name}.save.sqlite"
    print(f"Opening DB file {db_file}")
    CONNECTION = sqlite3.connect(db_file)
    cursor = CONNECTION.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS history (save, role, message, time)")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reducers 
        (
            save,
            key,
            value,
            budget,
            UNIQUE(save, key)
        )
    """)
    return CONNECTION

def push_history(connection, name, role, message):
    now = time.time()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO history VALUES (?, ?, ?, ?)", (name, role, message, now))
    connection.commit()

def push_reducer(connection, name, key, value, budget):
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO reducers VALUES (?, ?, ?, ?) 
        ON CONFLICT (save, key) DO UPDATE SET 
            value=excluded.value,
            budget=excluded.budget
    """, (name, key, value, budget))
    connection.commit()

def get_history(connection, name, limit):
    cursor = connection.cursor()
    cursor.execute("SELECT time, role, message FROM history ORDER BY time DESC limit ?", (limit,))
    results = cursor.fetchall()
    return [{
        "role": role,
        "content": message
    } for (_, role, message) in results]

def get_most_recent_save(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT save FROM history ORDER BY time DESC limit 1")
    return cursor.fetchone()[0]

def get_reducers(connection, name):
    cursor = connection.cursor()
    cursor.execute("SELECT key, value, budget FROM reducers")
    results = cursor.fetchall()
    return [
        {
            "key": key,
            "value": value,
            "budget": budget,
        } for (key, value, budget) in results
    ]
