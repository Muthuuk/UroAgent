"""Short-term memory: a SQLite checkpointer keyed by thread id (case study 6.4).

Each conversation thread keeps its message history and the last clinical findings, so
follow-up questions ("what should I do first?") are answered in context. Threads are
isolated from each other. Use ':memory:' for a throwaway store.
"""
import json
import sqlite3
import time


class SQLiteCheckpointer:
    def __init__(self, path="uroagent_memory.sqlite"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT PRIMARY KEY, state TEXT, updated REAL)")

    def load(self, thread_id: str) -> dict:
        row = self.conn.execute("SELECT state FROM checkpoints WHERE thread_id=?", (thread_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save(self, thread_id: str, state: dict):
        keep = {k: state.get(k) for k in ("messages", "entities", "prediction", "confidence", "clinical_context")}
        self.conn.execute("REPLACE INTO checkpoints VALUES (?,?,?)", (thread_id, json.dumps(keep), time.time()))
        self.conn.commit()

    def clear(self, thread_id: str):
        self.conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
        self.conn.commit()
