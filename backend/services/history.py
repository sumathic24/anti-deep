"""
Analysis history — every completed text/image analysis is recorded in a local
SQLite database so the dashboard shows real activity instead of sample data.
"""
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.db")
LABEL_MAX_CHARS = 80


class AnalysisHistory:
    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  TEXT    NOT NULL,  -- local time, YYYY-MM-DD HH:MM:SS
                    kind        TEXT    NOT NULL,  -- 'text' | 'image'
                    label       TEXT    NOT NULL,  -- text snippet or file name
                    prediction  TEXT    NOT NULL,
                    is_ai       INTEGER NOT NULL,
                    confidence  REAL    NOT NULL   -- confidence in the predicted class, 0-1
                )""")
            self._conn.commit()

    def record(self, kind: str, label: str, result: Dict[str, Any]):
        ai = float(result["ai_confidence"])
        label = " ".join(label.split())
        if len(label) > LABEL_MAX_CHARS:
            label = label[:LABEL_MAX_CHARS - 1] + "…"
        with self._lock:
            self._conn.execute(
                "INSERT INTO analyses (created_at, kind, label, prediction, is_ai, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kind, label,
                 result["prediction"], int(ai > 0.5), max(ai, 1.0 - ai)))
            self._conn.commit()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            groups = self._conn.execute(
                "SELECT kind, is_ai, COUNT(*), AVG(confidence) FROM analyses GROUP BY kind, is_ai"
            ).fetchall()
            week_start = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
            daily = self._conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, kind, is_ai, COUNT(*) FROM analyses "
                "WHERE created_at >= ? GROUP BY day, kind, is_ai", (week_start,)
            ).fetchall()
            recent = self._conn.execute(
                "SELECT created_at, kind, label, prediction, is_ai, confidence FROM analyses "
                "ORDER BY id DESC LIMIT 5"
            ).fetchall()

        names = {("text", 1): "ai_text", ("text", 0): "human_text",
                 ("image", 1): "ai_image", ("image", 0): "real_image"}
        breakdown = {name: 0 for name in names.values()}
        avg_conf_by_type = {name: None for name in names.values()}
        total_conf = 0.0
        for kind, is_ai, count, avg_conf in groups:
            name = names[(kind, is_ai)]
            breakdown[name] = count
            avg_conf_by_type[name] = round(avg_conf * 100, 1)
            total_conf += avg_conf * count

        total = sum(breakdown.values())
        counts = {}
        for day, kind, is_ai, n in daily:
            for key in (kind, "ai" if is_ai else "human"):
                counts[(day, key)] = counts.get((day, key), 0) + n
        weekly = []
        for offset in range(6, -1, -1):
            d = datetime.now() - timedelta(days=offset)
            date = d.strftime("%Y-%m-%d")
            weekly.append({"day": d.strftime("%a"), "date": date,
                           **{key: counts.get((date, key), 0) for key in ("text", "image", "ai", "human")}})

        return {
            "total_analyses": total,
            "ai_content_detected": breakdown["ai_text"] + breakdown["ai_image"],
            "verified_real": breakdown["human_text"] + breakdown["real_image"],
            "avg_confidence": round(total_conf / total * 100, 1) if total else None,
            "text_analyses": breakdown["ai_text"] + breakdown["human_text"],
            "image_analyses": breakdown["ai_image"] + breakdown["real_image"],
            "weekly": weekly,
            "breakdown": breakdown,
            "avg_confidence_by_type": avg_conf_by_type,
            "recent": [{"created_at": c, "kind": k, "label": l, "prediction": p,
                        "is_ai": bool(a), "confidence": round(conf * 100, 1)}
                       for c, k, l, p, a, conf in recent],
        }


history = AnalysisHistory()
