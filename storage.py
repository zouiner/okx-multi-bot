# storage.py — SQLite storage with history, summary, export

import sqlite3, time, csv, datetime as dt
from typing import List, Tuple, Dict, Any

class Storage:
    def __init__(self, path: str = "bot_history.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            ts INTEGER,
            side TEXT,
            price REAL,
            qty REAL
        )""")
        self.conn.commit()

    def log_trade(self, symbol: str, side: str, price: float, qty: float) -> None:
        self.cur.execute(
            "INSERT INTO trades(symbol, ts, side, price, qty) VALUES(?,?,?,?,?)",
            (symbol, int(time.time()), side.lower(), price, qty)
        )
        self.conn.commit()

    def get_trades(self, symbol: str, n: int = 10) -> List[Tuple[int, str, float, float]]:
        self.cur.execute(
            "SELECT ts, side, price, qty FROM trades WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, n)
        )
        return self.cur.fetchall()

def summarize_symbol(db_path: str, symbol: str, days: int = 7) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cutoff = int(time.time()) - days * 86400
    cur.execute("""
        SELECT side, COUNT(*), SUM(price*qty)
        FROM trades WHERE symbol=? AND ts>=?
        GROUP BY side
    """, (symbol, cutoff))
    by_side = {}
    total_trades = 0
    for side, count, notional in cur.fetchall():
        by_side[side] = {"count": int(count), "notional": float(notional or 0.0)}
        total_trades += int(count)
    conn.close()
    return {"symbol": symbol, "days": days, "total_trades": total_trades, "by_side": by_side}

def export_trades_csv(db_path: str, out_path: str, symbol: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT ts, side, price, qty FROM trades WHERE symbol=? ORDER BY ts ASC", (symbol,))
    rows = cur.fetchall()
    conn.close()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "datetime_utc", "symbol", "side", "price", "qty"])
        for ts, side, price, qty in rows:
            iso = dt.datetime.utcfromtimestamp(ts).isoformat()
            w.writerow([ts, iso, symbol, side, price, qty])
    return out_path
