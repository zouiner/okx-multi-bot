import sqlite3, time

class Storage:
    def __init__(self, path="bot_history.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("""CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, ts INTEGER, side TEXT, price REAL, qty REAL
        )""")
        self.conn.commit()

    def log_trade(self, symbol, side, price, qty):
        self.cur.execute("INSERT INTO trades(symbol, ts, side, price, qty) VALUES(?,?,?,?,?)",
                         (symbol, int(time.time()), side, price, qty))
        self.conn.commit()

    def get_trades(self, symbol, n=10):
        self.cur.execute("SELECT ts, side, price, qty FROM trades WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                         (symbol, n))
        return self.cur.fetchall()
