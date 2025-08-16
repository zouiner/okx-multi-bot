# data.py — minimal OKX data fetchers (no auth required)
import requests

OKX_BASE = "https://www.okx.com"

def _get(url, params=None, timeout=12):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and str(j.get("code", "0")) != "0":
        raise RuntimeError(f"OKX API error {j.get('code')}: {j.get('msg')}")
    return j.get("data", j)

def fetch_last_price(symbol: str) -> float:
    data = _get(f"{OKX_BASE}/api/v5/market/ticker", {"instId": symbol})
    return float(data[0]["last"]) if data else 0.0

def fetch_ohlcv(symbol: str, bar: str = "1m", days: int = 1):
    """
    Returns oldest->newest list of candles.
    OKX candle row can have 7–9+ fields. We only use:
      0: ts, 1: open, 2: high, 3: low, 4: close, 5: vol (base)
    """
    per_day = 1440 if bar == "1m" else 24
    limit = min(max(1, days * per_day), 1000)

    data = _get(f"{OKX_BASE}/api/v5/market/candles",
                {"instId": symbol, "bar": bar, "limit": str(limit)})

    candles = []
    for row in data:
        ts   = int(row[0])
        o    = float(row[1])
        h    = float(row[2])
        l    = float(row[3])
        c    = float(row[4])
        vol  = float(row[5]) if len(row) > 5 else 0.0
        candles.append({
            "ts": ts,
            "open": o,
            "high": h,
            "low":  l,
            "close": c,
            "volume": vol,
        })

    # OKX returns newest-first → reverse to oldest-first
    return list(reversed(candles))
