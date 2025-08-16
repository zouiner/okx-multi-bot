# strategy_grid.py — adaptive grid backtest with fees, ATR band, EMA trend gate
from typing import List, Dict, Any, Tuple
from math import inf

# -------- helpers --------
def ema(series: List[float], n: int) -> List[float]:
    if n <= 1 or not series:
        return list(series)
    k = 2.0 / (n + 1.0)
    m = None
    out = []
    for x in series:
        m = x if m is None else (x - m) * k + m
        out.append(m)
    return out

def atr_from_candles(candles: List[dict], n: int = 14) -> List[float]:
    """Wilders-like ATR (simple mean of TR over window)."""
    if not candles:
        return []
    tr = []
    for i, c in enumerate(candles):
        h, l = c["high"], c["low"]
        pc = candles[i - 1]["close"] if i > 0 else c["close"]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    out = []
    for i in range(len(tr)):
        if i + 1 < n:
            out.append(0.0)
        else:
            window = tr[i - n + 1 : i + 1]
            out.append(sum(window) / float(n))
    return out

def max_drawdown(values: List[float]) -> float:
    """Return max drawdown in % (positive number)."""
    peak = -inf
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak * 100.0
            if dd < max_dd:
                max_dd = dd
    return abs(max_dd)

def build_levels(center: float, span: float, grid_size: int, min_step_pct: float) -> List[float]:
    """Build symmetric grid around center with step >= min_step_pct * center."""
    if grid_size < 2 or center <= 0 or span <= 0:
        return []
    raw_step = (2.0 * span) / grid_size
    min_step = max(min_step_pct * center, 1e-12)
    step = max(raw_step, min_step)
    lo = center - (grid_size // 2) * step
    hi = center + (grid_size // 2) * step
    levels = []
    lv = lo + step
    while lv < hi - 1e-12:
        levels.append(lv)
        lv += step
    return levels

# -------- backtest --------
def run_grid_backtest(
    symbol: str,
    candles: List[dict],
    *,
    quote_cap: float = 1000.0,
    grid_size: int = 12,
    per_order_quote: float = 40.0,
    fee_rate: float = 0.001,      # 0.10% per side
    atr_mult: float = 2.0,        # ATR band multiplier
    ema_fast_n: int = 20,
    ema_slow_n: int = 60,
    recenter_threshold: float = 0.35,  # recenter when price drifts > 35% of half-span from center
    recenter_every: int = 240,         # also recenter every N candles (e.g., 240 = 4h on 1m bars)
) -> Dict[str, Any]:
    """
    Adaptive grid:
      - Band size = ATR * atr_mult, centered on EMA blend.
      - Enforce min grid step >= 3*fee_rate OR 0.2% (whichever larger).
      - EMA trend gate blocks buys in strong down-trend & sells in strong up-trend.
      - Recenter band when price drifts too far from center or every recenter_every bars.
      - Fees reduce equity on each trade (taker-style by default).
    Returns dict incl. equity_curve: List[Tuple[ts_ms, equity_val]].
    """
    base_result = {
        "trades": 0,
        "fees_paid": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "roi_pct": 0.0,
        "final_value": quote_cap,
        "max_drawdown_pct": 0.0,
        "avg_gross_per_trade": 0.0,
        "avg_fee_per_trade": 0.0,
        "equity_curve": [],  # (ts, equity)
        "params": {
            "grid_size": grid_size,
            "per_order_quote": per_order_quote,
            "fee_rate": fee_rate,
            "atr_mult": atr_mult,
            "ema_fast_n": ema_fast_n,
            "ema_slow_n": ema_slow_n,
            "recenter_threshold": recenter_threshold,
            "recenter_every": recenter_every,
            "min_step_pct": max(3.0 * fee_rate, 0.002),
            "start_equity": quote_cap,
        },
    }
    if not candles:
        return base_result

    prices = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]
    ts    = [c["ts"]   for c in candles]

    fast = ema(prices, ema_fast_n)
    slow = ema(prices, ema_slow_n)
    atr  = atr_from_candles(candles, n=14)

    min_step_pct = base_result["params"]["min_step_pct"]

    def band_center(i: int) -> float:
        return 0.6 * fast[i] + 0.4 * slow[i]

    i0 = max(ema_fast_n, ema_slow_n, 14) + 1
    i0 = min(i0, len(candles) - 1)

    center = band_center(i0)
    span   = max(atr[i0] * atr_mult, prices[i0] * 0.005)  # floor: 0.5% span if ATR tiny
    levels = build_levels(center, span, grid_size, min_step_pct)
    lower_levels = [lv for lv in levels if lv <= prices[i0]]
    upper_levels = [lv for lv in levels if lv >  prices[i0]]

    quote = float(quote_cap)
    base  = 0.0
    trades = 0
    fees_paid = 0.0
    equity_curve_vals: List[Tuple[int, float]] = []
    last_recenter = i0

    for i in range(i0, len(candles)):
        px = prices[i]

        # Trend gate
        is_up   = fast[i] > slow[i] * 1.001
        is_down = fast[i] < slow[i] * 0.999

        # Recenter?
        half_span = span
        drift = abs(px - center)
        recenter_due_time = (i - last_recenter) >= max(1, recenter_every)
        recenter_due_drift = (half_span > 0) and (drift / half_span >= recenter_threshold)
        if recenter_due_time or recenter_due_drift:
            center = band_center(i)
            span = max(atr[i] * atr_mult, px * 0.005)
            levels = build_levels(center, span, grid_size, min_step_pct)
            lower_levels = [lv for lv in levels if lv <= px]
            upper_levels = [lv for lv in levels if lv >  px]
            last_recenter = i

        # Buys (not in strong down-trend)
        while lower_levels and px <= lower_levels[-1] and not is_down:
            lv = lower_levels[-1]
            qty = per_order_quote / max(lv, 1e-12)
            fee = lv * qty * fee_rate
            total_cost = lv * qty + fee
            if quote + 1e-12 < total_cost:
                break
            quote -= total_cost
            base  += qty
            fees_paid += fee
            trades += 1
            lower_levels.pop()

        # Sells (not in strong up-trend)
        while upper_levels and px >= upper_levels[0] and not is_up:
            lv = upper_levels[0]
            qty_plan = per_order_quote / max(lv, 1e-12)
            qty = min(base, qty_plan)
            if qty <= 0:
                break
            fee = lv * qty * fee_rate
            proceeds = lv * qty - fee
            base  -= qty
            quote += proceeds
            fees_paid += fee
            trades += 1
            upper_levels.pop(0)

        equity_curve_vals.append((ts[i], quote + base * px))

    final_value = equity_curve_vals[-1][1]
    net_pnl = final_value - quote_cap
    gross_pnl = net_pnl + fees_paid
    roi_pct = (net_pnl / quote_cap * 100.0) if quote_cap > 0 else 0.0
    mdd = max_drawdown([v for _, v in equity_curve_vals])
    avg_fee = (fees_paid / trades) if trades else 0.0
    avg_gross = (gross_pnl / trades) if trades else 0.0

    base_result.update({
        "trades": int(trades),
        "fees_paid": float(fees_paid),
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "roi_pct": float(roi_pct),
        "final_value": float(final_value),
        "max_drawdown_pct": float(mdd),
        "avg_gross_per_trade": float(avg_gross),
        "avg_fee_per_trade": float(avg_fee),
        "equity_curve": equity_curve_vals,
    })
    return base_result
