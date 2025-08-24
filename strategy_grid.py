# strategy_grid.py — adaptive grid backtest with fees, ATR band, EMA trend gate
from typing import List, Dict, Any, Tuple, Optional
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
    # ----- NEW: AI + DCA overlay -----
    ai_bias: Optional[str] = None,     # "long" | "short" | "neutral" | None
    ai_confidence: float = 0.0,        # 0..1
    ai_crisis: bool = False,
    dca_tp_pct: float = 0.10,          # +10% over blended WAP to take profit
    dca_add_drawdown_pct: float = 0.15,# add if price <= last_buy * (1 - 15%)
    dca_max_adds: int = 2,             # max rescue adds
    dca_equity_fraction: float = 0.10, # spend at most 10% of current equity per add
    atr_extreme_mult: float = 4.0,     # if ATR > extreme_mult * median_ATR, block DCA adds
) -> Dict[str, Any]:
    """
    Adaptive grid + AI/DCA overlay:
      - Grid: ATR band centered on EMA blend; min step >= 3*fee OR 0.2%.
      - EMA trend gate blocks buys in strong down-trend & sells in strong up-trend.
      - Recenter when drift too far or every recenter_every bars.
      - Fees reduce equity on each trade.
      - AI overlay: crisis halts entries; bias nudges side-permissions.
      - DCA overlay: If underwater by dca_add_drawdown_pct, add ≤10% equity and require TP at WAP*(1+dca_tp_pct).
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
            # expose new params for audit
            "ai_bias": ai_bias or "none",
            "ai_confidence": ai_confidence,
            "ai_crisis": ai_crisis,
            "dca_tp_pct": dca_tp_pct,
            "dca_add_drawdown_pct": dca_add_drawdown_pct,
            "dca_max_adds": dca_max_adds,
            "dca_equity_fraction": dca_equity_fraction,
            "atr_extreme_mult": atr_extreme_mult,
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

    # --- DCA bookkeeping ---
    dca_adds = 0
    last_buy_px = None
    # WAP tracker: weighted average price of long inventory
    def current_wap() -> float:
        # If no long inventory, WAP is undefined; return 0
        # We maintain WAP implicitly via base & a running cost sum
        return (cost_sum / base) if base > 1e-12 else 0.0

    cost_sum = 0.0  # total cost of long inventory (excl. fees for TP calc)
    median_atr = sorted([x for x in atr if x > 0.0])[len([x for x in atr if x > 0.0]) // 2] if any(atr) else 0.0

    for i in range(i0, len(candles)):
        px = prices[i]

        # Trend gate (base)
        is_up   = fast[i] > slow[i] * 1.001
        is_down = fast[i] < slow[i] * 0.999

        # AI crisis: freeze new entries (only allow profit-taking sells)
        crisis_freeze = ai_crisis and ai_confidence >= 0.5

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

        # ----- BUY logic -----
        allow_buy = not is_down  # base grid rule
        # AI bias: prefer buys in long bias; restrict in short bias
        if ai_bias == "long" and ai_confidence >= 0.5:
            allow_buy = True
        if ai_bias == "short" and ai_confidence >= 0.5:
            allow_buy = False
        # Crisis freezes new buys
        if crisis_freeze:
            allow_buy = False

        while lower_levels and px <= lower_levels[-1] and allow_buy:
            lv = lower_levels[-1]

            # normal grid size
            qty_plan = per_order_quote / max(lv, 1e-12)

            # DCA add? if underwater vs last buy and constraints pass
            do_dca = False
            if last_buy_px is not None and px <= last_buy_px * (1.0 - dca_add_drawdown_pct):
                extreme_atr = (atr[i] > atr_extreme_mult * median_atr) if median_atr > 0 else False
                if (dca_adds < dca_max_adds) and (not extreme_atr) and (not crisis_freeze):
                    # budget ≤ 10% of current equity
                    current_equity = quote + base * px
                    dca_budget = max(0.0, dca_equity_fraction * current_equity)
                    if dca_budget > 1e-9:
                        qty_plan = dca_budget / max(lv, 1e-12)
                        do_dca = True

            qty = qty_plan
            # Ensure we have quote including fee
            fee = lv * qty * fee_rate
            total_cost = lv * qty + fee
            if quote + 1e-12 < total_cost:
                break

            # execute buy
            quote -= total_cost
            base  += qty
            cost_sum += lv * qty  # for WAP
            fees_paid += fee
            trades += 1
            last_buy_px = lv
            if do_dca:
                dca_adds += 1

            lower_levels.pop()

        # ----- SELL logic -----
        allow_sell = not is_up  # base grid rule
        # AI bias: prefer sells in short bias; restrict in long bias
        if ai_bias == "short" and ai_confidence >= 0.5:
            allow_sell = True
        if ai_bias == "long" and ai_confidence >= 0.5:
            allow_sell = False
        # In crisis we can allow sells to reduce risk (keep allow_sell as computed)

        # If we have inventory, enforce DCA TP rule: only sell when >= WAP*(1+tp)
        wap = current_wap()
        tp_gate = (base > 1e-12 and wap > 0.0 and px >= wap * (1.0 + dca_tp_pct))

        while upper_levels and px >= upper_levels[0] and allow_sell:
            lv = upper_levels[0]

            # TP guard: if we hold inventory and we're in DCA mode, require profit over WAP
            # If no inventory / not DCA engaged, selling grid proceeds as usual
            if base > 1e-12 and dca_adds > 0:
                if not tp_gate and lv < wap * (1.0 + dca_tp_pct) - 1e-12:
                    # skip this upper level until TP achieved
                    upper_levels.pop(0)
                    continue

            qty_plan = per_order_quote / max(lv, 1e-12)
            qty = min(base, qty_plan)
            if qty <= 0:
                break

            fee = lv * qty * fee_rate
            proceeds = lv * qty - fee
            base  -= qty
            quote += proceeds
            # reduce cost_sum by sold portion at WAP proportion
            if base > 1e-12 and wap > 0.0:
                cost_sum -= wap * qty
                cost_sum = max(cost_sum, 0.0)
            else:
                # fully exited
                cost_sum = 0.0
                dca_adds = 0
                last_buy_px = None

            fees_paid += fee
            trades += 1
            upper_levels.pop(0)

            # recompute WAP & tp_gate after sell
            wap = current_wap()
            tp_gate = (base > 1e-12 and wap > 0.0 and px >= wap * (1.0 + dca_tp_pct))

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
