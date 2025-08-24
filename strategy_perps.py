# strategy_perps.py — AI + Indicator fused long/short backtest (spot data, perps logic)
from typing import List, Dict, Any, Optional, Tuple
from math import inf

# -----------------
# Basic indicators
# -----------------
def ema(series: List[float], n: int) -> List[float]:
    if n <= 1 or not series: return list(series)
    k = 2.0 / (n + 1.0)
    m = None; out = []
    for x in series:
        m = x if m is None else (x - m) * k + m
        out.append(m)
    return out

def rsi(prices: List[float], n: int = 14) -> List[float]:
    if not prices: return []
    gains, losses = [0.0], [0.0]
    for i in range(1, len(prices)):
        ch = prices[i] - prices[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    rsis = []
    avg_gain = sum(gains[1:n+1]) / n if len(gains) > n else 0.0
    avg_loss = sum(losses[1:n+1]) / n if len(losses) > n else 0.0
    rsis = [0.0] * len(prices)
    for i in range(n+1, len(prices)):
        avg_gain = (avg_gain*(n-1) + gains[i]) / n
        avg_loss = (avg_loss*(n-1) + losses[i]) / n
        rs = (avg_gain / avg_loss) if avg_loss > 1e-12 else float('inf')
        rsis[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsis

def macd(prices: List[float], fast=12, slow=26, signal=9) -> Tuple[List[float], List[float], List[float]]:
    fast_e = ema(prices, fast)
    slow_e = ema(prices, slow)
    macd_line = [ (f - s) for f, s in zip(fast_e, slow_e) ]
    sig_line = ema(macd_line, signal)
    hist = [ m - s for m, s in zip(macd_line, sig_line) ]
    return macd_line, sig_line, hist

def atr_from_candles(candles: List[dict], n: int = 14) -> List[float]:
    if not candles: return []
    tr = []
    for i, c in enumerate(candles):
        h, l = c["high"], c["low"]
        pc = candles[i-1]["close"] if i > 0 else c["close"]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    out = []
    for i in range(len(tr)):
        if i + 1 < n: out.append(0.0)
        else: out.append(sum(tr[i-n+1:i+1]) / n)
    return out

def vwap_from_candles(candles: List[dict]) -> List[float]:
    pv_sum = 0.0; v_sum = 0.0; out = []
    for c in candles:
        typical = (c["high"] + c["low"] + c["close"]) / 3.0
        vol = c.get("volume", 0.0)
        pv_sum += typical * vol
        v_sum += vol
        out.append((pv_sum / v_sum) if v_sum > 1e-12 else typical)
    return out

def bollinger(prices: List[float], n: int = 20, k: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    ma = []
    stds = []
    for i in range(len(prices)):
        if i+1 < n:
            ma.append(0.0); stds.append(0.0)
        else:
            w = prices[i-n+1:i+1]
            m = sum(w)/n
            v = sum((x-m)*(x-m) for x in w)/n
            s = v**0.5
            ma.append(m); stds.append(s)
    upper = [m + k*s for m, s in zip(ma, stds)]
    lower = [m - k*s for m, s in zip(ma, stds)]
    return lower, ma, upper

def max_drawdown(equity: List[float]) -> float:
    peak = -inf; max_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        if peak > 0:
            dd = (v - peak)/peak * 100.0
            if dd < max_dd: max_dd = dd
    return abs(max_dd)

# -----------------
# Fusion scoring
# -----------------
def clamp(x, a, b): return a if x < a else (b if x > b else x)

def indicator_score(prices: List[float], candles: List[dict]) -> List[float]:
    """Return score in [-1,1] each bar: >0 long bias, <0 short bias."""
    fast, slow = ema(prices, 20), ema(prices, 60)
    r = rsi(prices, 14)
    m_line, m_sig, m_hist = macd(prices, 12, 26, 9)
    vwap = vwap_from_candles(candles)
    bb_lo, bb_mid, bb_hi = bollinger(prices, 20, 2.0)

    scores = []
    for i in range(len(prices)):
        s = 0.0; wsum = 0.0

        # EMA trend
        if slow[i] > 0:
            trend = (fast[i] - slow[i]) / slow[i]
            s += clamp(trend*5.0, -1.0, 1.0) * 0.40; wsum += 0.40

        # RSI
        if r[i] > 0:
            # map RSI 30..70 roughly to -0.7..+0.7, clip outside
            rnorm = clamp((r[i]-50.0)/20.0, -1.0, 1.0)
            s += rnorm * 0.20; wsum += 0.20

        # MACD histogram sign/strength
        if m_hist[i] != 0 or m_line[i] != 0:
            macd_norm = clamp(m_hist[i] / (abs(m_line[i])+1e-12), -1.0, 1.0)
            s += macd_norm * 0.25; wsum += 0.25

        # VWAP location: above vwap => bullish, below => bearish
        if vwap[i] > 0:
            vloc = (prices[i] - vwap[i]) / vwap[i]
            s += clamp(vloc*5.0, -1.0, 1.0) * 0.10; wsum += 0.10

        # Bollinger squeeze/edge hint (touch lower => bearish exhaustion -> slight +; touch upper => slight -)
        if bb_hi[i] > bb_lo[i] > 0:
            if prices[i] <= bb_lo[i]: s += 0.05
            elif prices[i] >= bb_hi[i]: s -= 0.05
            wsum += 0.05

        scores.append(clamp((s/wsum) if wsum>0 else 0.0, -1.0, 1.0))
    return scores

# -----------------
# Perps-style backtest
# -----------------
def run_perps_backtest(
    symbol: str,
    candles: List[dict],
    *,
    start_equity: float = 1000.0,
    fee_rate: float = 0.0008,         # 0.08% per side (taker-ish)
    max_leverage: float = 3.0,
    risk_per_trade_pct: float = 0.01, # 1% of equity risk per trade
    atr_n: int = 14,
    sl_atr_mult: float = 1.5,
    tp_r_multiple: float = 1.5,
    trail_atr_mult: float = 1.0,
    enter_threshold: float = 0.25,    # score > +0.25 go long, < -0.25 go short
    exit_threshold: float = 0.05,     # neutrality to flatten when no SL/TP hit
    pyramids: int = 1,                # allow add-ons up to N (position scale-in)
    ai_bias: Optional[str] = None,    # "long","short","neutral", or None
    ai_confidence: float = 0.0,       # 0..1
    ai_crisis: bool = False,
) -> Dict[str, Any]:
    """
    Opens directional long/short using fused indicator score and AI bias.
    Position Sizing: ATR-based stop distance & 1% equity risk define position notional (capped by leverage).
    Exits: SL by ATR, TP by R-multiple, optional trailing stop.
    """
    if not candles:
        return {"trades":0, "net_pnl":0.0, "roi_pct":0.0, "equity_curve":[], "params":{}}

    prices = [c["close"] for c in candles]
    atr = atr_from_candles(candles, n=atr_n)
    score = indicator_score(prices, candles)

    # AI nudges score
    def ai_adj(s: float) -> float:
        if ai_bias is None or ai_confidence < 0.5: return s
        nud = 0.2 * ai_confidence  # up to +/-0.2 push
        if ai_bias == "long":  return clamp(s + nud, -1.0, 1.0)
        if ai_bias == "short": return clamp(s - nud, -1.0, 1.0)
        return s

    equity = start_equity
    equity_curve: List[Tuple[int,float]] = []
    trades = 0
    fees_paid = 0.0
    max_adds = max(0, pyramids)

    # position state
    side = 0  # +1 long, -1 short, 0 flat
    entry = 0.0
    size  = 0.0  # base units
    adds = 0
    stop = 0.0
    take = 0.0
    trail = None

    def flat_position(px):
        nonlocal side, entry, size, adds, stop, take, trail
        side = 0; entry = 0.0; size = 0.0; adds = 0; stop = 0.0; take = 0.0; trail = None

    for i in range(len(candles)):
        px = prices[i]
        ts = candles[i]["ts"]
        equity_curve.append((ts, equity))

        # ATR may be 0 early
        if atr[i] <= 0 or px <= 0:
            continue

        fused = ai_adj(score[i])

        # Crisis → no new entries; manage exits only
        can_enter = not ai_crisis

        # Manage trailing stop if in position
        if side != 0 and trail is not None:
            if side > 0:
                trail = max(trail, px - trail_atr_mult * atr[i])
                if px <= trail:
                    # exit long at trail
                    notional = size * px
                    fee = abs(notional) * fee_rate
                    pnl = (px - entry) * size - fee
                    equity += pnl
                    fees_paid += fee; trades += 1
                    flat_position(px)
            else:
                trail = min(trail, px + trail_atr_mult * atr[i])
                if px >= trail:
                    notional = size * px
                    fee = abs(notional) * fee_rate
                    pnl = (entry - px) * size - fee
                    equity += pnl
                    fees_paid += fee; trades += 1
                    flat_position(px)

        # Check SL/TP
        if side != 0:
            if side > 0:
                if px <= stop or px >= take:
                    notional = size * px
                    fee = abs(notional) * fee_rate
                    pnl = (px - entry) * size - fee
                    equity += pnl
                    fees_paid += fee; trades += 1
                    flat_position(px)
            else:
                if px >= stop or px <= take:
                    notional = size * px
                    fee = abs(notional) * fee_rate
                    pnl = (entry - px) * size - fee
                    equity += pnl
                    fees_paid += fee; trades += 1
                    flat_position(px)

        # Optional early flatten if bias evaporates
        if side != 0 and abs(fused) < exit_threshold:
            notional = size * px
            fee = abs(notional) * fee_rate
            pnl = (px - entry) * size if side>0 else (entry - px) * size
            pnl -= fee
            equity += pnl
            fees_paid += fee; trades += 1
            flat_position(px)

        # Entry / Add logic
        if side == 0 and can_enter:
            # Risk-based position sizing
            stop_dist = sl_atr_mult * atr[i]
            if stop_dist > 0:
                risk_amt = risk_per_trade_pct * equity
                qty = (risk_amt / stop_dist)
                notional = qty * px
                # leverage cap
                max_notional = equity * max_leverage
                if notional > max_notional:
                    scale = max_notional / notional
                    qty *= scale; notional *= scale
                # Entry decision
                if fused >= enter_threshold:
                    # open long
                    fee = abs(notional) * fee_rate
                    equity -= fee; fees_paid += fee
                    side = +1; entry = px; size = qty; adds = 0; trades += 1
                    stop = entry - stop_dist
                    take = entry + tp_r_multiple * stop_dist
                    trail = entry  # start trail baseline
                elif fused <= -enter_threshold:
                    # open short
                    fee = abs(notional) * fee_rate
                    equity -= fee; fees_paid += fee
                    side = -1; entry = px; size = qty; adds = 0; trades += 1
                    stop = entry + stop_dist
                    take = entry - tp_r_multiple * stop_dist
                    trail = entry
        elif side != 0 and can_enter and adds < max_adds:
            # Pyramid only if fused keeps pushing same direction
            if (side > 0 and fused >= enter_threshold*1.2) or (side < 0 and fused <= -enter_threshold*1.2):
                stop_dist = sl_atr_mult * atr[i]
                risk_amt = (risk_per_trade_pct * equity) * 0.7  # smaller risk for adds
                qty = (risk_amt / stop_dist)
                notional = qty * px
                if notional > equity * max_leverage:
                    qty = (equity * max_leverage) / px
                # add
                fee = abs(notional) * fee_rate
                equity -= fee; fees_paid += fee
                # re-average
                new_size = size + qty
                entry = (entry*size + px*qty) / max(new_size, 1e-12)
                size = new_size
                adds += 1; trades += 1
                # reset SL/TP by new entry
                if side > 0:
                    stop = entry - stop_dist
                    take = entry + tp_r_multiple * stop_dist
                else:
                    stop = entry + stop_dist
                    take = entry - tp_r_multiple * stop_dist
                # keep trail concept
                trail = entry if trail is None else trail

    final_value = equity
    roi = (final_value - start_equity)/start_equity*100.0 if start_equity>0 else 0.0
    mdd = max_drawdown([v for _, v in equity_curve])
    return {
        "symbol": symbol,
        "trades": trades,
        "fees_paid": fees_paid,
        "net_pnl": final_value - start_equity,
        "roi_pct": roi,
        "final_value": final_value,
        "max_drawdown_pct": mdd,
        "equity_curve": equity_curve,
        "params": {
            "start_equity": start_equity,
            "fee_rate": fee_rate,
            "max_leverage": max_leverage,
            "risk_per_trade_pct": risk_per_trade_pct,
            "atr_n": atr_n,
            "sl_atr_mult": sl_atr_mult,
            "tp_r_multiple": tp_r_multiple,
            "trail_atr_mult": trail_atr_mult,
            "enter_threshold": enter_threshold,
            "exit_threshold": exit_threshold,
            "pyramids": pyramids,
            "ai_bias": ai_bias or "none",
            "ai_confidence": ai_confidence,
            "ai_crisis": ai_crisis,
        }
    }
