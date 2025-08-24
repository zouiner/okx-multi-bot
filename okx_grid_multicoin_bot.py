#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, logging, io
from datetime import datetime
from typing import Dict, Any, Tuple

from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from storage import Storage, summarize_symbol, export_trades_csv
from data import fetch_ohlcv, fetch_last_price
from strategy_grid import run_grid_backtest
from strategy_perps import run_perps_backtest

# Gemini AI
try:
    from agent_gemini import GeminiAgent
except Exception:
    GeminiAgent = None  # handle gracefully if module missing

# matplotlib for charts
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# ---------- Load configs.json ----------
CONFIG: Dict[str, Any] = {}
if os.path.exists("configs.json"):
    with open("configs.json") as f:
        CONFIG = json.load(f)
else:
    raise FileNotFoundError("⚠️ configs.json not found. Create it and put your keys there.")

TELEGRAM_BOT_TOKEN = CONFIG.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = CONFIG.get("TELEGRAM_CHAT_ID")
GEMINI_KEY = CONFIG.get("GEMINI_API_KEY")
GEMINI_MODEL = CONFIG.get("GEMINI_MODEL", "gemini-2.5-flash")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing in configs.json")

# init AI agent if key provided
ai_agent = GeminiAgent(GEMINI_KEY, GEMINI_MODEL) if (GEMINI_KEY and GeminiAgent) else None

# ---------- Commands registry ----------
COMMAND_SPECS = [
    {"cmd":"start","args":"","desc":"Show welcome message.","example":"/start"},
    {"cmd":"help","args":"[command]","desc":"Show all commands or details for one.","example":"/help add"},
    {"cmd":"ping","args":"","desc":"Health check.","example":"/ping"},
    {"cmd":"add","args":"SYMBOL [interval|auto] [paper|live] [quote_cap]","desc":"Track a coin (paper by default). If interval omitted or 'auto', it picks based on global auto_days.","example":"/add SUI-USDT auto paper 1000"},
    {"cmd":"list","args":"","desc":"Show all tracked coins and basic settings.","example":"/list"},
    {"cmd":"status","args":"[SYMBOL]","desc":"Status for one symbol or all (incl. AI/DCA).","example":"/status SUI-USDT"},
    {"cmd":"set","args":"<SYMBOL|global> <key> <value>","desc":"Tune params. See /help for keys.","example":"/set global dca_tp_pct 0.10"},
    {"cmd":"history","args":"SYMBOL [N]","desc":"Show last N trades for SYMBOL.","example":"/history SUI-USDT 10"},
    {"cmd":"summary","args":"SYMBOL [days]","desc":"Aggregate stats over the window.","example":"/summary SUI-USDT 7"},
    {"cmd":"summaryall","args":"[days]","desc":"Aggregate stats for ALL tracked coins.","example":"/summaryall 7"},
    {"cmd":"export","args":"SYMBOL","desc":"Export all trades for SYMBOL to CSV.","example":"/export SUI-USDT"},
    {"cmd":"backtest","args":"SYMBOL [days]","desc":"Spot grid backtest (auto timeframe) with AI+DCA overlay.","example":"/backtest SUI-USDT 30"},
    {"cmd":"backtestperp","args":"SYMBOL [days]","desc":"AI+indicator fused long/short perps-style backtest.","example":"/backtestperp BTC-USDT 60"},
    {"cmd":"autotune","args":"on|off [hours] [apply:true|false]","desc":"Schedule AI parameter tuning every N hours.","example":"/autotune on 12 false"},
    {"cmd":"tune","args":"SYMBOL","desc":"Run a single AI parameter tuning cycle for SYMBOL now.","example":"/tune BTC-USDT"},
    {"cmd":"tuneall","args":"","desc":"Run a single AI parameter tuning cycle for all tracked symbols.","example":"/tuneall"},
    {"cmd":"paperlive","args":"on|off [poll_seconds]","desc":"Toggle real‑time PAPER trading (simulated) with indicator/AI signals.","example":"/paperlive on 15"},
]
def _cmd_index(): return {c["cmd"]: c for c in COMMAND_SPECS}
def build_help_text(cmd: str | None = None) -> str:
    idx = _cmd_index()
    if cmd and cmd in idx:
        c = idx[cmd]; lines = [f"🔹 /{c['cmd']} {c['args']}".strip(), f"   {c['desc']}"]
        if c.get("example"): lines.append(f"   Example: {c['example']}")
        return "\n".join(lines)
    lines = ["🤖 Available commands:\n"]; pad = max(len(c["cmd"]) for c in COMMAND_SPECS)
    for c in COMMAND_SPECS:
        args = f" {c['args']}" if c["args"] else ""
        lines.append(f"/{c['cmd']:<{pad}}{args}"); lines.append(f"  {c['desc']}")
    lines.append("\nTip: `/help backtest` for details."); return "\n".join(lines)

# ---------- Interval picker ----------
BAR_CANDLES_PER_DAY = [
    ("1m", 1440), ("3m", 480), ("5m", 288), ("15m", 96), ("30m", 48),
    ("1H", 24), ("2H", 12), ("4H", 6), ("6H", 4), ("12H", 2), ("1D", 1),
]
def pick_bar_for_days(days: int, target_candles: int = 900) -> str:
    days = max(1, int(days))
    for bar, per_day in BAR_CANDLES_PER_DAY:
        if days * per_day <= target_candles:
            return bar
    return "1D"

# ---------- Utils ----------
def parse_bool(s: str) -> bool:
    s = (s or "").strip().lower()
    return s in ("1","true","yes","y","on")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def bounded_update(old: float | int | None, target: float | int, cap_frac: float, hard_lo: float, hard_hi: float):
    # limit how far we move from old toward target this cycle
    if old is None:
        return clamp(target, hard_lo, hard_hi)
    delta = target - old
    max_step = abs(old) * cap_frac
    if abs(delta) > max_step:
        target = old + (max_step if delta > 0 else -max_step)
    return clamp(target, hard_lo, hard_hi)

def ema(series, n):
    k=2/(n+1); m=None; out=[]
    for x in series:
        m = x if m is None else (x-m)*k+m
        out.append(m)
    return out

def build_snapshot(candles):
    # simple example volatility via ATR% and trend via EMA
    if not candles:
        return {"price": 0, "volatility": 0.0, "trend": "flat", "liquidity": "med"}
    prices = [c["close"] for c in candles]
    price = prices[-1]
    # ATR rough
    tr = []
    for i,c in enumerate(candles):
        h,l = c["high"], c["low"]
        pc = candles[i-1]["close"] if i>0 else c["close"]
        tr.append(max(h-l, abs(h-pc), abs(l-pc)))
    atr = sum(tr[-14:])/14.0 if len(tr)>=14 else (sum(tr)/max(1,len(tr)))
    vol_pct = (atr/price) if price>0 else 0.0

    fast, slow = ema(prices, 20), ema(prices, 60)
    trend = "up" if fast[-1] > slow[-1]*1.002 else ("down" if fast[-1] < slow[-1]*0.998 else "flat")

    liq = "high" if len(candles) >= 500 else ("med" if len(candles)>=200 else "low")
    return {"price": price, "volatility": round(vol_pct,4), "trend": trend, "liquidity": liq}

# ---------- Global State ----------
state: Dict[str, Any] = {
    "bots": {},
    "global": {
        # Grid defaults
        "grids": 12,
        "per_order_quote": 40.0,
        "min_notional": 5.0,
        "fee_rate": 0.001,
        "atr_mult": 2.0,
        "ema_fast_n": 20,
        "ema_slow_n": 60,
        "recenter_threshold": 0.35,
        "recenter_every": 240,
        "auto_days": 7,             # for auto interval choice on /add
        # AI + DCA defaults
        "ai_enabled": True,
        "dca_tp_pct": 0.10,
        "dca_add_drawdown_pct": 0.15,
        "dca_max_adds": 2,
        "dca_equity_fraction": 0.10,
        "atr_extreme_mult": 4.0,
        # Perps defaults
        "perps_start_equity": 1000.0,
        "perps_fee_rate": 0.0008,
        "perps_max_leverage": 3.0,
        "perps_risk_per_trade_pct": 0.01,
        "perps_atr_n": 14,
        "perps_sl_atr_mult": 1.5,
        "perps_tp_r_multiple": 1.5,
        "perps_trail_atr_mult": 1.0,
        "perps_enter_threshold": 0.25,
        "perps_exit_threshold": 0.05,
        "perps_pyramids": 1,
        # AI Autotune
        "ai_autotune_enabled": False,
        "ai_autotune_hours": 12,
        "ai_autotune_apply": False,   # False = dry-run report only
        "ai_change_cap_pct": 0.20,    # max ±20% per cycle
        # Paper-live mode
        "paper_live_enabled": False,
        "paper_poll_seconds": 15,
        "paper_virtual_equity": 1000.0,  # used for risk sizing in paper-live
    }
}
storage = Storage("bot_history.db")

# job handles
autotune_job_handle = None
paper_job_handle = None

# in-memory paper positions for paper-live
paper_pos: Dict[str, Dict[str, float]] = {}  # sym -> {side, entry, size, stop, take}

# ---------- Alerts ----------
async def push_alert(app, text: str):
    if TELEGRAM_CHAT_ID:
        try:
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        except Exception:
            pass

# ---------- Handlers ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Multi-coin bot ready. Try `/help`.\n"
        "Add a coin: `/add SUI-USDT auto paper 1000`"
    )

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    arg = ctx.args[0].lower() if ctx.args else None
    await update.message.reply_text(build_help_text(arg), parse_mode=ParseMode.MARKDOWN)

async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

async def add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /add SYMBOL [interval|auto] [paper|live] [quote_cap]
    If interval omitted or set to 'auto', pick bar via pick_bar_for_days(global.auto_days).
    Also seeds per-symbol AI/DCA from global defaults.
    """
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /add SYMBOL [interval|auto] [paper|live] [quote_cap]")
    sym = ctx.args[0].upper()

    # interval logic
    if len(ctx.args) > 1:
        raw_interval = ctx.args[1].lower()
        if raw_interval == "auto":
            auto_days = int(state["global"].get("auto_days", 7))
            interval = pick_bar_for_days(auto_days)
        else:
            interval = ctx.args[1]
    else:
        auto_days = int(state["global"].get("auto_days", 7))
        interval = pick_bar_for_days(auto_days)

    mode = (ctx.args[2].lower() if len(ctx.args) > 2 else "paper")
    quote_cap = float(ctx.args[3]) if len(ctx.args) > 3 else 1000.0

    g = state["global"]
    state["bots"][sym] = {
        "interval": interval, "mode": mode, "quote_cap": quote_cap,
        "grids": g["grids"],
        "per_order_quote": g["per_order_quote"],
        "min_notional": g["min_notional"],
        "added_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        # per-symbol AI/DCA (seeded from global; editable via /set SYMBOL ...)
        "ai_enabled": bool(g.get("ai_enabled", True)),
        "dca_tp_pct": float(g.get("dca_tp_pct", 0.10)),
        "dca_add_drawdown_pct": float(g.get("dca_add_drawdown_pct", 0.15)),
        "dca_max_adds": int(g.get("dca_max_adds", 2)),
        "dca_equity_fraction": float(g.get("dca_equity_fraction", 0.10)),
        "atr_extreme_mult": float(g.get("atr_extreme_mult", 4.0)),
        # perps overrides default to global unless changed per symbol
    }
    await update.message.reply_text(f"➕ Added {sym} {state['bots'][sym]}")

async def list_bots(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not state["bots"]:
        return await update.message.reply_text("No coins tracked. Use `/add SYMBOL`")
    lines = ["📊 Tracking:"]
    for k, v in state["bots"].items():
        lines.append(f"- {k}: {v}")
    await update.message.reply_text("\n".join(lines))

async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        sym = ctx.args[0].upper()
        if sym not in state["bots"]:
            return await update.message.reply_text("Unknown symbol.")
        try:
            price = fetch_last_price(sym) or 0.0
        except Exception:
            price = 0.0
        cfg = state["bots"][sym]
        mode_str = "PAPER (simulated)" if cfg.get("mode") == "paper" else "LIVE (real orders)"
        await update.message.reply_text(
            f"{sym} @ {price:.6f} | interval={cfg['interval']} | mode={mode_str}\n"
            f"grids={cfg.get('grids')} per_order={cfg.get('per_order_quote')} min_notional={cfg.get('min_notional')}\n"
            f"AI={cfg.get('ai_enabled', True)} DCA(tp={cfg.get('dca_tp_pct',0.1)}, add_at={cfg.get('dca_add_drawdown_pct',0.15)}, "
            f"max_adds={cfg.get('dca_max_adds',2)}, eq_frac={cfg.get('dca_equity_fraction',0.10)}, atr_mult={cfg.get('atr_extreme_mult',4.0)})"
        )
    else:
        if not state["bots"]:
            return await update.message.reply_text("No coins tracked.")
        lines = []
        for sym, cfg in state["bots"].items():
            try:
                price = fetch_last_price(sym) or 0.0
            except Exception:
                price = 0.0
            mode_str = "PAPER (simulated)" if cfg.get("mode") == "paper" else "LIVE (real orders)"
            lines.append(f"{sym} @ {price:.6f} | {cfg['interval']} | {mode_str} | AI={cfg.get('ai_enabled', True)}")
        await update.message.reply_text("\n".join(lines))

async def set_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 3:
        return await update.message.reply_text("Usage: /set <SYMBOL|global> <key> <value>")
    target, key, val = ctx.args[0], ctx.args[1], " ".join(ctx.args[2:])
    if target.lower() == "global":
        g = state["global"]
        try:
            if key in ("grids","ema_fast_n","ema_slow_n","recenter_every","auto_days","dca_max_adds",
                       "perps_atr_n","perps_pyramids","ai_autotune_hours","paper_poll_seconds"):
                g[key] = int(val)
            elif key in ("per_order_quote","min_notional","fee_rate","atr_mult","recenter_threshold",
                         "dca_tp_pct","dca_add_drawdown_pct","dca_equity_fraction","atr_extreme_mult",
                         "perps_start_equity","perps_fee_rate","perps_max_leverage","perps_risk_per_trade_pct",
                         "perps_sl_atr_mult","perps_tp_r_multiple","perps_trail_atr_mult",
                         "perps_enter_threshold","perps_exit_threshold","ai_change_cap_pct","paper_virtual_equity"):
                g[key] = float(val)
            elif key in ("ai_enabled","ai_autotune_enabled","ai_autotune_apply","paper_live_enabled"):
                g[key] = parse_bool(val)
            else:
                return await update.message.reply_text("Unknown global key.")
            return await update.message.reply_text(f"Updated global {key} = {g[key]}")
        except Exception as e:
            return await update.message.reply_text(f"Error: {e}")
    # per-symbol
    sym = target.upper()
    if sym not in state["bots"]:
        return await update.message.reply_text("Unknown symbol.")
    try:
        if key == "interval":
            state["bots"][sym]["interval"] = val
        elif key == "mode":
            state["bots"][sym]["mode"] = val.lower()
        elif key == "quote_cap":
            state["bots"][sym]["quote_cap"] = float(val)
        elif key == "grids":
            state["bots"][sym]["grids"] = int(val)
        elif key == "per_order_quote":
            state["bots"][sym]["per_order_quote"] = float(val)
        elif key == "min_notional":
            state["bots"][sym]["min_notional"] = float(val)
        # AI/DCA per-symbol
        elif key == "ai_enabled":
            state["bots"][sym]["ai_enabled"] = parse_bool(val)
        elif key == "dca_tp_pct":
            state["bots"][sym]["dca_tp_pct"] = float(val)
        elif key == "dca_add_drawdown_pct":
            state["bots"][sym]["dca_add_drawdown_pct"] = float(val)
        elif key == "dca_max_adds":
            state["bots"][sym]["dca_max_adds"] = int(val)
        elif key == "dca_equity_fraction":
            state["bots"][sym]["dca_equity_fraction"] = float(val)
        elif key == "atr_extreme_mult":
            state["bots"][sym]["atr_extreme_mult"] = float(val)
        # Perps per-symbol overrides
        elif key.startswith("perps_"):
            try_val = float(val)
            if key in ("perps_atr_n","perps_pyramids"):
                try_val = int(try_val)
            state["bots"][sym][key] = try_val
        else:
            return await update.message.reply_text("Unknown key.")
        await update.message.reply_text(f"Updated {sym} {key} = {state['bots'][sym][key]}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /history SYMBOL [N]")
    symbol = ctx.args[0].upper()
    n = int(ctx.args[1]) if len(ctx.args) > 1 else 10
    rows = storage.get_trades(symbol, n)
    if not rows:
        return await update.message.reply_text(f"No trades for {symbol}")
    lines = [f"📜 History {symbol}:"]
    for ts, side, price, qty in rows:
        dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{dt} {side.upper():4}  px={price:.6f}  qty={qty:.6f}")
    await update.message.reply_text("\n".join(lines[:60]))

async def summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /summary SYMBOL [days]")
    symbol = ctx.args[0].upper()
    days = int(ctx.args[1]) if len(ctx.args) > 1 else 7
    s = summarize_symbol("bot_history.db", symbol, days)
    b = s["by_side"]; buys=b.get("buy",{"count":0,"notional":0.0}); sells=b.get("sell",{"count":0,"notional":0.0})
    cfg = state["bots"].get(symbol, {})
    mode_str = "PAPER (simulated)" if cfg.get("mode") == "paper" else "LIVE (real orders)"
    await update.message.reply_text(
        f"Summary {symbol} (last {days}d)\n"
        f"Trades: {s['total_trades']}\n"
        f"Buys:  {buys['count']}  Notional: {buys['notional']:.2f}\n"
        f"Sells: {sells['count']}  Notional: {sells['notional']:.2f}\n"
        f"Mode: {mode_str}"
    )

async def summaryall(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    days = int(ctx.args[0]) if ctx.args else 7
    if not state["bots"]:
        return await update.message.reply_text("No coins tracked.")
    lines = [f"📊 Summary (last {days}d):"]
    for sym in state["bots"].keys():
        s = summarize_symbol("bot_history.db", sym, days)
        b = s["by_side"]; buys=b.get("buy",{"count":0,"notional":0.0}); sells=b.get("sell",{"count":0,"notional":0.0})
        mode_str = "PAPER" if state['bots'][sym].get("mode") == "paper" else "LIVE"
        lines.append(f"{sym}: {s['total_trades']} | Buys {buys['count']}({buys['notional']:.2f}) | "
                     f"Sells {sells['count']}({sells['notional']:.2f}) | Mode={mode_str}")
    await update.message.reply_text("\n".join(lines))

async def export_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /export SYMBOL")
    symbol = ctx.args[0].upper()
    out = f"{symbol.replace('-','_')}_trades.csv"
    export_trades_csv("bot_history.db", out, symbol)
    await update.message.reply_text(f"Exported to {os.path.abspath(out)}")

async def backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /backtest SYMBOL [days]
    Automatically picks the best bar. Uses Gemini AI to steer bias/crisis (if enabled) and DCA overlay.
    """
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /backtest SYMBOL [days]")
    symbol = ctx.args[0].upper()
    days = int(ctx.args[1]) if len(ctx.args) > 1 else 7

    bar = pick_bar_for_days(days)
    await update.message.reply_text(f"⏳ Backtest {symbol} over {days}d ({bar} candles)...")

    try:
        candles = fetch_ohlcv(symbol, bar=bar, days=days)
    except Exception as e:
        return await update.message.reply_text(f"Fetch error: {e}")
    if not candles:
        return await update.message.reply_text("No candles fetched.")

    # AI signal (optional)
    ai_bias = None; ai_conf = 0.0; ai_crisis = False; ai_used = False

    g = state["global"]
    cfg = state["bots"].get(symbol, {})
    ai_enabled = bool(cfg.get("ai_enabled", g.get("ai_enabled", True)))
    dca_tp_pct = float(cfg.get("dca_tp_pct", g.get("dca_tp_pct", 0.10)))
    dca_add_drawdown_pct = float(cfg.get("dca_add_drawdown_pct", g.get("dca_add_drawdown_pct", 0.15)))
    dca_max_adds = int(cfg.get("dca_max_adds", g.get("dca_max_adds", 2)))
    dca_equity_fraction = float(cfg.get("dca_equity_fraction", g.get("dca_equity_fraction", 0.10)))
    atr_extreme_mult = float(cfg.get("atr_extreme_mult", g.get("atr_extreme_mult", 4.0)))

    if ai_enabled and ai_agent:
        try:
            last_price = candles[-1]["close"]
            sig = ai_agent.get_signal(symbol, last_price)
            rec = (str(sig.get("recommendation","")).lower())
            if "long" in rec or "buy" in rec:   ai_bias = "long"
            elif "short" in rec or "sell" in rec: ai_bias = "short"
            else: ai_bias = "neutral"
            ai_conf = float(sig.get("confidence") or 0.0)
            ai_crisis = bool(sig.get("crisis") or False)
            ai_used = True
        except Exception:
            pass

    res = run_grid_backtest(
        symbol, candles,
        quote_cap=float(cfg.get("quote_cap", 1000.0)),
        grid_size=int(cfg.get("grids", g["grids"])),
        per_order_quote=float(cfg.get("per_order_quote", g["per_order_quote"])),
        fee_rate=float(g["fee_rate"]),
        atr_mult=float(g["atr_mult"]),
        ema_fast_n=int(g["ema_fast_n"]),
        ema_slow_n=int(g["ema_slow_n"]),
        recenter_threshold=float(g["recenter_threshold"]),
        recenter_every=int(g["recenter_every"]),
        # AI + DCA overlay
        ai_bias=ai_bias,
        ai_confidence=ai_conf,
        ai_crisis=ai_crisis,
        dca_tp_pct=dca_tp_pct,
        dca_add_drawdown_pct=dca_add_drawdown_pct,
        dca_max_adds=dca_max_adds,
        dca_equity_fraction=dca_equity_fraction,
        atr_extreme_mult=atr_extreme_mult,
    )

    ai_line = f"\nAI: bias={ai_bias} conf={ai_conf:.2f} crisis={ai_crisis}" if ai_used else ""
    msg = (
        f"📊 Backtest {symbol} ({days}d, {bar})\n"
        f"Trades: {res['trades']}\n"
        f"Gross PnL: {res['gross_pnl']:.2f}\n"
        f"Fees Paid: {res['fees_paid']:.2f}\n"
        f"Net PnL: {res['net_pnl']:.2f}\n"
        f"Net ROI: {res['roi_pct']:.2f}%\n"
        f"Max Drawdown: {res['max_drawdown_pct']:.2f}%\n"
        f"Avg Gross/Trade: {res.get('avg_gross_per_trade',0):.4f}\n"
        f"Avg Fee/Trade:   {res.get('avg_fee_per_trade',0):.4f}\n"
        f"DCA: tp={dca_tp_pct:.2f}, add_at={dca_add_drawdown_pct:.2f}, max_adds={dca_max_adds}, eq_frac={dca_equity_fraction:.2f}, atr_mult={atr_extreme_mult:.2f}"
        f"{ai_line}\n"
        f"Final Value: {res['final_value']:.2f} (Start {res['params']['start_equity']:.2f})\n"
        f"Params: grids={res['params']['grid_size']}, per_order={res['params']['per_order_quote']}, "
        f"fee_rate={res['params']['fee_rate']}, atr_mult={res['params']['atr_mult']}, "
        f"ema=({res['params']['ema_fast_n']},{res['params']['ema_slow_n']}), "
        f"recent_th={res['params']['recenter_threshold']}, recent_every={res['params']['recenter_every']}, "
        f"min_step_pct≈{res['params']['min_step_pct']:.4f}"
    )
    await update.message.reply_text(msg)

    eq = res.get("equity_curve", [])
    if eq:
        xs_ms = [t for t, _ in eq]
        ys = [v for _, v in eq]
        xs = [datetime.utcfromtimestamp(t/1000.0) for t in xs_ms]

        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)
        ax.set_title(f"Equity Curve — {symbol}")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Equity")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=buf, caption=f"Equity Curve: {symbol} ({days}d, {bar})")

async def backtestperp(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /backtestperp SYMBOL [days]")
    symbol = ctx.args[0].upper()
    days = int(ctx.args[1]) if len(ctx.args) > 1 else 30

    bar = pick_bar_for_days(days)
    await update.message.reply_text(f"⏳ Perps Backtest {symbol} over {days}d ({bar})...")

    try:
        candles = fetch_ohlcv(symbol, bar=bar, days=days)
    except Exception as e:
        return await update.message.reply_text(f"Fetch error: {e}")
    if not candles:
        return await update.message.reply_text("No candles fetched.")

    g = state["global"]
    cfg = state["bots"].get(symbol, {})

    # AI (optional)
    ai_bias = None; ai_conf = 0.0; ai_crisis = False; ai_used = False
    ai_enabled = bool(cfg.get("ai_enabled", g.get("ai_enabled", True)))
    if ai_enabled and ai_agent:
        try:
            last_price = candles[-1]["close"]
            sig = ai_agent.get_signal(symbol, last_price)
            rec = (str(sig.get("recommendation","")).lower())
            if "long" in rec or "buy" in rec:   ai_bias = "long"
            elif "short" in rec or "sell" in rec: ai_bias = "short"
            else: ai_bias = "neutral"
            ai_conf = float(sig.get("confidence") or 0.0)
            ai_crisis = bool(sig.get("crisis") or False)
            ai_used = True
        except Exception:
            pass

    def getp(key, default):
        return cfg.get(key, g.get(key, default))

    res = run_perps_backtest(
        symbol, candles,
        start_equity=float(getp("perps_start_equity", 1000.0)),
        fee_rate=float(getp("perps_fee_rate", 0.0008)),
        max_leverage=float(getp("perps_max_leverage", 3.0)),
        risk_per_trade_pct=float(getp("perps_risk_per_trade_pct", 0.01)),
        atr_n=int(getp("perps_atr_n", 14)),
        sl_atr_mult=float(getp("perps_sl_atr_mult", 1.5)),
        tp_r_multiple=float(getp("perps_tp_r_multiple", 1.5)),
        trail_atr_mult=float(getp("perps_trail_atr_mult", 1.0)),
        enter_threshold=float(getp("perps_enter_threshold", 0.25)),
        exit_threshold=float(getp("perps_exit_threshold", 0.05)),
        pyramids=int(getp("perps_pyramids", 1)),
        ai_bias=ai_bias, ai_confidence=ai_conf, ai_crisis=ai_crisis
    )

    msg = (
        f"📈 Perps Backtest {symbol} ({days}d, {bar})\n"
        f"Trades: {res['trades']}\n"
        f"Net PnL: {res['net_pnl']:.2f}\n"
        f"Net ROI: {res['roi_pct']:.2f}%\n"
        f"Max Drawdown: {res['max_drawdown_pct']:.2f}%\n"
        f"Final Value: {res['final_value']:.2f} (Start {res['params']['start_equity']:.2f})\n"
        f"Risk: fee={res['params']['fee_rate']:.4f} lev={res['params']['max_leverage']:.1f} "
        f"RPT={res['params']['risk_per_trade_pct']:.3f} ATRn={res['params']['atr_n']} "
        f"SL={res['params']['sl_atr_mult']:.2f}×ATR TP={res['params']['tp_r_multiple']:.2f}R Trail={res['params']['trail_atr_mult']:.2f}×ATR\n"
        f"Enter>|{res['params']['enter_threshold']:.2f}| Exit<{res['params']['exit_threshold']:.2f} Pyramids={res['params']['pyramids']}\n"
        f"AI: bias={res['params']['ai_bias']} conf={res['params']['ai_confidence']:.2f} crisis={res['params']['ai_crisis']}"
    )
    await update.message.reply_text(msg)

    try:
        xs = [c[0] for c in res["equity_curve"]]
        ys = [c[1] for c in res["equity_curve"]]
        xs_dt = [datetime.utcfromtimestamp(t/1000.0) for t in xs]
        fig = plt.figure(figsize=(8,3))
        ax = fig.add_subplot(111)
        ax.plot(xs_dt, ys)
        ax.set_title(f"Perps Equity — {symbol}")
        ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Equity")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150)
        plt.close(fig); buf.seek(0)
        await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=buf, caption=f"Perps Equity: {symbol} ({days}d, {bar})")
    except Exception:
        pass

# ---------- PAPER-LIVE: real-time paper trading ----------
async def paperlive_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """ /paperlive on|off [poll_seconds] """
    global paper_job_handle
    g = state["global"]
    if not ctx.args:
        return await update.message.reply_text(
            f"paper_live_enabled={g.get('paper_live_enabled')} every {g.get('paper_poll_seconds')}s "
            f"(virtual_equity={g.get('paper_virtual_equity')})"
        )

    mode = ctx.args[0].lower()
    if mode in ("on","off"):
        g["paper_live_enabled"] = (mode == "on")
    if len(ctx.args) >= 2:
        try: g["paper_poll_seconds"] = int(ctx.args[1])
        except: pass

    # reschedule job
    if paper_job_handle:
        try:
            paper_job_handle.schedule_removal()
        except Exception:
            pass

    if g["paper_live_enabled"]:
        paper_job_handle = ctx.application.job_queue.run_repeating(
            paperlive_job, interval=max(5, g["paper_poll_seconds"]), first=5
        )
    else:
        paper_job_handle = None

    await update.message.reply_text(
        f"Paper-live set: enabled={g['paper_live_enabled']} every {g['paper_poll_seconds']}s "
        f"(virtual_equity={g['paper_virtual_equity']})"
    )

async def paperlive_job(context: ContextTypes.DEFAULT_TYPE):
    """Polls latest candles and simulates perps-style entries/exits in PAPER mode only."""
    from strategy_perps import indicator_score, atr_from_candles  # lazy import
    app = context.application
    g = state["global"]
    if not g.get("paper_live_enabled", False):
        return

    rpt = float(g.get("perps_risk_per_trade_pct", 0.01))
    sl_mult = float(g.get("perps_sl_atr_mult", 1.5))
    tp_R = float(g.get("perps_tp_r_multiple", 1.5))
    enter_th = float(g.get("perps_enter_threshold", 0.25))
    exit_th = float(g.get("perps_exit_threshold", 0.05))
    virt_eq = float(g.get("paper_virtual_equity", 1000.0))

    for sym, cfg in list(state["bots"].items()):
        if cfg.get("mode") != "paper":
            continue

        bar = cfg.get("interval", "1m")
        try:
            candles = fetch_ohlcv(sym, bar=bar, days=1)  # recent history
        except Exception:
            continue
        if not candles or len(candles) < 60:
            continue

        prices = [c["close"] for c in candles]
        score = indicator_score(prices, candles)[-1]
        atr = atr_from_candles(candles, n=int(g.get("perps_atr_n", 14)))[-1]
        px = prices[-1]

        pos = paper_pos.setdefault(sym, {"side":0.0,"entry":0.0,"size":0.0,"stop":0.0,"take":0.0})
        side = int(pos["side"])

        # flatten on neutrality
        if side != 0 and abs(score) < exit_th:
            pnl = (px - pos["entry"]) * pos["size"] if side>0 else (pos["entry"] - px) * pos["size"]
            paper_pos[sym] = {"side":0.0,"entry":0.0,"size":0.0,"stop":0.0,"take":0.0}
            await push_alert(app, f"📄 {sym} PAPER exit @ {px:.6f}  PnL={pnl:.4f} (neutrality)")
            continue

        # if flat, enter
        if side == 0 and atr > 0:
            stop_dist = sl_mult * atr
            size = (rpt * virt_eq) / max(stop_dist, 1e-12)
            if score >= enter_th:
                paper_pos[sym].update({"side":+1,"entry":px,"size":size,"stop":px-stop_dist,"take":px+tp_R*stop_dist})
                await push_alert(app, f"📄 {sym} PAPER LONG @ {px:.6f} | size={size:.6f} stop={px-stop_dist:.6f} take={px+tp_R*stop_dist:.6f}")
                continue
            elif score <= -enter_th:
                paper_pos[sym].update({"side":-1,"entry":px,"size":size,"stop":px+stop_dist,"take":px-tp_R*stop_dist})
                await push_alert(app, f"📄 {sym} PAPER SHORT @ {px:.6f} | size={size:.6f} stop={px+stop_dist:.6f} take={px-tp_R*stop_dist:.6f}")
                continue

        # manage SL/TP
        if side != 0:
            stop = pos["stop"]; take = pos["take"]
            hit = (px <= stop or px >= take) if side>0 else (px >= stop or px <= take)
            if hit:
                pnl = (px - pos["entry"]) * pos["size"] if side>0 else (pos["entry"] - px) * pos["size"]
                paper_pos[sym] = {"side":0.0,"entry":0.0,"size":0.0,"stop":0.0,"take":0.0}
                await push_alert(app, f"📄 {sym} PAPER SL/TP exit @ {px:.6f}  PnL={pnl:.4f}")

# ---------- AI Autotune ----------
async def autotune_once(app, symbol: str, days: int = 7):
    g = state["global"]
    cfg = state["bots"].get(symbol, {})
    if not cfg:
        return f"{symbol}: not tracked."

    bar = pick_bar_for_days(days)
    try:
        candles = fetch_ohlcv(symbol, bar=bar, days=days)
    except Exception as e:
        return f"{symbol}: fetch error {e}"
    if not candles:
        return f"{symbol}: no candles."

    snap = build_snapshot(candles)
    if not ai_agent:
        return f"{symbol}: AI unavailable."

    try:
        props = ai_agent.suggest_params(symbol, snap) or {}
    except Exception as e:
        return f"{symbol}: AI error {e}"

    cap = float(g.get("ai_change_cap_pct", 0.20))
    apply_changes = bool(g.get("ai_autotune_apply", False))

    before = dict(cfg)
    after = dict(cfg)

    def upd(key, target, lo, hi, cast=float):
        if target is None: return
        old = after.get(key, None)
        try:
            target = cast(target)
        except Exception:
            return
        newv = bounded_update(old if old is not None else target, target, cap, lo, hi)
        after[key] = newv

    grid = (props.get("grid") or {})
    upd("grids", grid.get("grids"), 4, 60, int)
    upd("per_order_quote", grid.get("per_order_quote"), 2.0, 500.0, float)
    upd("atr_mult", grid.get("atr_mult"), 0.8, 5.0, float)
    upd("ema_fast_n", grid.get("ema_fast_n"), 5, 80, int)
    upd("ema_slow_n", grid.get("ema_slow_n"), 20, 240, int)
    upd("recenter_threshold", grid.get("recenter_threshold"), 0.10, 0.80, float)
    upd("recenter_every", grid.get("recenter_every"), 30, 2880, int)
    upd("dca_tp_pct", grid.get("dca_tp_pct"), 0.03, 0.30, float)
    upd("dca_add_drawdown_pct", grid.get("dca_add_drawdown_pct"), 0.05, 0.40, float)
    upd("dca_max_adds", grid.get("dca_max_adds"), 0, 5, int)
    upd("dca_equity_fraction", grid.get("dca_equity_fraction"), 0.02, 0.30, float)

    perps = (props.get("perps") or {})
    upd("perps_risk_per_trade_pct", perps.get("risk_per_trade_pct"), 0.001, 0.03, float)
    upd("perps_max_leverage", perps.get("max_leverage"), 1.0, 5.0, float)
    upd("perps_sl_atr_mult", perps.get("sl_atr_mult"), 0.8, 4.0, float)
    upd("perps_tp_r_multiple", perps.get("tp_r_multiple"), 1.0, 3.5, float)
    upd("perps_enter_threshold", perps.get("enter_threshold"), 0.05, 0.60, float)
    upd("perps_exit_threshold", perps.get("exit_threshold"), 0.01, 0.30, float)
    upd("perps_pyramids", perps.get("pyramids"), 0, 4, int)

    changed = []
    for k in after:
        if k in before and before[k] != after[k]:
            changed.append((k, before[k], after[k]))

    if not changed:
        return f"{symbol}: no changes suggested. snap={snap}"

    if apply_changes:
        state["bots"][symbol].update(after)
        await push_alert(app, f"🤖 Autotune applied for {symbol}:\n" + "\n".join([f"{k}: {a} → {b}" for k,a,b in changed]))
        return f"{symbol}: applied {len(changed)} changes."
    else:
        await push_alert(app, f"🤖 Autotune (dry‑run) for {symbol}:\n" + "\n".join([f"{k}: {a} → {b}" for k,a,b in changed]))
        return f"{symbol}: suggested {len(changed)} changes (dry‑run)."

async def autotune_job(context: ContextTypes.DEFAULT_TYPE):
    g = state["global"]
    if not g.get("ai_autotune_enabled", False):
        return
    app = context.application
    results = []
    for sym in list(state["bots"].keys()):
        try:
            r = await autotune_once(app, sym, days=g.get("auto_days", 7))
        except Exception as e:
            r = f"{sym}: error {e}"
        results.append(r)
    if results:
        await push_alert(app, "📈 Autotune cycle:\n" + "\n".join(results))

async def autotune_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """ /autotune on|off [hours] [apply:true|false] """
    global autotune_job_handle
    g = state["global"]
    if not ctx.args:
        return await update.message.reply_text(
            f"Autotune: enabled={g.get('ai_autotune_enabled')} "
            f"period={g.get('ai_autotune_hours')}h apply={g.get('ai_autotune_apply')}"
        )
    mode = ctx.args[0].lower()
    if mode in ("on","off"):
        g["ai_autotune_enabled"] = (mode=="on")
    if len(ctx.args) >= 2:
        try: g["ai_autotune_hours"] = int(ctx.args[1])
        except: pass
    if len(ctx.args) >= 3:
        g["ai_autotune_apply"] = (ctx.args[2].lower() in ("true","1","yes","on"))

    # reschedule
    if autotune_job_handle:
        try:
            autotune_job_handle.schedule_removal()
        except Exception:
            pass
        autotune_job_handle = None
    if g["ai_autotune_enabled"]:
        autotune_job_handle = ctx.application.job_queue.run_repeating(
            autotune_job, interval=g["ai_autotune_hours"]*3600, first=5
        )

    await update.message.reply_text(
        f"Autotune set: enabled={g['ai_autotune_enabled']} "
        f"period={g['ai_autotune_hours']}h apply={g['ai_autotune_apply']}"
    )

async def tune_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /tune SYMBOL")
    sym = ctx.args[0].upper()
    r = await autotune_once(ctx.application, sym, days=state["global"].get("auto_days", 7))
    await update.message.reply_text(r)

async def tuneall_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    results=[]
    for sym in list(state["bots"].keys()):
        r = await autotune_once(ctx.application, sym, days=state["global"].get("auto_days", 7))
        results.append(r)
    await update.message.reply_text("TuneAll:\n" + "\n".join(results))

async def paperstatus_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not paper_pos:
        return await update.message.reply_text("No paper positions tracked.")
    lines = ["📄 Active Paper Positions:"]
    for sym, pos in paper_pos.items():
        side = int(pos.get("side", 0))
        if side == 0 or pos.get("size", 0) == 0:
            continue
        side_str = "LONG" if side > 0 else "SHORT"
        entry = pos.get("entry", 0.0)
        size = pos.get("size", 0.0)
        stop = pos.get("stop", 0.0)
        take = pos.get("take", 0.0)
        lines.append(f"{sym}: {side_str} entry={entry:.6f} size={size:.6f} stop={stop:.6f} take={take:.6f}")
    if len(lines) == 1:
        lines.append("(none open)")
    await update.message.reply_text("\n".join(lines))

# ---------- Startup ----------
async def on_startup(app: Application):
    cmds = [BotCommand(c["cmd"], c["desc"]) for c in COMMAND_SPECS]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception:
        pass
    # schedule autotune if enabled
    g = state["global"]
    if g.get("ai_autotune_enabled", False):
        global autotune_job_handle
        autotune_job_handle = app.job_queue.run_repeating(
            autotune_job, interval=g.get("ai_autotune_hours",12)*3600, first=30
        )
    # schedule paper-live if enabled
    if g.get("paper_live_enabled", False):
        global paper_job_handle
        paper_job_handle = app.job_queue.run_repeating(
            paperlive_job, interval=max(5, g.get("paper_poll_seconds", 15)), first=5
        )

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(on_startup).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("list", list_bots))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("set", set_cmd))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("summaryall", summaryall))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CommandHandler("backtestperp", backtestperp))
    app.add_handler(CommandHandler("autotune", autotune_cmd))
    app.add_handler(CommandHandler("tune", tune_cmd))
    app.add_handler(CommandHandler("tuneall", tuneall_cmd))
    app.add_handler(CommandHandler("paperlive", paperlive_cmd))
    app.add_handler(CommandHandler("paperstatus", paperstatus_cmd))
    print("🤖 Bot running... try /help")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
