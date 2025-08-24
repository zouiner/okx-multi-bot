#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, logging, io
from datetime import datetime
from typing import Dict, Any

from telegram import Update, BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from storage import Storage, summarize_symbol, export_trades_csv
from data import fetch_ohlcv, fetch_last_price
from strategy_grid import run_grid_backtest
from strategy_perps import run_perps_backtest

# NEW: Gemini AI
try:
    from agent_gemini import GeminiAgent
except Exception:
    GeminiAgent = None  # handle gracefully if module missing

# NEW: matplotlib for chart
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
    raise FileNotFoundError("⚠️ configs.json not found. Create it from configs.example.json")

TELEGRAM_BOT_TOKEN = CONFIG.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = CONFIG.get("TELEGRAM_CHAT_ID")
GEMINI_KEY = CONFIG.get("GEMINI_API_KEY")
GEMINI_MODEL = CONFIG.get("GEMINI_MODEL", "gemini-2.5-flash")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing in configs.json")

# init AI agent if key provided
ai_agent = GeminiAgent(GEMINI_KEY, GEMINI_MODEL) if (GEMINI_KEY and GeminiAgent) else None

# ---------- Dynamic help registry ----------
COMMAND_SPECS = [
    {"cmd":"start","args":"","desc":"Show welcome message.","example":"/start"},
    {"cmd":"help","args":"[command]","desc":"Show all commands or details for one.","example":"/help add"},
    {"cmd":"ping","args":"","desc":"Health check.","example":"/ping"},
    {"cmd":"add","args":"SYMBOL [interval|auto] [paper|live] [quote_cap]","desc":"Track a coin (paper by default). If interval omitted or 'auto', it picks based on global auto_days.","example":"/add SUI-USDT auto paper 1000"},
    {"cmd":"list","args":"","desc":"Show all tracked coins and basic settings.","example":"/list"},
    {"cmd":"status","args":"[SYMBOL]","desc":"Status for one symbol or all (incl. AI/DCA).","example":"/status SUI-USDT"},
    {"cmd":"set","args":"<SYMBOL|global> <key> <value>","desc":"Tune params. Global: grids, per_order_quote, min_notional, fee_rate, atr_mult, ema_fast_n, ema_slow_n, recenter_threshold, recenter_every, auto_days, ai_enabled, dca_tp_pct, dca_add_drawdown_pct, dca_max_adds, dca_equity_fraction, atr_extreme_mult","example":"/set global dca_tp_pct 0.10"},
    {"cmd":"history","args":"SYMBOL [N]","desc":"Show last N trades for SYMBOL.","example":"/history SUI-USDT 10"},
    {"cmd":"summary","args":"SYMBOL [days]","desc":"Aggregate stats over the window.","example":"/summary SUI-USDT 7"},
    {"cmd":"summaryall","args":"[days]","desc":"Aggregate stats for ALL tracked coins.","example":"/summaryall 7"},
    {"cmd":"export","args":"SYMBOL","desc":"Export all trades for SYMBOL to CSV.","example":"/export SUI-USDT"},
    {"cmd":"backtest","args":"SYMBOL [days]","desc":"Adaptive fee-aware grid backtest + chart. Auto-picks best bar for the span, with AI+DCA overlay.","example":"/backtest SUI-USDT 30"},
    {"cmd":"backtestperp","args":"SYMBOL [days]","desc":"AI+Indicator fused long/short backtest (perps-style).","example":"/backtestperp BTC-USDT 60"},

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
    ("1m", 1440),
    ("3m", 480),
    ("5m", 288),
    ("15m", 96),
    ("30m", 48),
    ("1H", 24),
    ("2H", 12),
    ("4H", 6),
    ("6H", 4),
    ("12H", 2),
    ("1D", 1),
]
def pick_bar_for_days(days: int, target_candles: int = 900) -> str:
    """Return the highest-resolution bar whose total candles <= target_candles."""
    days = max(1, int(days))
    for bar, per_day in BAR_CANDLES_PER_DAY:
        if days * per_day <= target_candles:
            return bar
    return "1D"  # fallback for very long spans

# ---------- Utils ----------
def parse_bool(s: str) -> bool:
    s = (s or "").strip().lower()
    return s in ("1","true","yes","y","on")

# ---------- Global State ----------
state: Dict[str, Any] = {
    "bots": {},
    "global": {
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
        # ---- AI + DCA defaults ----
        "ai_enabled": True,
        "dca_tp_pct": 0.10,             # +10% TP over blended WAP after adds
        "dca_add_drawdown_pct": 0.15,   # add when price <= last_buy * (1-15%)
        "dca_max_adds": 2,              # at most 2 rescue adds
        "dca_equity_fraction": 0.10,    # spend ≤10% of equity per add
        "atr_extreme_mult": 4.0,        # block adds if ATR > 4x median
        # --- Perps knobs (defaults) ---
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

    }
}
storage = Storage("bot_history.db")

# ---------- Optional alerts ----------
async def push_alert(app, text: str):
    if TELEGRAM_CHAT_ID:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

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

    # interval logic: omitted or 'auto' -> pick automatically based on global auto_days
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
        await update.message.reply_text(
            f"{sym} @ {price:.6f} | interval={cfg['interval']} | mode={cfg['mode']}\n"
            f"grids={cfg['grids']} per_order={cfg['per_order_quote']} min_notional={cfg['min_notional']}\n"
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
            lines.append(f"{sym} @ {price:.6f} | {cfg['interval']} | {cfg['mode']} | AI={cfg.get('ai_enabled', True)}")
        await update.message.reply_text("\n".join(lines))

async def set_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 3:
        return await update.message.reply_text("Usage: /set <SYMBOL|global> <key> <value>")
    target, key, val = ctx.args[0], ctx.args[1], " ".join(ctx.args[2:])
    if target.lower() == "global":
        g = state["global"]
        try:
            if key in ("grids","ema_fast_n","ema_slow_n","recenter_every","auto_days","dca_max_adds"):
                g[key] = int(val)
            elif key in ("per_order_quote","min_notional","fee_rate","atr_mult","recenter_threshold","dca_tp_pct","dca_add_drawdown_pct","dca_equity_fraction","atr_extreme_mult"):
                g[key] = float(val)
            elif key == "ai_enabled":
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
    await update.message.reply_text(
        f"Summary {symbol} (last {days}d)\n"
        f"Trades: {s['total_trades']}\n"
        f"Buys:  {buys['count']}  Notional: {buys['notional']:.2f}\n"
        f"Sells: {sells['count']}  Notional: {sells['notional']:.2f}"
    )

async def summaryall(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    days = int(ctx.args[0]) if ctx.args else 7
    if not state["bots"]:
        return await update.message.reply_text("No coins tracked.")
    lines = [f"📊 Summary (last {days}d):"]
    for sym in state["bots"].keys():
        s = summarize_symbol("bot_history.db", sym, days)
        b = s["by_side"]; buys=b.get("buy",{"count":0,"notional":0.0}); sells=b.get("sell",{"count":0,"notional":0.0})
        lines.append(f"{sym}: {s['total_trades']} | Buys {buys['count']}({buys['notional']:.2f}) | Sells {sells['count']}({sells['notional']:.2f})")
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
    Automatically picks the best bar (interval) so that total candles ≲ 900.
    Uses Gemini AI to steer bias/crisis (if enabled) and DCA overlay for rescue adds + 10% TP from WAP.
    """
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /backtest SYMBOL [days]")
    symbol = ctx.args[0].upper()
    days = int(ctx.args[1]) if len(ctx.args) > 1 else 7

    # auto pick interval based on span
    bar = pick_bar_for_days(days)
    await update.message.reply_text(f"⏳ Backtest {symbol} over {days}d ({bar} candles)...")

    try:
        candles = fetch_ohlcv(symbol, bar=bar, days=days)
    except Exception as e:
        return await update.message.reply_text(f"Fetch error: {e}")
    if not candles:
        return await update.message.reply_text("No candles fetched.")

    # AI signal (optional)
    ai_bias = None
    ai_conf = 0.0
    ai_crisis = False
    ai_used = False

    # Pull per-symbol config or fall back to global
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
            if "long" in rec or "buy" in rec:
                ai_bias = "long"
            elif "short" in rec or "sell" in rec:
                ai_bias = "short"
            else:
                ai_bias = "neutral"
            ai_conf = float(sig.get("confidence") or 0.0)
            ai_crisis = bool(sig.get("crisis") or False)
            ai_used = True
        except Exception as e:
            ai_bias, ai_conf, ai_crisis = None, 0.0, False
            ai_used = False

    res = run_grid_backtest(
        symbol,
        candles,
        quote_cap=float(cfg.get("quote_cap", 1000.0)),
        grid_size=int(cfg.get("grids", g["grids"])),
        per_order_quote=float(cfg.get("per_order_quote", g["per_order_quote"])),
        fee_rate=float(g["fee_rate"]),
        atr_mult=float(g["atr_mult"]),
        ema_fast_n=int(g["ema_fast_n"]),
        ema_slow_n=int(g["ema_slow_n"]),
        recenter_threshold=float(g["recenter_threshold"]),
        recenter_every=int(g["recenter_every"]),
        # NEW: AI + DCA overlay
        ai_bias=ai_bias,
        ai_confidence=ai_conf,
        ai_crisis=ai_crisis,
        dca_tp_pct=dca_tp_pct,
        dca_add_drawdown_pct=dca_add_drawdown_pct,
        dca_max_adds=dca_max_adds,
        dca_equity_fraction=dca_equity_fraction,
        atr_extreme_mult=atr_extreme_mult,
    )

    # Text summary
    ai_line = ""
    if ai_used:
        ai_line = f"\nAI: bias={ai_bias} conf={ai_conf:.2f} crisis={ai_crisis}"
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

    # PNG equity curve
    eq = res.get("equity_curve", [])
    if eq:
        xs_ms = [t for t, _ in eq]
        ys = [v for _, v in eq]
        xs = [datetime.utcfromtimestamp(t/1000.0) for t in xs_ms]  # ms -> dt

        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)  # no explicit colors/styles
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

    # Read params (symbol override -> global)
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

    # Optional equity curve PNG (reuse your matplotlib snippet)
    try:
        import matplotlib.pyplot as plt, io
        xs = [c[0] for c in res["equity_curve"]]
        ys = [c[1] for c in res["equity_curve"]]
        from datetime import datetime
        xs_dt = [datetime.utcfromtimestamp(t/1000.0) for t in xs]
        import matplotlib
        matplotlib.use("Agg")
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


# ---------- Startup ----------
async def on_startup(app: Application):
    cmds = [BotCommand(c["cmd"], c["desc"]) for c in COMMAND_SPECS]
    try:
        await app.bot.set_my_commands(cmds)
    except Exception:
        pass

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
    print("🤖 Bot running... try /help")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
