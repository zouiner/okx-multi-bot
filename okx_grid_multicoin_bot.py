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

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN missing in configs.json")

# ---------- Dynamic help registry ----------
COMMAND_SPECS = [
    {"cmd":"start","args":"","desc":"Show welcome message.","example":"/start"},
    {"cmd":"help","args":"[command]","desc":"Show all commands or details for one.","example":"/help add"},
    {"cmd":"ping","args":"","desc":"Health check.","example":"/ping"},
    {"cmd":"add","args":"SYMBOL [interval] [paper|live] [quote_cap]","desc":"Track a coin (paper by default).","example":"/add SUI-USDT 1m paper 1000"},
    {"cmd":"list","args":"","desc":"Show all tracked coins and basic settings.","example":"/list"},
    {"cmd":"status","args":"[SYMBOL]","desc":"Status for one symbol or all.","example":"/status SUI-USDT"},
    {"cmd":"set","args":"<SYMBOL|global> <key> <value>","desc":"Tune params. Global: grids, per_order_quote, min_notional, fee_rate, atr_mult, ema_fast_n, ema_slow_n, recenter_threshold, recenter_every","example":"/set global fee_rate 0.0008"},
    {"cmd":"history","args":"SYMBOL [N]","desc":"Show last N trades for SYMBOL.","example":"/history SUI-USDT 10"},
    {"cmd":"summary","args":"SYMBOL [days]","desc":"Aggregate stats over the window.","example":"/summary SUI-USDT 7"},
    {"cmd":"summaryall","args":"[days]","desc":"Aggregate stats for ALL tracked coins.","example":"/summaryall 7"},
    {"cmd":"export","args":"SYMBOL","desc":"Export all trades for SYMBOL to CSV.","example":"/export SUI-USDT"},
    {"cmd":"backtest","args":"SYMBOL [days]","desc":"Adaptive fee-aware grid backtest + chart.","example":"/backtest SUI-USDT 30"},
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
    }
}
storage = Storage("bot_history.db")

# ---------- Optional alerts ----------
async def push_alert(app, text: str):
    if TELEGRAM_CHAT_ID:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

# ---------- Handlers ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Multi-coin bot ready. Try `/help`.\nAdd a coin: `/add SUI-USDT 1m paper 1000`")

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    arg = ctx.args[0].lower() if ctx.args else None
    await update.message.reply_text(build_help_text(arg), parse_mode=ParseMode.MARKDOWN)

async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

async def add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /add SYMBOL [interval] [paper|live] [quote_cap]")
    sym = ctx.args[0].upper()
    interval = ctx.args[1] if len(ctx.args) > 1 else "1m"
    mode = (ctx.args[2].lower() if len(ctx.args) > 2 else "paper")
    quote_cap = float(ctx.args[3]) if len(ctx.args) > 3 else 1000.0
    state["bots"][sym] = {
        "interval": interval, "mode": mode, "quote_cap": quote_cap,
        "grids": state["global"]["grids"],
        "per_order_quote": state["global"]["per_order_quote"],
        "min_notional": state["global"]["min_notional"],
        "added_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    await update.message.reply_text(f"➕ Added {sym} {state['bots'][sym]}")

async def list_bots(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not state["bots"]:
        return await update.message.reply_text("No coins tracked. Use `/add SYMBOL`")
    lines = ["📊 Tracking:"] + [f"- {k}: {v}" for k, v in state["bots"].items()]
    await update.message.reply_text("\n".join(lines))

async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        sym = ctx.args[0].upper()
        if sym not in state["bots"]:
            return await update.message.reply_text("Unknown symbol.")
        try: price = fetch_last_price(sym) or 0.0
        except Exception: price = 0.0
        cfg = state["bots"][sym]
        await update.message.reply_text(
            f"{sym} @ {price:.6f} | interval={cfg['interval']} | mode={cfg['mode']}\n"
            f"grids={cfg['grids']} per_order={cfg['per_order_quote']} min_notional={cfg['min_notional']}"
        )
    else:
        if not state["bots"]: return await update.message.reply_text("No coins tracked.")
        lines=[]
        for sym,cfg in state["bots"].items():
            try: price = fetch_last_price(sym) or 0.0
            except Exception: price = 0.0
            lines.append(f"{sym} @ {price:.6f} | {cfg['interval']} | {cfg['mode']}")
        await update.message.reply_text("\n".join(lines))

async def set_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 3:
        return await update.message.reply_text("Usage: /set <SYMBOL|global> <key> <value>")
    target, key, val = ctx.args[0], ctx.args[1], " ".join(ctx.args[2:])
    if target.lower() == "global":
        g = state["global"]
        try:
            if key in ("grids","ema_fast_n","ema_slow_n","recenter_every"): g[key] = int(val)
            elif key in ("per_order_quote","min_notional","fee_rate","atr_mult","recenter_threshold"): g[key] = float(val)
            else: return await update.message.reply_text("Unknown global key.")
            return await update.message.reply_text(f"Updated global {key} = {g[key]}")
        except Exception as e:
            return await update.message.reply_text(f"Error: {e}")
    # per-symbol
    sym = target.upper()
    if sym not in state["bots"]:
        return await update.message.reply_text("Unknown symbol.")
    try:
        if key == "interval": state["bots"][sym]["interval"] = val
        elif key == "mode": state["bots"][sym]["mode"] = val.lower()
        elif key == "quote_cap": state["bots"][sym]["quote_cap"] = float(val)
        elif key == "grids": state["bots"][sym]["grids"] = int(val)
        elif key == "per_order_quote": state["bots"][sym]["per_order_quote"] = float(val)
        elif key == "min_notional": state["bots"][sym]["min_notional"] = float(val)
        else: return await update.message.reply_text("Unknown key.")
        await update.message.reply_text(f"Updated {sym} {key} = {state['bots'][sym][key]}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /history SYMBOL [N]")
    symbol = ctx.args[0].upper()
    n = int(ctx.args[1]) if len(ctx.args) > 1 else 10
    rows = storage.get_trades(symbol, n)
    if not rows: return await update.message.reply_text(f"No trades for {symbol}")
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
    if not state["bots"]: return await update.message.reply_text("No coins tracked.")
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
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /backtest SYMBOL [days]")
    symbol = ctx.args[0].upper()
    days = int(ctx.args[1]) if len(ctx.args) > 1 else 7
    await update.message.reply_text(f"⏳ Backtest {symbol} over {days}d (1m candles)...")
    try:
        candles = fetch_ohlcv(symbol, bar="1m", days=days)
    except Exception as e:
        return await update.message.reply_text(f"Fetch error: {e}")
    if not candles:
        return await update.message.reply_text("No candles fetched.")

    g = state["global"]
    cfg = state["bots"].get(symbol, {})
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
    )

    # Text summary
    msg = (
        f"📊 Backtest {symbol} ({days}d)\n"
        f"Trades: {res['trades']}\n"
        f"Gross PnL: {res['gross_pnl']:.2f}\n"
        f"Fees Paid: {res['fees_paid']:.2f}\n"
        f"Net PnL: {res['net_pnl']:.2f}\n"
        f"Net ROI: {res['roi_pct']:.2f}%\n"
        f"Max Drawdown: {res['max_drawdown_pct']:.2f}%\n"
        f"Avg Gross/Trade: {res.get('avg_gross_per_trade',0):.4f}\n"
        f"Avg Fee/Trade:   {res.get('avg_fee_per_trade',0):.4f}\n"
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
        # convert ms -> datetime (seconds)
        xs = [datetime.utcfromtimestamp(t/1000.0) for t in xs_ms]

        fig = plt.figure(figsize=(8, 3))     # single clean plot
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)                      # no explicit colors/styles
        ax.set_title(f"Equity Curve — {symbol}")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Equity")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=buf, caption=f"Equity Curve: {symbol} ({days}d)")

# ---------- Startup ----------
async def on_startup(app: Application):
    cmds = [BotCommand(c["cmd"], c["desc"]) for c in COMMAND_SPECS]
    try: await app.bot.set_my_commands(cmds)
    except Exception: pass

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
    print("🤖 Bot running... try /help")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
