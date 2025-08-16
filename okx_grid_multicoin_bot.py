import os, json, time, logging, asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from storage import Storage
from agent_gemini import GeminiAgent

logging.basicConfig(level=logging.INFO)

# --- Load configs.json ---
CONFIG = {}
if os.path.exists("configs.json"):
    with open("configs.json") as f:
        CONFIG = json.load(f)
else:
    raise FileNotFoundError("⚠️ configs.json not found. Please create it from configs.example.json")

TELEGRAM_BOT_TOKEN = CONFIG.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = CONFIG.get("TELEGRAM_CHAT_ID")
OKX_API_KEY = CONFIG.get("OKX_API_KEY")
OKX_API_SECRET = CONFIG.get("OKX_API_SECRET")
OKX_API_PASSPHRASE = CONFIG.get("OKX_API_PASSPHRASE")
GEMINI_API_KEY = CONFIG.get("GEMINI_API_KEY")
GEMINI_MODEL = CONFIG.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Global State ---
state = {
    "bots": {},  # symbol -> config
    "global": {
        "ai": "gemini",
        "crisis_drop_pct": 5,
        "crisis_rise_pct": 7,
        "crisis_window_min": 5,
        "gemini_model": GEMINI_MODEL,
    }
}

storage = Storage("bot_history.db")
ai_agent = GeminiAgent(GEMINI_API_KEY, model=GEMINI_MODEL)

# --- Telegram Commands ---
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Multi-coin bot running. Use /add SYMBOL to begin.")

async def add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /add SYMBOL [interval] [paper|live] [quote_cap]")
    symbol = ctx.args[0].upper()
    state["bots"][symbol] = {
        "interval": ctx.args[1] if len(ctx.args) > 1 else "1m",
        "mode": ctx.args[2] if len(ctx.args) > 2 else "paper",
        "quote_cap": float(ctx.args[3]) if len(ctx.args) > 3 else 1000,
    }
    await update.message.reply_text(f"➕ Added {symbol} with {state['bots'][symbol]}")

async def list_bots(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not state["bots"]:
        return await update.message.reply_text("No coins tracked. Use /add SYMBOL")
    msg = "\n".join([f"{k}: {v}" for k,v in state["bots"].items()])
    await update.message.reply_text(f"📊 Tracking:\n{msg}")

async def history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 1:
        return await update.message.reply_text("Usage: /history SYMBOL [N]")
    symbol = ctx.args[0].upper()
    n = int(ctx.args[1]) if len(ctx.args) > 1 else 10
    rows = storage.get_trades(symbol, n)
    if not rows:
        return await update.message.reply_text(f"No trades for {symbol}")
    msg = "\n".join([f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in rows])
    await update.message.reply_text(f"📜 History {symbol}:\n{msg}")

# Crisis alert push (auto to TELEGRAM_CHAT_ID)
async def push_alert(app, msg: str):
    if TELEGRAM_CHAT_ID:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

# --- Main ---
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("list", list_bots))
    app.add_handler(CommandHandler("history", history))
    print("🤖 Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
