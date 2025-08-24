# Crypto Bot — Grid (Spot) + AI‑Fused Perps (Long/Short)

An OKX‑ready Telegram bot that runs:

- **Adaptive Grid (spot)** with ATR band, EMA trend gate, re‑centering, fee‑aware spacing, optional **DCA “10% rescue”** overlay.
- **AI‑Fused Perps Backtester** (true long/short logic) that blends **Gemini AI** bias with robust indicators (EMA, RSI, MACD, VWAP, Bollinger) + ATR risk.

It supports **auto timeframe selection** (1m → 1D) so backtests don’t overload on candles, and exposes nearly all parameters via `/set`.

---

## Features

- **Spot Grid (paper/live ready):**
  - ATR‑scaled band centered on EMA(20/60).
  - Trend gate blocks buys in strong downtrends / sells in strong uptrends.
  - Re‑centers over time or on drift.
  - **DCA overlay**: add ≤10% equity on drawdowns, then require TP at **WAP × (1+10%)**.
- **AI (Gemini) integration:**
  - Directional **bias** (long/short/neutral), **confidence**, and **crisis** flag.
  - Crisis freezes entries; bias nudges buy/sell permissions.
- **Perps backtester (long/short):**
  - Indicator fusion score in [-1,1] (EMA, RSI, MACD, VWAP, Bollinger).
  - ATR‑based position sizing, SL/TP, optional ATR trailing, pyramids.
- **Telegram UX:**
  - `/add`, `/list`, `/status`, `/set`, `/backtest`, `/backtestperp`, `/history`, `/summary`, `/summaryall`, `/export`.
  - Equity curve PNGs sent directly to chat.
- **Auto timeframe** for backtests via `pick_bar_for_days(days)`.

---

## Repo Layout

```
.
├── okx_grid_multicoin_bot.py       # Telegram bot (grid + AI/DCA + perps backtests)
├── strategy_grid.py                 # Spot grid backtester (AI/DCA aware)
├── strategy_perps.py               # NEW: AI+indicator fused long/short backtester
├── data.py                          # OKX public market data (ticker/ohlcv)
├── storage.py                       # SQLite trade logs + summaries
├── agent_gemini.py                  # Gemini wrapper (JSON parsing)
├── requirements.txt                 # python-telegram-bot, google-generativeai, matplotlib
└── configs.json                     # your keys & tokens (see below)
```

> Note: The code expects `configs.json` (plural). If your file is `config.json`, rename it.

---

## Requirements

- Python 3.9+
- Telegram Bot token
- (Optional) Gemini API key (for AI overlay)
- (Optional) OKX API keys (only needed if you later enable live trading)

Install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:

```
python-telegram-bot==20.7
google-generativeai
matplotlib
```

---

## Configuration (`configs.json`)

```json
{
  "OKX_API_KEY": "xxx",
  "OKX_API_SECRET": "xxx",
  "OKX_API_PASSPHRASE": "xxx",
  "TELEGRAM_TOKEN": "123456:abc-your-telegram-bot-token",
  "TELEGRAM_CHAT_ID": "123456789",
  "GEMINI_API_KEY": "your-gemini-key",
  "GEMINI_MODEL": "gemini-2.5-flash"
}
```

- **OKX keys** are optional for backtests & paper. Keep them for future live trading.
- **GEMINI_*** optional; without it, AI overlay is skipped.
- **TELEGRAM_CHAT_ID** optional; if set, bot can push alerts.

---

## Run the bot

```bash
python okx_grid_multicoin_bot.py
```

You should see: `🤖 Bot running... try /help`

Open Telegram and talk to your bot.

---

## Quick Start (Telegram)

### Add a coin in spot grid (paper mode):

```
/add SUI-USDT auto paper 1000
/list
/status SUI-USDT
```

### Backtest spot grid with AI+DCA overlay:

```
/backtest SUI-USDT 30
```

### Backtest perps long/short with AI+indicators:

```
/backtestperp BTC-USDT 60
```

---

## Tuning parameters

Examples:

```
/set global fee_rate 0.0008
/set global auto_days 30
/set global dca_tp_pct 0.1
/set global perps_tp_r_multiple 2.0
```

Per‑symbol overrides:

```
/set BTC-USDT perps_risk_per_trade_pct 0.0075
```

---

## Backtest Only (no Telegram)

You can run directly in Python REPL:

```python
from data import fetch_ohlcv
from strategy_perps import run_perps_backtest

candles = fetch_ohlcv("BTC-USDT", bar="1H", days=30)
res = run_perps_backtest("BTC-USDT", candles, ai_bias="long", ai_confidence=0.8)
print(res["net_pnl"], res["roi_pct"])
```

---

## Safety

- Always test in **paper** mode before enabling live.
- In live perps, use small allocations and keep leverage reasonable.
- Gemini AI is advisory — all signals are combined with hard risk controls (ATR stops, max drawdown guard). 
