# TopTrapHunterBot

A Telegram bot that scans **BTC, XRP, XLM, HBAR, SUI, LINK** for **A+ trap setups** (both long & short) based on your Green Light rules (programmatic version) and sends you an alert **only when** the setup is confirmed.

## What it checks
- **Trap event** on the **15m** (last closed candle):
  - Sweep of prior swing high (or low) by ~0.25%
  - Close back inside the range (reclaim/reject)
  - Volume spike ≥ 1.25× recent average
- **Confluence** on **15m + 1h**:
  - EMA9 < EMA21 for shorts (above for longs)
  - RSI < 50 for shorts ( > 50 for longs)
  - MACD histogram negative for shorts (positive for longs)
- **R:R sanity** (approx) ≥ 2.0×
- **Asia window (ET)** default: `20:00–02:00`

> All checks use **last CLOSED candles** to avoid repaint/algo noise.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install python-telegram-bot==20.6 ccxt pandas numpy pytz python-dotenv
```

## Configure
Create a `.env` in the same folder:
```
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
```

> **Security tip:** Regenerate your token anytime it has been shared publicly.

## Run
```bash
python top_trap_hunter_bot.py
```

In Telegram:
1. DM your bot and send `/start` to register your chat.
2. Send `/arm` to begin scanning.
3. Use `/status` to see active settings.

## Commands
- `/arm` – start scanning
- `/disarm` – pause scanning
- `/status` – show settings
- `/setwindow 20:00 02:00` – set Asia window (ET)
- `/setsymbols BTC,XRP,XLM,HBAR,SUI,LINK` – choose symbols
- `/setrr 2.0` – set minimum R:R

## Notes
- Uses **Binance spot OHLCV** via `ccxt` for robust data; futures funding integration can be added later.
- This is a **helper** for your A+ system — always confirm on your charts before entering.
- You can change parameters near the top of the script to match your exact rules.
