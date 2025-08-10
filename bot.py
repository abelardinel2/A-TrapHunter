#!/usr/bin/env python3
# TopTrapHunterBot (fixed event loop)
# PTB v20+, python 3.10+

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ------------------------- CONFIG -------------------------

SYMBOLS_DEFAULT = ["BTC/USDT", "XRP/USDT", "XLM/USDT", "HBAR/USDT", "SUI/USDT", "LINK/USDT"]
ASIA_OPEN_ET = "20:00"   # 8pm ET Sun–Thu
ASIA_CLOSE_ET = "04:00"  # 4am ET Mon–Fri
MIN_RR = 2.0             # minimum risk:reward to alert

CHAT_ID_FILE = "tth_chat_id.txt"
STATE_FILE = "tth_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("TopTrapHunter")

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise SystemExit("Missing TELEGRAM_BOT_TOKEN in environment (or .env).")

# ------------------------- STATE --------------------------

@dataclass
class BotState:
    armed: bool = False
    symbols: List[str] = field(default_factory=lambda: SYMBOLS_DEFAULT.copy())
    asia_open_et: str = ASIA_OPEN_ET
    asia_close_et: str = ASIA_CLOSE_ET
    min_rr: float = MIN_RR
    last_alert_ts: Dict[str, float] = field(default_factory=dict)

STATE = BotState()

def _load_chat_id() -> Optional[int]:
    try:
        if os.path.exists(CHAT_ID_FILE):
            return int(open(CHAT_ID_FILE, "r").read().strip())
    except Exception:
        pass
    return None

def _save_chat_id(chat_id: int) -> None:
    with open(CHAT_ID_FILE, "w") as f:
        f.write(str(chat_id))

def _load_state() -> None:
    if not os.path.exists(STATE_FILE):
        return
    try:
        data = json.load(open(STATE_FILE, "r"))
        STATE.armed = bool(data.get("armed", False))
        STATE.symbols = list(data.get("symbols", SYMBOLS_DEFAULT))
        STATE.asia_open_et = data.get("asia_open_et", ASIA_OPEN_ET)
        STATE.asia_close_et = data.get("asia_close_et", ASIA_CLOSE_ET)
        STATE.min_rr = float(data.get("min_rr", MIN_RR))
        STATE.last_alert_ts = dict(data.get("last_alert_ts", {}))
    except Exception as e:
        log.warning("Failed loading state: %s", e)

def _save_state() -> None:
    data = {
        "armed": STATE.armed,
        "symbols": STATE.symbols,
        "asia_open_et": STATE.asia_open_et,
        "asia_close_et": STATE.asia_close_et,
        "min_rr": STATE.min_rr,
        "last_alert_ts": STATE.last_alert_ts,
    }
    json.dump(data, open(STATE_FILE, "w"))

CHAT_ID = _load_chat_id()
_load_state()

# ------------------------- EXCHANGE -----------------------

exchange = ccxt.binance({
    "enableRateLimit": True,
})

async def fetch_ohlcv(symbol: str, timeframe="15m", limit=200) -> pd.DataFrame:
    data = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ------------------------- SIGNALS (toy “trap” logic) -----

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def green_light_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    Very lightweight “trap” placeholder:
    - Price pushes above a prior local high on falling volume (potential fake-out),
    - RSI/EMA divergence-ish check,
    - Returns short setup with example RR if found.
    """
    if len(df) < 80:
        return None
    close = df["close"]
    vol = df["vol"]

    # prior swing high
    swing_high = close.rolling(50).max().shift(1)
    broke_high = (close.iloc[-2] <= swing_high.iloc[-2]) and (close.iloc[-1] > swing_high.iloc[-1])

    falling_vol = vol.iloc[-1] < vol.iloc[-10: -1].mean()

    ema_fast = ema(close, 8)
    ema_slow = ema(close, 21)
    loss_momentum = ema_fast.diff().iloc[-1] < 0 and ema_fast.iloc[-1] > ema_slow.iloc[-1]

    if broke_high and falling_vol and loss_momentum:
        entry = float(close.iloc[-1])
        stop  = float(df["high"].iloc[-5: ].max()) * 1.004  # a tiny buffer
        target = entry - (stop - entry) * STATE.min_rr
        rr = (entry - target) / (stop - entry) if stop > entry else None
        if rr and rr >= STATE.min_rr:
            return {"side": "short", "entry": entry, "stop": stop, "target": target, "rr": rr}
    return None

# ------------------------- TELEGRAM HANDLERS --------------

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global CHAT_ID
    CHAT_ID = update.effective_chat.id
    _save_chat_id(CHAT_ID)
    await update.message.reply_text(
        "TopTrapHunter armed assistant.\n"
        "Use /arm to start scanning, /disarm to stop.\n"
        "Use /status to view config, /setsymbols, /setrr, /test."
    )

async def cmd_arm(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.armed = True
    _save_state()
    await update.message.reply_text("✅ Armed. Scanning…")

async def cmd_disarm(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.armed = False
    _save_state()
    await update.message.reply_text("⛔️ Disarmed.")

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Armed: {STATE.armed}\n"
        f"Symbols: {', '.join(STATE.symbols)}\n"
        f"Asia window (ET): {STATE.asia_open_et} → {STATE.asia_close_et}\n"
        f"Min RR: {STATE.min_rr}"
    )

async def cmd_setsymbols(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: /setsymbols BTC/USDT,ETH/USDT,…")
        return
    raw = " ".join(ctx.args)
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if symbols:
        STATE.symbols = symbols
        _save_state()
        await update.message.reply_text(f"Symbols updated: {', '.join(symbols)}")
    else:
        await update.message.reply_text("Could not parse symbols.")

async def cmd_setrr(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: /setrr 2.0")
        return
    try:
        STATE.min_rr = float(ctx.args[0])
        _save_state()
        await update.message.reply_text(f"Min RR set to {STATE.min_rr}")
    except Exception:
        await update.message.reply_text("Invalid number.")

async def cmd_test(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Bot alive ✅")

# ------------------------- SCANNER LOOP -------------------

async def notify(app: Application, text: str) -> None:
    chat_id = CHAT_ID
    if chat_id:
        try:
            await app.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            log.warning("Notify failed: %s", e)

def _within_asia_window(now_utc: pd.Timestamp) -> bool:
    # Simple check: always scan; you can add tz-aware ET window if you like.
    return True

async def scanner_loop(app: Application) -> None:
    log.info("Scanner loop started.")
    while True:
        try:
            if STATE.armed and _within_asia_window(pd.Timestamp.utcnow()):
                for sym in STATE.symbols:
                    try:
                        df = await fetch_ohlcv(sym, timeframe="15m", limit=200)
                        sig = green_light_signal(df)
                        if sig:
                            msg = (
                                f"⚠️ Possible TRAP on {sym}\n"
                                f"Side: {sig['side']}  RR≈{sig['rr']:.2f}\n"
                                f"Entry: {sig['entry']:.4f}\n"
                                f"Stop:  {sig['stop']:.4f}\n"
                                f"Target:{sig['target']:.4f}\n"
                                f"(15m, vol fade + HL fake-out prototype)\n"
                            )
                            await notify(app, msg)
                        await asyncio.sleep(0.3)  # rate limit
                    except Exception as e:
                        log.warning("Scan error %s: %s", sym, e)
                        await asyncio.sleep(0.2)
            await asyncio.sleep(20)  # main cadence
        except asyncio.CancelledError:
            log.info("Scanner loop cancelled.")
            break
        except Exception as e:
            log.exception("Scanner loop exception: %s", e)
            await asyncio.sleep(5)

# ------------------------- APP WIRES (fixed post_init) ----

async def post_init(app: Application) -> None:
    """
    PTB v20+ requires post_init to be async.
    Use app.create_task() so it runs on the bot's loop.
    """
    app.create_task(scanner_loop(app))

def build_app() -> Application:
    application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("arm", cmd_arm))
    application.add_handler(CommandHandler("disarm", cmd_disarm))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("setsymbols", cmd_setsymbols))
    application.add_handler(CommandHandler("setrr", cmd_setrr))
    application.add_handler(CommandHandler("test", cmd_test))

    application.post_init = post_init  # ← important

    return application

def main() -> None:
    app = build_app()
    # run_polling creates & manages the event loop — do NOT create your own loop.
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()