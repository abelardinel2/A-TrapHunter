#!/usr/bin/env python3
"""
TopTrapHunterBot
----------------
Telegram bot that scans BTC, XRP, XLM, HBAR, SUI, LINK for A+ trap setups (both long & short)
based on your Green Light rules (simplified, programmatic version) and sends alerts to Telegram.

Requirements:
- Python 3.10+
- pip install python-telegram-bot==20.6 ccxt pandas numpy pytz python-dotenv

Usage:
- Put your TELEGRAM_BOT_TOKEN in a .env file (recommended) or set env var TELEGRAM_BOT_TOKEN.
- Start the bot: python top_trap_hunter_bot.py
- In Telegram, DM your bot and send /start to register your chat ID.
- Use /arm to start scanning, /disarm to stop.
- Use /status to see current settings.

DISCLAIMER: This is a best-effort automation of your "A+ + Trap + Market" logic.
It is a tool to assist; always confirm on your charts.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# --------------------------- CONFIG ---------------------------

SYMBOLS_DEFAULT = ["BTC/USDT", "XRP/USDT", "XLM/USDT", "HBAR/USDT", "SUI/USDT", "LINK/USDT"]
EXCHANGE_ID = "binance"  # spot for OHLCV; many endpoints require API keys for futures; trap logic works on spot OHLCV
TIMEFRAMES = ["15m", "1h"]  # lower TF for trap, higher TF for structure check
CANDLES = 200               # candles to fetch per timeframe

# Asia session window (ET): default 20:00 - 02:00 (8 PM - 2 AM ET). You can change via /setwindow 20:00 02:00
ASIA_OPEN_ET = "20:00"
ASIA_CLOSE_ET = "02:00"

# Trap detection parameters
SWEEP_EPS = 0.0025    # 0.25% overshoot to count as a sweep beyond prior swing
VOL_SPIKE = 1.25      # rejection candle volume must be >= 1.25x avg(20)
EMA_FAST = 9
EMA_SLOW = 21

# RR sanity check (approx) – we don't place orders, but we estimate R:R vs recent range
MIN_RR = 2.0

# -------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TopTrapHunterBot")

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

CHAT_ID_FILE = "tth_chat_id.txt"
STATE_FILE = "tth_state.json"

@dataclass
class BotState:
    armed: bool = False
    symbols: List[str] = field(default_factory=lambda: SYMBOLS_DEFAULT.copy())
    asia_open_et: str = ASIA_OPEN_ET
    asia_close_et: str = ASIA_CLOSE_ET
    min_rr: float = MIN_RR
    last_alert_ts: Dict[str, float] = field(default_factory=dict)  # rate-limit per symbol

STATE = BotState()

# ---------------------- Helper Functions ----------------------

def load_chat_id() -> Optional[int]:
    if os.path.exists(CHAT_ID_FILE):
        try:
            with open(CHAT_ID_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None
    return None

def save_chat_id(chat_id: int) -> None:
    with open(CHAT_ID_FILE, "w") as f:
        f.write(str(chat_id))

def parse_time_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def is_within_asia_window(now_utc: float, open_et: str, close_et: str) -> bool:
    # Convert UTC timestamp to ET and check if between open and close (may cross midnight)
    et = pytz.timezone("America/New_York")
    dt = pd.to_datetime(now_utc, unit="s", utc=True).tz_convert(et)
    oh, om = parse_time_hhmm(open_et)
    ch, cm = parse_time_hhmm(close_et)
    open_dt = dt.replace(hour=oh, minute=om, second=0, microsecond=0)
    close_dt = dt.replace(hour=ch, minute=cm, second=0, microsecond=0)

    if open_et == close_et:
        return True  # full day
    if open_dt <= close_dt:
        return open_dt <= dt <= close_dt
    else:
        # crosses midnight
        return dt >= open_dt or dt <= close_dt

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def detect_sweep_trap(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detects both bull and bear traps on the last *closed* candle (index -2),
    using prior swing highs/lows and volume spike + close-back-inside criteria.
    Returns dict flags: {"short_trap": bool, "long_trap": bool}
    """
    if len(df) < 60:
        return {"short_trap": False, "long_trap": False}

    # Use last completed candle
    i = -2
    last_high = df["high"].iloc[i]
    last_low = df["low"].iloc[i]
    last_close = df["close"].iloc[i]
    last_open = df["open"].iloc[i]
    last_vol = df["volume"].iloc[i]

    # Prior swing high/low (exclude last 5 bars)
    window = df.iloc[:-5]
    swing_high = window["high"].max()
    swing_low = window["low"].min()

    avg_vol = window["volume"].tail(20).mean()

    # Short trap condition (bearish setup): took out highs, then closed back below swing high with volume
    swept_high = last_high > swing_high * (1 + SWEEP_EPS)
    rejected_high = last_close < swing_high
    vol_ok = last_vol >= avg_vol * VOL_SPIKE

    short_trap = bool(swept_high and rejected_high and vol_ok)

    # Long trap condition (bullish setup): took out lows, then closed back above swing low with volume
    swept_low = last_low < swing_low * (1 - SWEEP_EPS)
    reclaimed_low = last_close > swing_low
    long_trap = bool(swept_low and reclaimed_low and vol_ok)

    return {"short_trap": short_trap, "long_trap": long_trap}

def confluence_checks(df15: pd.DataFrame, df1h: pd.DataFrame) -> Dict[str, bool]:
    """
    Additional indicator confluence: EMA rejection + RSI/MACD confirmation on 15m and 1h
    """
    # Use last closed candle (-2) to avoid repaint
    i = -2
    close15 = df15["close"].iloc[i]
    vol15 = df15["volume"].iloc[i]
    ema9_15 = ema(df15["close"], EMA_FAST).iloc[i]
    ema21_15 = ema(df15["close"], EMA_SLOW).iloc[i]
    rsi15 = rsi(df15["close"]).iloc[i]
    _, _, hist15 = macd(df15["close"])

    close1h = df1h["close"].iloc[i]
    ema9_1h = ema(df1h["close"], EMA_FAST).iloc[i]
    ema21_1h = ema(df1h["close"], EMA_SLOW).iloc[i]
    rsi1h = rsi(df1h["close"]).iloc[i]
    _, _, hist1h = macd(df1h["close"])

    # Short-side confluence: price below EMAs + RSI < 50 + MACD hist negative on both TFs
    short_conf = (close15 < ema9_15 < ema21_15) and (close1h < ema9_1h < ema21_1h) and (rsi15 < 50) and (rsi1h < 50) and (hist15.iloc[i] < 0) and (hist1h.iloc[i] < 0)

    # Long-side confluence: price above EMAs + RSI > 50 + MACD hist positive on both TFs
    long_conf = (close15 > ema9_15 > ema21_15) and (close1h > ema9_1h > ema21_1h) and (rsi15 > 50) and (rsi1h > 50) and (hist15.iloc[i] > 0) and (hist1h.iloc[i] > 0)

    return {"short_conf": short_conf, "long_conf": long_conf, "vol15": vol15}

def approx_rr(df: pd.DataFrame, side: str) -> float:
    """
    Approximate R:R by comparing recent ATR-like move vs likely stop distance.
    For simplicity: stop distance = 0.6 * recent range(20), target distance = 1.2 * recent range(20).
    """
    rng = (df["high"].tail(20).max() - df["low"].tail(20).min())
    if rng <= 0:
        return 0.0
    stop = 0.6 * rng
    target = 1.2 * rng
    rr = (target / stop) if stop > 0 else 0.0
    return rr

def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({
        "enableRateLimit": True,
        "timeout": 20000,
    })
    return ex

async def fetch_ohlcv(ex, symbol: str, timeframe: str, limit: int = CANDLES) -> pd.DataFrame:
    o = await asyncio.get_event_loop().run_in_executor(None, ex.fetch_ohlcv, symbol, timeframe, None, limit)
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

async def scan_symbol(symbol: str) -> Optional[str]:
    """
    Returns alert text if an A+ trap (long or short) is detected with confluence and RR.
    """
    try:
        ex = get_exchange()
        df15 = await fetch_ohlcv(ex, symbol, "15m")
        df1h = await fetch_ohlcv(ex, symbol, "1h")

        traps = detect_sweep_trap(df15)
        conf = confluence_checks(df15, df1h)

        # Decide side
        alerts = []
        if traps["short_trap"] and conf["short_conf"]:
            rr = approx_rr(df15, "short")
            if rr >= STATE.min_rr:
                price = df15["close"].iloc[-2]
                swing_high = df15.iloc[:-5]["high"].max()
                entry_zone = f"{price*0.995:.4f} – {price*0.998:.4f}"
                sl = f"{swing_high*1.01:.4f}"
                tp1 = f"{price*0.97:.4f}"
                tp2 = f"{price*0.94:.4f}"
                alerts.append(
                    f"🚨 A+ SHORT READY: {symbol.replace('/','')}\n"
                    f"Entry: {entry_zone}\nSL: {sl}\nTP1: {tp1} | TP2: {tp2}\n"
                    f"Confluence: trap (sweep+reclaim), EMA rejection, RSI<50, MACD<0\n"
                    f"RR (approx): {rr:.1f}x"
                )

        if traps["long_trap"] and conf["long_conf"]:
            rr = approx_rr(df15, "long")
            if rr >= STATE.min_rr:
                price = df15["close"].iloc[-2]
                swing_low = df15.iloc[:-5]["low"].min()
                entry_zone = f"{price*1.002:.4f} – {price*1.005:.4f}"
                sl = f"{swing_low*0.99:.4f}"
                tp1 = f"{price*1.03:.4f}"
                tp2 = f"{price*1.06:.4f}"
                alerts.append(
                    f"🚀 A+ LONG READY: {symbol.replace('/','')}\n"
                    f"Entry: {entry_zone}\nSL: {sl}\nTP1: {tp1} | TP2: {tp2}\n"
                    f"Confluence: trap (flush+reclaim), EMA support, RSI>50, MACD>0\n"
                    f"RR (approx): {rr:.1f}x"
                )

        return "\n\n".join(alerts) if alerts else None
    except Exception as e:
        logger.exception(f"scan_symbol error for {symbol}: {e}")
        return None

async def scanner_loop(app: Application):
    chat_id = load_chat_id()
    if not chat_id:
        logger.info("No chat ID yet. Send /start to the bot first.")
    while True:
        try:
            if STATE.armed and chat_id:
                now = time.time()
                if is_within_asia_window(now, STATE.asia_open_et, STATE.asia_close_et):
                    for sym in STATE.symbols:
                        # rate-limit per symbol (avoid spamming; 5 min)
                        last_ts = STATE.last_alert_ts.get(sym, 0)
                        if now - last_ts < 300:
                            continue
                        alert = await scan_symbol(sym)
                        if alert:
                            await app.bot.send_message(chat_id=chat_id, text=f"[TopTrapHunter] {alert}")
                            STATE.last_alert_ts[sym] = now
                else:
                    # outside window; sleep a bit longer
                    pass
            await asyncio.sleep(30)  # scan cadence
        except Exception as e:
            logger.exception(f"scanner loop error: {e}")
            await asyncio.sleep(10)

# ---------------------- Telegram Handlers ----------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Register chat id
    chat_id = update.effective_chat.id
    save_chat_id(chat_id)
    await update.message.reply_text(
        "TopTrapHunterBot is online.\n"
        "I'll scan BTC, XRP, XLM, HBAR, SUI, LINK for A+ traps (long/short) during your Asia window.\n\n"
        "Commands:\n"
        "/arm – start scanning\n"
        "/disarm – stop scanning\n"
        f"/setwindow HH:MM HH:MM – set Asia window (ET). Current: {STATE.asia_open_et}–{STATE.asia_close_et}\n"
        "/setsymbols BTC,XRP,XLM,HBAR,SUI,LINK – set symbols\n"
        f"/setrr 2.0 – min R:R (current {STATE.min_rr})\n"
        "/status – show settings"
    )

async def arm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.armed = True
    await update.message.reply_text("✅ Armed. I will scan on schedule and alert only A+ trap setups.")

async def disarm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE.armed = False
    await update.message.reply_text("⏸️ Disarmed. Scanning paused.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = load_chat_id()
    await update.message.reply_text(
        f"Armed: {STATE.armed}\n"
        f"Chat ID: {chat_id}\n"
        f"Symbols: {', '.join(STATE.symbols)}\n"
        f"Asia window (ET): {STATE.asia_open_et} – {STATE.asia_close_et}\n"
        f"Min RR: {STATE.min_rr}\n"
        "Note: I alert only on last CLOSED candles (no repaint)."
    )

async def setwindow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        open_s, close_s = context.args[0], context.args[1]
        parse_time_hhmm(open_s)
        parse_time_hhmm(close_s)
        STATE.asia_open_et = open_s
        STATE.asia_close_et = close_s
        await update.message.reply_text(f"✅ Asia window set to {open_s}–{close_s} (ET)")
    except Exception:
        await update.message.reply_text("Usage: /setwindow 20:00 02:00")

async def setsymbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        raw = context.args[0]
        parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
        # Map to ccxt symbols with /USDT suffix if needed
        mapped = [p if "/" in p else f"{p}/USDT" for p in parts]
        STATE.symbols = mapped
        await update.message.reply_text(f"✅ Symbols set: {', '.join(mapped)}")
    except Exception:
        await update.message.reply_text("Usage: /setsymbols BTC,XRP,XLM,HBAR,SUI,LINK")

async def setrr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        v = float(context.args[0])
        STATE.min_rr = max(1.0, min(5.0, v))
        await update.message.reply_text(f"✅ Min R:R set to {STATE.min_rr}")
    except Exception:
        await update.message.reply_text("Usage: /setrr 2.0")

# --------------------------- Main -----------------------------

def build_app() -> Application:
    token = BOT_TOKEN
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var. Create a .env with TELEGRAM_BOT_TOKEN=...")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("arm", arm))
    app.add_handler(CommandHandler("disarm", disarm))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("setwindow", setwindow))
    app.add_handler(CommandHandler("setsymbols", setsymbols))
    app.add_handler(CommandHandler("setrr", setrr))
    return app

async def main():
    app = build_app()
    # Start background scanner
    asyncio.create_task(scanner_loop(app))
    await app.initialize()
    await app.start()
    logger.info("TopTrapHunterBot started. Send /start in Telegram.")
    try:
        await app.updater.start_polling()
    except AttributeError:
        # PTB v20: use app.run_polling instead; but we want to keep custom loop
        pass
    # Keep running
    await app.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
