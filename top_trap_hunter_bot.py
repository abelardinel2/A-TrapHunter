#!/usr/bin/env python3
"""
TopTrapHunterBot
----------------
A Telegram bot that scans BTC, XRP, XLM, HBAR, SUI, LINK on Binance
for “trap zone” shorts: price near recent swing-high + RSI overbought +
bearish momentum/volume hints. Sends alerts only when ARMED.

Requirements:
  pip install python-telegram-bot==20.6 ccxt pandas numpy python-dotenv pytz

Run:
  python top_trap_hunter_bot.py
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
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
)

# ------------------------------ Config ------------------------------

DEFAULT_SYMBOLS = ["BTC/USDT", "XRP/USDT", "XLM/USDT", "HBAR/USDT", "SUI/USDT", "LINK/USDT"]

# Session (U.S. Eastern) — scan more aggressively during Asia open by default
ASIA_OPEN_ET = "20:00"  # 8:00 PM ET
ASIA_CLOSE_ET = "04:00"  # 4:00 AM ET

# Scan cadence
SCAN_EVERY_SEC = 300  # 5 minutes

# Data lookbacks
LOOKBACK_MIN = "1h"      # primary timeframe
LOOKBACK_BARS = 420      # ~17.5 days on 1h
VOL_WINDOW = 5           # recent vs prior volume comparison
RSI_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL = 9

# Risk / Reward baseline (can be changed with /setrr)
MIN_RR = 2.0

# Files
CHAT_ID_FILE = "tth_chat_id.txt"
STATE_FILE = "tth_state.json"

# ------------------------------ Logging ------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("TopTrapHunterBot")

# ------------------------------ State ------------------------------

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

@dataclass
class BotState:
    armed: bool = False
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    asia_open_et: str = ASIA_OPEN_ET
    asia_close_et: str = ASIA_CLOSE_ET
    min_rr: float = MIN_RR
    last_alert_ts: Dict[str, float] = field(default_factory=dict)

STATE = BotState()

# ------------------------------ Helpers ------------------------------

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

def in_asia_session(now_ts: float, asia_open: str, asia_close: str) -> bool:
    """Return True if current time (US/Eastern) is within Asia session."""
    tz = pytz.timezone("US/Eastern")
    now = pd.Timestamp.fromtimestamp(now_ts, tz=tz)
    open_t = pd.Timestamp(now.date(), tz=tz) + pd.Timedelta(
        hours=int(asia_open.split(":")[0]), minutes=int(asia_open.split(":")[1])
    )
    close_t = pd.Timestamp(now.date(), tz=tz) + pd.Timedelta(
        hours=int(asia_close.split(":")[0]), minutes=int(asia_close.split(":")[1])
    )
    if asia_open < asia_close:
        return open_t <= now <= close_t
    else:
        # spans midnight
        return now >= open_t or now <= close_t

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(series, EMA_FAST)
    slow = ema(series, EMA_SLOW)
    macd_line = fast - slow
    signal = ema(macd_line, MACD_SIGNAL)
    hist = macd_line - signal
    return macd_line, signal, hist

def recent_swing_high(prices: pd.Series, bars: int = 240) -> float:
    bars = min(bars, len(prices))
    return float(prices.tail(bars).max())

def percent_from(value: float, anchor: float) -> float:
    if anchor == 0:
        return 0.0
    return (value - anchor) / anchor * 100.0

def volume_diverging(vol: pd.Series, window: int = VOL_WINDOW) -> bool:
    """Recent avg vol < prior avg vol => waning participation (bearish at highs)."""
    if len(vol) < window * 2:
        return False
    recent = vol.tail(window).mean()
    prior = vol.tail(window * 2).head(window).mean()
    return recent < prior

def cooldown_ok(symbol: str, cooldown_sec: int = 1800) -> bool:
    """Prevent spam: don’t alert on same symbol too frequently (default 30 min)."""
    last = STATE.last_alert_ts.get(symbol, 0)
    return (time.time() - last) > cooldown_sec

def mark_alert(symbol: str) -> None:
    STATE.last_alert_ts[symbol] = time.time()

# ------------------------------ Exchange ------------------------------

def get_exchange() -> ccxt.Exchange:
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return ex

async def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df
    except Exception as e:
        logger.warning(f"OHLCV fetch failed {symbol} {timeframe}: {e}")
        return None

# ------------------------------ Trap Logic ------------------------------

def find_short_trap(df: pd.DataFrame) -> Optional[Dict]:
    """
    A pragmatic, conservative trap:
      1) Price near recent swing high (within ~0.5% .. 1.2%)
      2) RSI(14) > 70 (overbought)
      3) Volume divergence: recent volume lower than prior
      4) MACD histogram rolling over (last < previous)
    Returns dict with levels if matched.
    """
    if df is None or len(df) < max(EMA_SLOW + MACD_SIGNAL + 5, RSI_PERIOD + 5):
        return None

    close = df["close"]
    high = df["high"]
    vol = df["volume"]

    swing = recent_swing_high(high, bars=240)
    last = float(close.iloc[-1])
    dist = abs(percent_from(last, swing))

    rsi_series = rsi(close, RSI_PERIOD)
    macd_line, signal, hist = macd(close)

    cond_near_high = (0.0 <= percent_from(last, swing) <= 1.2)  # at/just under swing high
    cond_rsi_overbought = rsi_series.iloc[-1] > 70
    cond_vol_fading = volume_diverging(vol, VOL_WINDOW)
    cond_hist_rollover = len(hist) > 2 and hist.iloc[-1] < hist.iloc[-2]

    if cond_near_high and cond_rsi_overbought and cond_vol_fading and cond_hist_rollover:
        # Simple levels: entry = last, stop a tad above swing, target = mid of last range or 50EMA
        entry = last
        stop = float(swing * 1.004)  # ~0.4% above swing
        # target: either 50-EMA or last visible support
        ema50 = ema(close, 50).iloc[-1]
        support = float(min(df["low"].tail(48)))  # last 2 days (1h) swing low
        tgt = float(max(min(entry - (stop - entry) * STATE.min_rr, entry - (entry - support) * 0.8), ema50))
        rr = (entry - tgt) / (stop - entry) if (stop - entry) > 0 else 0
        return {
            "entry": round(entry, 6),
            "stop": round(stop, 6),
            "target": round(tgt, 6),
            "rr": round(rr, 2),
            "swing": round(swing, 6),
            "rsi": round(float(rsi_series.iloc[-1]), 2),
        }
    return None

# ------------------------------ Telegram Handlers ------------------------------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        save_chat_id(update.effective_chat.id)
    await update.message.reply_text(
        "👋 TopTrapHunterBot online.\n"
        "Commands:\n"
        "  /arm – start scanning\n"
        "  /disarm – stop scanning\n"
        "  /status – show state\n"
        "  /setsymbols BTC/USDT,XRP/USDT,...\n"
        "  /setrr 2.5  (min risk:reward)\n"
        "I’ll only alert when ARMED."
    )

async def arm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.armed = True
    await update.message.reply_text("🟢 Armed. Scanning every 5 minutes.")

async def disarm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE.armed = False
    await update.message.reply_text("🔴 Disarmed. Scanning paused.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    now = time.time()
    asia = in_asia_session(now, STATE.asia_open_et, STATE.asia_close_et)
    await update.message.reply_text(
        "📊 Status\n"
        f"Armed: {STATE.armed}\n"
        f"Symbols: {', '.join(STATE.symbols)}\n"
        f"Min R:R: {STATE.min_rr}\n"
        f"Asia (ET): {STATE.asia_open_et}–{STATE.asia_close_et} | Now in-window: {asia}"
    )

async def setsymbols_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /setsymbols BTC/USDT,XRP/USDT,...")
        return
    raw = " ".join(context.args).strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        await update.message.reply_text("No symbols parsed.")
        return
    STATE.symbols = parts
    await update.message.reply_text(f"✅ Symbols updated:\n{', '.join(STATE.symbols)}")

async def setrr_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /setrr 2.0")
        return
    try:
        rr = float(context.args[0])
        if rr <= 0:
            raise ValueError
        STATE.min_rr = rr
        await update.message.reply_text(f"✅ Min R:R set to {STATE.min_rr}")
    except Exception:
        await update.message.reply_text("Please provide a positive number, e.g. /setrr 2.0")

# ------------------------------ Scanner Job ------------------------------

async def scanner_loop(app) -> None:
    """Background loop: every SCAN_EVERY_SEC, scan when armed."""
    ex = get_exchange()
    chat_id = load_chat_id()

    # startup ping
    if chat_id:
        try:
            await app.bot.send_message(chat_id=chat_id, text="✅ TopTrapHunterBot is online and scanning traps!")
        except Exception as e:
            logger.warning(f"Startup message failed: {e}")

    while True:
        try:
            if STATE.armed:
                now = time.time()
                prefer_alerts = in_asia_session(now, STATE.asia_open_et, STATE.asia_close_et)
                for sym in STATE.symbols:
                    # Light cooldown to avoid spam
                    if not cooldown_ok(sym, cooldown_sec=1800 if not prefer_alerts else 900):
                        continue

                    df = await fetch_ohlcv_df(ex, sym, LOOKBACK_MIN, LOOKBACK_BARS)
                    if df is None:
                        continue

                    trap = find_short_trap(df)
                    if trap and trap["rr"] >= STATE.min_rr:
                        mark_alert(sym)
                        text = (
                            f"⚠️ Possible SHORT Trap on *{sym}*\n"
                            f"Price near swing high {trap['swing']}\n"
                            f"RSI: {trap['rsi']}\n"
                            f"Entry: `{trap['entry']}`  Stop: `{trap['stop']}`  Target: `{trap['target']}`\n"
                            f"Est. R:R ≈ *{trap['rr']}*\n"
                            f"_Confirm on your chart (1H/15m) before acting._"
                        )
                        if chat_id:
                            try:
                                await app.bot.send_message(
                                    chat_id=chat_id,
                                    text=text,
                                    parse_mode="Markdown",
                                )
                            except Exception as e:
                                logger.warning(f"Send failed: {e}")
            await asyncio.sleep(SCAN_EVERY_SEC)
        except Exception as e:
            logger.exception(f"scanner loop error: {e}")
            await asyncio.sleep(10)

# ------------------------------ Main ------------------------------

def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

    application = ApplicationBuilder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("arm", arm_cmd))
    application.add_handler(CommandHandler("disarm", disarm_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("setsymbols", setsymbols_cmd))
    application.add_handler(CommandHandler("setrr", setrr_cmd))

    # Run scanner in background
    application.post_init = lambda app: asyncio.create_task(scanner_loop(app))

    application.run_polling(close_loop=False)

if __name__ == "__main__":
    main()