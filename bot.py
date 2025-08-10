#!/usr/bin/env python3
# TopTrapHunterBot (fixed event loop with enhancements)
# PTB v20+, Python 3.10+

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import aiofiles
import ccxt
import numpy as np
import pandas as pd
import pytz
from datetime import time
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
MIN_RR = 2.0             # Minimum risk:reward to alert
COOLDOWN_MINUTES = 60    # Alert cooldown per symbol

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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ------------------------- STATE --------------------------

@dataclass
class BotState:
    armed: bool = False
    symbols: List[str] = field(default_factory=lambda: SYMBOLS_DEFAULT.copy())
    asia_open_et: str = ASIA_OPEN_ET
    asia_close_et: str = ASIA_CLOSE_ET
    min_rr: float = MIN_RR
    last_alert_ts: Dict[str, float] = field(default_factory=dict)
    timeframe: str = "15m"

STATE = BotState()

async def _load_chat_ids() -> Set[int]:
    try:
        if os.path.exists(CHAT_ID_FILE):
            async with aiofiles.open(CHAT_ID_FILE, "r") as f:
                return {int(cid) for cid in (await f.read()).splitlines() if cid.strip()}
    except Exception as e:
        log.warning("Failed loading chat IDs: %s", e)
    return set()

async def _save_chat_ids(chat_ids: Set[int]) -> None:
    try:
        async with aiofiles.open(CHAT_ID_FILE, "w") as f:
            await f.write("\n".join(str(cid) for cid in chat_ids))
    except Exception as e:
        log.warning("Failed saving chat IDs: %s", e)

async def _load_state() -> None:
    if not os.path.exists(STATE_FILE):
        return
    try:
        async with aiofiles.open(STATE_FILE, "r") as f:
            data = json.loads(await f.read())
            STATE.armed = bool(data.get("armed", False))
            STATE.symbols = list(data.get("symbols", SYMBOLS_DEFAULT))
            STATE.asia_open_et = data.get("asia_open_et", ASIA_OPEN_ET)
            STATE.asia_close_et = data.get("asia_close_et", ASIA_CLOSE_ET)
            STATE.min_rr = float(data.get("min_rr", MIN_RR))
            STATE.last_alert_ts = dict(data.get("last_alert_ts", {}))
            STATE.timeframe = data.get("timeframe", "15m")
    except Exception as e:
        log.warning("Failed loading state: %s", e)

async def _save_state() -> None:
    data = {
        "armed": STATE.armed,
        "symbols": STATE.symbols,
        "asia_open_et": STATE.asia_open_et,
        "asia_close_et": STATE.asia_close_et,
        "min_rr": STATE.min_rr,
        "last_alert_ts": STATE.last_alert_ts,
        "timeframe": STATE.timeframe,
    }
    try:
        async with aiofiles.open(STATE_FILE, "w") as f:
            await f.write(json.dumps(data))
    except Exception as e:
        log.warning("Failed saving state: %s", e)

CHAT_IDS = asyncio.run(_load_chat_ids())  # Run sync at startup
asyncio.run(_load_state())
if TELEGRAM_CHAT_ID:
    try:
        CHAT_IDS.add(int(TELEGRAM_CHAT_ID))
        asyncio.run(_save_chat_ids(CHAT_IDS))
    except ValueError:
        log.warning("Invalid TELEGRAM_CHAT_ID in .env: %s", TELEGRAM_CHAT_ID)

# ------------------------- EXCHANGE -----------------------

exchange = ccxt.binance({"enableRateLimit": True})

async def fetch_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 200) -> pd.DataFrame:
    try:
        data = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df
    except Exception as e:
        log.error("Failed to fetch OHLCV for %s: %s", symbol, e)
        raise

# ------------------------- SIGNALS ------------------------

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

async def green_light_signal(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    if symbol in STATE.last_alert_ts:
        last_ts = pd.Timestamp(STATE.last_alert_ts[symbol], unit="s", tz="UTC")
        if (pd.Timestamp.utcnow() - last_ts).total_seconds() < COOLDOWN_MINUTES * 60:
            return None
    if len(df) < 80:
        return None
    close = df["close"]
    vol = df["vol"]
    swing_high = close.rolling(50