#!/usr/bin/env python3
"""
pt_engine5.py — Engine 5: AI Advisory Read (TEST MODE)  (peterpt 2026-07-26)

WHAT THIS IS
  A standalone engine that, when an instrument reaches a decision point,
  assembles a feature vector from the OTHER engines' existing outputs and asks
  a configurable AI provider (OpenAI / Anthropic / Google) for a single read:
  is this a good place to buy, sell, or stay out?

  In TEST MODE (the only mode for now) it does NOTHING to the trading system.
  It observes, asks the AI, and LOGS the read alongside the moment/level so you
  can grade it against the chart later — "AI said neutral here, E3 sold on the
  cross; did the sell win or lose?". It is a silent observer earning evidence.

WHAT THIS IS NOT (yet)
  It is NOT wired into powertrader.py, and no engine consults it. Connecting it
  (Phase 2 — engines ask E5 before acting) happens only AFTER the logged reads
  prove the AI adds edge. Until then this file runs alone or not at all.

SAFETY POSTURE (same discipline as the rest of PowerTrader)
  • The AI is advisory. This engine never sends a trade signal.
  • The AI call can fail/stall/time-out — it comes back as a structured error and
    E5 just logs "no read"; nothing breaks (fragile component is never load-bearing).
  • temperature is forced low and output is clamped to a fixed menu — an AI that
    can only pick from {buy, sell, neutral, avoid} can't invent a crazy action.
  • Provider + key + model come from config; the call is identical regardless of
    which provider is chosen ("equal to all engines").

OUTPUT
  hub_data/engine5_reads.jsonl   — append-only log of every read (dashboard tab)
  hub_data/engine5_latest_<SYM>.json — most recent read per instrument
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, Optional, List

import pt_e5_providers as providers

# ── Paths (relative to this file, matching sentiment_fetcher's convention) ────
_HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT      = _HERE
HUB_DIR   = os.environ.get("POWERTRADER_HUB") or os.path.join(ROOT, "hub_data")
RAMFS_DIR = os.path.join(HUB_DIR, "ramfs")
GUI_SETTINGS_PATH = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(ROOT, "gui_settings.json")

READS_LOG_PATH    = os.path.join(HUB_DIR, "engine5_reads.jsonl")
# Static per-instrument "personality" notes (peterpt/ChatGPT 2026-07-29, #10):
# hand-written domain hints (GOLD trends, NATURALGAS chaotic, ...) injected as a
# one-line prior. Pure config, no computation — tune freely.
INSTRUMENT_NOTES_PATH = os.path.join(ROOT, "instrument_personality.json")
CANDLES_DIR       = os.path.join(HUB_DIR, "candles")
QUOTA_FILE        = os.path.join(ROOT, "e5_quota_blocked.txt")  # circuit-breaker

_RUN = True


def _sigterm(*_a):
    global _RUN
    _RUN = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ── Config ────────────────────────────────────────────────────────────────────
def _seed_e5_defaults() -> None:
    """Write E5's settings into gui_settings.json if missing (peterpt 2026-07-26),
    the same self-seeding pattern the other engines use. Only ADDS missing keys —
    never overwrites values the user has set. So first launch creates the config
    the user can then edit (via dashboard later, or by hand now)."""
    defaults = {
        "e5_enabled":    False,      # opt-in — must be turned on to do anything
        "e5_provider":   "openai",   # openai | anthropic | google
        "e5_model":      "",         # blank = provider default (see providers file)
        "e5_mode":       "test",     # test = log only (the only mode for now)
        "e5_cooldown_s": 900,        # min seconds between reads per instrument
        "e5_decision_threshold": 0.3,  # global -1..+1 gate threshold (all engines)
        "e1_ai_decision": False,     # per-engine AI-gate toggles (all default OFF;
        "e2_ai_decision": False,     # can only be ON while e5_enabled; auto-flip
        "e3_ai_decision": False,     # OFF when E5 is disabled; manual re-arm after
        "e4_ai_decision": False,     # E5 is re-enabled)
    }
    try:
        gs = {}
        if os.path.isfile(GUI_SETTINGS_PATH):
            with open(GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
                gs = json.load(f)
        changed = False
        for k, v in defaults.items():
            if k not in gs:
                gs[k] = v
                changed = True
        # ensure the api_keys dict + provider slots exist (don't touch existing keys)
        keys = gs.get("api_keys")
        if not isinstance(keys, dict):
            keys = {}
            gs["api_keys"] = keys
            changed = True
        for prov in ("openai", "anthropic", "google"):
            if prov not in keys:
                keys[prov] = ""
                changed = True
        if changed:
            tmp = GUI_SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(gs, f, indent=2)
            os.replace(tmp, GUI_SETTINGS_PATH)
            log("[E5] seeded default settings into gui_settings.json "
                "(e5_enabled=False — turn it on to start)")
    except Exception as e:
        log(f"[E5] could not seed defaults: {e}")


def load_settings() -> dict:
    try:
        with open(GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _key_from_sentiment_sources(provider: str) -> str:
    """Fallback: read a provider key from sentiment_sources.json (peterpt
    2026-07-26). sentiment_fetcher already stores the OpenAI key there under
    api_keys.openai, so E5 can reuse it instead of the user re-pasting. Only
    OpenAI is typically present there, but we read whatever matches."""
    try:
        p = os.path.join(ROOT, "sentiment_sources.json")
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        k = (d.get("api_keys", {}) or {}).get(provider, "")
        return k if (k and not str(k).startswith("YOUR_")) else ""
    except Exception:
        return ""


def _resolve_key(provider: str, gui_keys: dict) -> str:
    """Resolve the API key for a provider with a fallback chain:
      1. E5's own key in gui_settings.json  (api_keys.<provider>)
      2. sentiment_sources.json             (api_keys.<provider>) — reuses the
         existing OpenAI key already on the box, no re-paste needed.
    Returns "" if neither has a usable key."""
    k = str(gui_keys.get(provider, "") or "")
    if k and not k.startswith("YOUR_"):
        return k
    return _key_from_sentiment_sources(provider)


def e5_config(gs: dict) -> dict:
    """E5's settings, with safe defaults. All live under gui_settings so the
    dashboard can edit them later.
      e5_enabled     : master on/off (default False — opt-in)
      e5_provider    : "openai" | "anthropic" | "google"
      e5_model       : override model (blank = provider default)
      e5_mode        : "test" (log only) — the only supported mode for now
      e5_cooldown_s  : min seconds between reads for the same instrument
    Keys live in gui_settings["api_keys"][provider].
    """
    keys = gs.get("api_keys", {}) or {}
    return {
        "enabled":   bool(gs.get("e5_enabled", False)),
        "provider":  str(gs.get("e5_provider", "openai")).lower().strip(),
        "model":     str(gs.get("e5_model", "")).strip(),
        "mode":      str(gs.get("e5_mode", "test")).lower().strip(),
        "cooldown_s": int(gs.get("e5_cooldown_s", 900)),
        # Legacy passive-observation loop (do_read). Superseded by e5_gate, which
        # the engines call for real decisions. do_read made a SECOND, duplicate
        # AI call per signal purely to log an observation — doubling token cost
        # and producing the paired no-read rows. Default OFF now; the gate is the
        # decision path. Set e5_observe=true only to resume passive logging.
        # (peterpt 2026-07-31)
        "observe":   bool(gs.get("e5_observe", False)),
        # Use AI-suggested entry price (peterpt 2026-07-31). When Enabled, engines
        # may place at the AI's suggested price instead of current price. Default
        # Disabled = current behaviour (trade at instrument price). PHASE 1: even
        # when enabled, we validate + LOG the AI price vs current but do not yet
        # change execution — proving the AI prices are sane and fill before they
        # touch real orders. Execution wiring is a deliberate second step.
        "use_ai_price": bool(gs.get("e5_use_ai_price", False)),
        "keys":      {
            # Resolve each with the fallback chain (gui_settings → sentiment_sources).
            "openai":    _resolve_key("openai", keys),
            "anthropic": _resolve_key("anthropic", keys),
            "google":    _resolve_key("google", keys),
        },
    }


# ── Small readers for other engines' existing outputs ─────────────────────────
def _rj(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _live_prices() -> dict:
    # Prefer ramfs (WebSocket-written) like the engines do.
    for p in (os.path.join(RAMFS_DIR, "live_prices.json"),
              os.path.join(HUB_DIR, "live_prices.json")):
        d = _rj(p)
        if isinstance(d, dict) and d:
            return d
    return {}


def _price_of(prices: dict, sym: str) -> Optional[float]:
    v = prices.get(sym)
    if isinstance(v, dict):
        for k in ("mid", "price", "bid", "ask"):
            if v.get(k):
                try:
                    return float(v[k])
                except Exception:
                    pass
    try:
        return float(v)
    except Exception:
        return None


def _recent_candles(sym: str, tf: str = "15min", n: int = 12) -> List[dict]:
    # For 15min, prefer the thinker's live sliding-window file that E3 trusts
    # (<SYM>_15min_rolling.json) — the plain <SYM>_15min.json can be short/stale,
    # which starved MACD (peterpt 2026-07-28). Fall back to the plain file.
    candidates = []
    if tf == "15min":
        candidates.append(f"{sym}_15min_rolling.json")
    candidates.append(f"{sym}_{tf}.json")
    for fname in candidates:
        d = _rj(os.path.join(CANDLES_DIR, fname))
        cs = d.get("candles", d) if isinstance(d, dict) else d
        if isinstance(cs, list) and cs:
            return cs[-n:]
    return []


def _neurals(sym: str) -> dict:
    """Nearest support/resistance from the neural cache, if present."""
    d = _rj(os.path.join(HUB_DIR, "neural_cache.json")) or {}
    entry = d.get(sym) if isinstance(d, dict) else None
    if not isinstance(entry, dict):
        return {}
    return entry


def _choch_latest(sym: str) -> dict:
    return _rj(os.path.join(HUB_DIR, f"engine3a_choch_latest_{sym}.json")) or {}


def _sentiment(sym: str) -> Optional[float]:
    d = _rj(os.path.join(HUB_DIR, "sentiment_approved.json")) or {}
    e = d.get(sym) if isinstance(d, dict) else None
    if isinstance(e, dict):
        try:
            return float(e.get("score"))
        except Exception:
            return None
    return None


# ── Feature assembly ──────────────────────────────────────────────────────────
def _swing_map(candles: List[dict], length: int = 2) -> List[dict]:
    """Extract swing highs/lows over the candle window with the volume traded at
    each swing (peterpt 2026-07-26). Gives the AI the *journey* of price — where
    it turned, and how much volume (liquidity) sat at each turn — not just the
    current snapshot. A swing high with heavy volume is a level price fought at;
    that's exactly what a human reads off a chart.

    Returns oldest→newest: [{type:'H'|'L', price, volume, ago_candles}].
    """
    out = []
    n = len(candles)
    for i in range(length, n - length):
        c = candles[i]
        try:
            hi = float(c.get("h", c.get("c")))
            lo = float(c.get("l", c.get("c")))
            vol = float(c.get("v", 0) or 0)
        except Exception:
            continue
        left = range(i - length, i)
        right = range(i + 1, i + length + 1)
        is_h = (all(hi >= float(candles[j].get("h", candles[j].get("c", 0))) for j in left) and
                all(hi >  float(candles[j].get("h", candles[j].get("c", 0))) for j in right))
        is_l = (all(lo <= float(candles[j].get("l", candles[j].get("c", 0))) for j in left) and
                all(lo <  float(candles[j].get("l", candles[j].get("c", 0))) for j in right))
        if is_h:
            out.append({"type": "H", "price": round(hi, 6),
                        "volume": round(vol), "ago_candles": (n - 1 - i)})
        if is_l:
            out.append({"type": "L", "price": round(lo, 6),
                        "volume": round(vol), "ago_candles": (n - 1 - i)})
    return out


def assemble_features(sym: str, trigger: str, prices: dict) -> Optional[dict]:
    """Build the per-instrument feature vector from what the other engines have
    already computed and written to disk. Returns None if we can't even get a
    price (nothing worth asking about)."""
    px = _price_of(prices, sym)
    if px is None:
        return None

    # 4 hours of 15min candles = 16 bars — enough to show the recent journey.
    candles = _recent_candles(sym, tf="15min", n=17)
    closes = [c.get("c") for c in candles if isinstance(c, dict)]
    # crude regime read: count local direction flips in recent closes
    flips = 0
    for i in range(2, len(closes)):
        try:
            a, b, c = closes[i - 2], closes[i - 1], closes[i]
            if (b - a) * (c - b) < 0:
                flips += 1
        except Exception:
            pass
    regime = "choppy" if flips >= max(3, len(closes) // 3) else "trending/quiet"

    # Swing map over the 4h window: where price turned + volume (liquidity) there.
    swings = _swing_map(candles, length=2)
    # 4h range summary so the AI sees the corridor price has traveled.
    _hs = [float(c.get("h", c.get("c", 0))) for c in candles if isinstance(c, dict)]
    _ls = [float(c.get("l", c.get("c", 0))) for c in candles if isinstance(c, dict)]
    range_4h = None
    if _hs and _ls:
        _hi, _lo = max(_hs), min(_ls)
        range_4h = {
            "high": round(_hi, 6), "low": round(_lo, 6),
            "pct": round((_hi - _lo) / px * 100, 3) if px else None,
            "position": (round((px - _lo) / (_hi - _lo), 2)
                         if (_hi - _lo) else None),   # 0=at low, 1=at high
        }

    neu = _neurals(sym)
    choch = _choch_latest(sym)
    sent = _sentiment(sym)

    feat = {
        "instrument": sym,
        "trigger":    trigger,   # what woke this read (e3b_cross, e1_touch, ...)
        "price":      round(px, 8),
        "range_4h":   range_4h,          # the corridor price traveled
        "swings_4h":  swings,            # turning points + liquidity at each
        "recent_closes": [round(float(c), 8) for c in closes if c is not None][-8:],
        "regime":     regime,
        "structure_choch": (choch.get("label") if choch else None),
        "structure_bias":  (choch.get("os") if choch else None),
        "sentiment_estimate": (
            {"value": sent, "reliability": "low"} if sent is not None else None),
    }
    # nearest S/R if the neural cache exposes it
    if isinstance(neu, dict):
        for k in ("support", "resistance", "nearest_support", "nearest_resistance"):
            if k in neu:
                feat[k] = neu[k]
    return feat


# ── Prompt (guardrailed) ──────────────────────────────────────────────────────
def _macd_state(candles: List[dict]) -> str:
    """Plain-language MACD state for the gate prompt (peterpt 2026-07-26):
    'upward', 'downward', 'reversing up', 'reversing down', or 'flat'. The AI
    gets a described state, not raw numbers — 'no cross but reversing' is more
    useful than two floats."""
    closes = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        v = c.get("c", c.get("close", c.get("close_price", c.get("Close"))))
        if v is not None:
            try:
                closes.append(float(v))
            except Exception:
                pass
    if len(closes) < 26:
        return "insufficient data"
    def _ema(vals, n):
        k = 2 / (n + 1)
        e = vals[0]
        out = [e]
        for v in vals[1:]:
            e = v * k + e * (1 - k)
            out.append(e)
        return out
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema(macd, 9)
    hist = [m - s for m, s in zip(macd, signal)]
    if len(hist) < 3:
        return "flat"
    h0, h1, h2 = hist[-1], hist[-2], hist[-3]
    # direction of the histogram + whether it's turning
    if h0 > 0 and h1 > 0:
        return "upward" if h0 >= h1 else "upward but weakening"
    if h0 < 0 and h1 < 0:
        return "downward" if h0 <= h1 else "downward but weakening"
    # sign change / near zero = reversing
    if h1 <= 0 < h0:
        return "reversing upward (bullish cross forming)"
    if h1 >= 0 > h0:
        return "reversing downward (bearish cross forming)"
    return "flat / no clear direction"


def _closest_neurals(sym: str, price: float) -> dict:
    """Nearest buy (support below) and sell (resistance above) levels for sym.

    Reads hub_data/signal_radar.json — the file the THINKER writes live every
    sweep (peterpt 2026-07-28), NOT neural_cache.json (which only holds levels
    when neurals are locked). signal_radar.json always has the current levels:
      nearest_long_level  = nearest support below price (a "buy" level)
      nearest_short_level = nearest resistance above price (a "sell" level)
    Falls back to computing from long_levels/short_levels if the nearest_* keys
    are absent."""
    out = {"closest_buy": None, "closest_sell": None}
    d = _rj(os.path.join(HUB_DIR, "signal_radar.json"))
    entry = d.get(sym) if isinstance(d, dict) else None
    if not isinstance(entry, dict):
        return out
    # Preferred: thinker already computed the nearest levels.
    nb = entry.get("nearest_long_level")
    ns = entry.get("nearest_short_level")
    try:
        if nb is not None:
            out["closest_buy"] = round(float(nb), 8)
    except Exception:
        pass
    try:
        if ns is not None:
            out["closest_sell"] = round(float(ns), 8)
    except Exception:
        pass
    # Fallback: derive from the level lists relative to price.
    if out["closest_buy"] is None or out["closest_sell"] is None:
        try:
            longs = [float(x) for x in (entry.get("long_levels") or [])]
            shorts = [float(x) for x in (entry.get("short_levels") or [])]
            p = price or float(entry.get("current_price", 0) or 0)
            if out["closest_buy"] is None:
                below = [x for x in longs if x < p] or longs
                if below:
                    out["closest_buy"] = round(max(below), 8)
            if out["closest_sell"] is None:
                above = [x for x in shorts if x > p] or shorts
                if above:
                    out["closest_sell"] = round(min(above), 8)
        except Exception:
            pass
    return out


def _sentiment_with_age(sym: str) -> Optional[dict]:
    d = _rj(os.path.join(HUB_DIR, "sentiment_approved.json")) or {}
    e = d.get(sym) if isinstance(d, dict) else None
    if not isinstance(e, dict):
        return None
    try:
        score = float(e.get("score"))
    except Exception:
        return None
    age_s = None
    for tk in ("ts", "timestamp", "approved_at"):
        if e.get(tk):
            try:
                age_s = int(time.time() - float(e[tk]))
                break
            except Exception:
                pass
    return {"value": score, "age_secs": age_s}


def _vwap_from_candles(sym: str, tf: str = "15min", n: int = 32) -> Optional[float]:
    """Session-ish VWAP from candles (peterpt/ChatGPT 2026-07-29, #2). VWAP =
    sum(typical_price * vol) / sum(vol), typical = (h+l+c)/3. Computed directly
    from E5's own candles (always available; no dependence on E3B's event log).
    Returns None if no volume data."""
    candles = _recent_candles(sym, tf=tf, n=n)
    num = den = 0.0
    for c in candles:
        if not isinstance(c, dict):
            continue
        try:
            h = float(c.get("h")); l = float(c.get("l"))
            cl = float(c.get("c")); v = float(c.get("v") or 0)
        except Exception:
            continue
        if v <= 0:
            continue
        typ = (h + l + cl) / 3.0
        num += typ * v
        den += v
    return (num / den) if den > 0 else None


def _alignment_score(tf_stack_pairs) -> int:
    """% of timeframes agreeing on direction (peterpt/ChatGPT 2026-07-29).
    tf_stack_pairs = list of 'UP'/'DOWN'/'flat' strings. Alignment = share of
    non-flat TFs pointing the same way as the majority. 100% = all agree."""
    dirs = [d for d in tf_stack_pairs if d in ("UP", "DOWN")]
    if not dirs:
        return 0
    ups = dirs.count("UP")
    downs = dirs.count("DOWN")
    majority = max(ups, downs)
    return round(majority / len(dirs) * 100)


def _engine_trigger(engine: str) -> str:
    """Brief description of WHY the engine consulted the AI (peterpt/ChatGPT
    2026-07-29, #7). Describes the TRIGGER TYPE, not a directional bias — the AI
    still judges direction on the data. Helps it weight the signal's nature
    (structural vs momentum vs reversion vs pattern)."""
    e = str(engine).upper()
    return {
        "E1":  "Triggered by: price touching a neural support/resistance level",
        "E2":  "Triggered by: candle pattern and/or news sentiment",
        "3A":  "Triggered by: ATR structural break (CHoCH/BOS)",
        "E3A": "Triggered by: ATR structural break (CHoCH/BOS)",
        "3B":  "Triggered by: MACD/MA momentum cross",
        "E3B": "Triggered by: MACD/MA momentum cross",
        "E4":  "Triggered by: mean-reversion from an extreme",
    }.get(e, "")


def _spread_pct(prices: dict, sym: str) -> Optional[float]:
    """Current bid/ask spread as % of price (peterpt/ChatGPT 2026-07-29, #5).
    A blown-out spread means high execution cost — a good setup with an unusually
    wide spread may not be worth taking. Especially relevant for GOLD/OIL/
    NATURALGAS. Returns None if bid/ask unavailable (line then omitted)."""
    v = prices.get(sym)
    if not isinstance(v, dict):
        return None
    try:
        bid = float(v.get("bid") or 0)
        ask = float(v.get("ask") or v.get("offer") or 0)
        if bid > 0 and ask > 0 and ask >= bid:
            mid = (bid + ask) / 2.0
            return (ask - bid) / mid * 100.0 if mid > 0 else None
    except Exception:
        return None
    return None


def _instrument_note(sym: str) -> str:
    """Static personality note for a symbol (peterpt/ChatGPT 2026-07-29, #10).
    Read from instrument_personality.json — hand-written domain hints. Returns ''
    if none, so the prompt line is simply omitted for unmapped instruments."""
    try:
        with open(INSTRUMENT_NOTES_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return str((d.get("notes", {}) or {}).get(sym, "")).strip()
    except Exception:
        return ""


def _market_session() -> str:
    """Current FX session, DST-AWARE (peterpt/ChatGPT 2026-07-31). The old fixed
    UTC windows were wrong half the year (they ignored BST/EDT). We now compute
    each market's LOCAL time via zoneinfo so London/NY open/close is always
    correct through daylight-saving changes. Sessions by local market hours:
      London  08:00-16:30 Europe/London
      New York 09:30-16:00 America/New_York
      Asia (Tokyo) 09:00-15:00 Asia/Tokyo (rough proxy for the Asian session)
    Overlaps = highest liquidity. Falls back to a UTC approximation if zoneinfo
    is unavailable."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        lon = now.astimezone(ZoneInfo("Europe/London"))
        ny  = now.astimezone(ZoneInfo("America/New_York"))
        tok = now.astimezone(ZoneInfo("Asia/Tokyo"))
        # WEEKEND (peterpt 2026-07-31): the STOCK EXCHANGES (NYSE/LSE/Tokyo)
        # close weekends — so the London/NY/Asia *session* liquidity is absent.
        # But crypto and the broker keep trading, so this is NOT "market closed"
        # — it's "equity sessions closed, thinner conditions". FX/CFD weekend
        # boundary ~Fri 22:00 UTC to Sun 22:00 UTC; crypto trades throughout.
        wd = now.weekday()          # Mon=0 .. Sat=5, Sun=6
        _weekend = (wd == 5
                    or (wd == 6 and now.hour < 22)
                    or (wd == 4 and now.hour >= 22))
        if _weekend:
            return "weekend (stock exchanges closed; crypto/FX only, thinner)"
        lon_open = 8 <= (lon.hour + lon.minute/60) < 16.5
        ny_open  = 9.5 <= (ny.hour + ny.minute/60) < 16
        asia_open = 9 <= tok.hour < 15
        if lon_open and ny_open:
            return "London/NY overlap (high activity)"
        if lon_open:
            return "London (active)"
        if ny_open:
            return "New York (active)"
        if asia_open:
            return "Asia (quieter)"
        return "dead hours (thin/choppy)"
    except Exception:
        # fallback: coarse UTC approximation (may be off by 1h across DST)
        h = now.hour
        if 12 <= h < 16:  return "London/NY overlap (high activity)"
        if 7 <= h < 16:   return "London (active)"
        if 12 <= h < 21:  return "New York (active)"
        if h >= 22 or h < 8: return "Asia (quieter)"
        return "dead hours (thin/choppy)"


def _time_context() -> str:
    """Raw clock context for the AI (peterpt 2026-07-31): the actual local times
    of the major markets, so the AI can reason about 'just opened' / 'about to
    close' nuance beyond the session label. We compute the times (deterministic,
    DST-aware) rather than making the AI do timezone math (which LLMs get wrong).
    Returns '' if zoneinfo unavailable."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        ny  = now.astimezone(ZoneInfo("America/New_York"))
        lon = now.astimezone(ZoneInfo("Europe/London"))
        return (f"Clock: UTC {now.strftime('%H:%M')} | "
                f"London {lon.strftime('%H:%M')} | "
                f"New York {ny.strftime('%H:%M')} "
                f"(NY mkt 09:30-16:00, London 08:00-16:30 local)")
    except Exception:
        return ""


def _htf_trend(sym: str) -> str:
    """Higher-timeframe (daily) trend in words for the gate prompt (peterpt
    2026-07-28, ChatGPT's #4 input). Reads daily candles; falls back to 4h if
    no daily. Returns 'bullish' / 'bearish' / 'flat' / 'unknown'. This is the
    big-picture context the 4h swings alone don't give."""
    for tf in ("1day", "4hour"):
        candles = _recent_candles(sym, tf=tf, n=20)
        closes = [float(c.get("c")) for c in candles
                  if isinstance(c, dict) and c.get("c") is not None]
        if len(closes) >= 6:
            # simple, robust: compare recent close to the window's start + slope
            first, last = closes[0], closes[-1]
            if first <= 0:
                continue
            pct = (last - first) / first * 100.0
            # need a meaningful move to call a trend (avoid labelling noise)
            if pct > 1.5:
                return f"bullish ({tf}, +{pct:.1f}%)"
            if pct < -1.5:
                return f"bearish ({tf}, {pct:.1f}%)"
            return f"flat ({tf}, {pct:+.1f}%)"
    return "unknown"


def _month_trend(sym: str) -> str:
    """1-month trend from daily candles (peterpt 2026-07-29 + ChatGPT's request).
    The existing _htf_trend uses a ~20-bar window; this gives the LONGER context
    both Pedro and ChatGPT flagged as missing — the multi-week direction the
    shorter windows can't show. Reads ~30 daily closes; falls back to 4h*180 if
    no daily. Summarised to one line (not raw candles) to keep tokens lean."""
    candles = _recent_candles(sym, tf="1day", n=30)
    closes = [float(c.get("c")) for c in candles
              if isinstance(c, dict) and c.get("c") is not None]
    tf_label = "1mo"
    if len(closes) < 10:
        # fall back to 4h bars spanning ~a month (30d * 6 bars/day = 180)
        candles = _recent_candles(sym, tf="4hour", n=180)
        closes = [float(c.get("c")) for c in candles
                  if isinstance(c, dict) and c.get("c") is not None]
    if len(closes) < 10:
        return "unknown"
    first, last = closes[0], closes[-1]
    if first <= 0:
        return "unknown"
    pct = (last - first) / first * 100.0
    # 1-month moves are larger, so use a wider deadband than the daily trend.
    if pct > 3.0:
        return f"bullish ({tf_label}, +{pct:.1f}%)"
    if pct < -3.0:
        return f"bearish ({tf_label}, {pct:.1f}%)"
    return f"flat ({tf_label}, {pct:+.1f}%)"


def _tf_dir(sym: str, tf: str, n: int, deadband: float) -> str:
    """One timeframe's direction as a compact word+pct: 'UP (+0.8%)' etc.
    Reads n candles of timeframe tf, compares first vs last close. Deadband
    (percent) sets the flat zone — wider for higher timeframes since their moves
    are larger. Returns 'n/a' if not enough data. peterpt 2026-07-29 (ChatGPT's
    #1 request: a multi-TF alignment stack the AI reasons very well over)."""
    candles = _recent_candles(sym, tf=tf, n=n)
    closes = [float(c.get("c")) for c in candles
              if isinstance(c, dict) and c.get("c") is not None]
    if len(closes) < max(3, n // 3):
        return "n/a"
    first, last = closes[0], closes[-1]
    if first <= 0:
        return "n/a"
    pct = (last - first) / first * 100.0
    if pct > deadband:
        return f"UP (+{pct:.1f}%)"
    if pct < -deadband:
        return f"DOWN ({pct:.1f}%)"
    return f"flat ({pct:+.1f}%)"


def _mtf_stack(sym: str) -> str:
    """Multi-timeframe trend stack (15m/1h/4h/1d) on one line, so the AI can see
    alignment at a glance. All-aligned => stronger conviction; mixed => weaker.
    Deadbands widen with timeframe (noise scales with horizon). peterpt/ChatGPT
    2026-07-29. ~16 bars per TF: enough to show the recent journey, token-lean."""
    # 15min×16 = 4h; 1h×12 = half-day; 4h×12 = 2 days; 1day×10 = ~2 weeks
    t15 = _tf_dir(sym, "15min", 16, 0.15)
    t1h = _tf_dir(sym, "1hour", 12, 0.30)
    t4h = _tf_dir(sym, "4hour", 12, 0.60)
    t1d = _tf_dir(sym, "1day", 10, 1.00)
    return f"15m {t15} | 1h {t1h} | 4h {t4h} | 1d {t1d}"


def _exhaustion(sym: str, atr: Optional[float], tf: str = "15min") -> str:
    """One line describing whether the last CLOSED candle looks like exhaustion.
    (peterpt + Claude 2026-08-03)

    WHY THIS EXISTS. On 2026-08-03 E1 proposed BUY SILVER twice; the AI flipped
    both to SELL and the short was closed at a loss while price then ran from
    56.5 to 58.2. peterpt read exhaustion off an oversized ATR candle — and the
    prompt had NO field expressing that. Worse, its priority line says
    "ATR = volatility only", explicitly instructing the model NOT to reason
    directionally from the one number that came closest. So the AI was told to
    disregard exactly the evidence that pointed the other way.

    A candle running several times ATR that closes far from its extreme is a
    climax, not a trend continuation. Both parts matter: size alone is just a
    big move; size PLUS a close rejected back off the extreme is exhaustion.

    Deterministic arithmetic on candles already loaded — no new data source, no
    heuristic the AI cannot overrule. It reports what the candle did and lets
    the model weigh it.

    Returns "" when there is nothing notable to say, so a normal candle costs
    zero tokens.
    """
    try:
        if not atr or atr <= 0:
            return ""
        cs = _recent_candles(sym, tf=tf, n=2)
        if len(cs) < 1:
            return ""
        c = cs[-1]
        hi = float(c.get("h")); lo = float(c.get("l"))
        cl = float(c.get("c", c.get("close", 0)) or 0)
        op = float(c.get("o", c.get("open", cl)) or cl)
        rng = hi - lo
        if rng <= 0 or cl <= 0:
            return ""
        mult = rng / float(atr)
        if mult < 2.0:
            return ""            # unremarkable candle — say nothing
        off_low  = (cl - lo) / rng * 100.0
        off_high = (hi - cl) / rng * 100.0
        direction = "down" if cl < op else "up"
        # A big DOWN candle closing well off its low = sellers rejected.
        # A big UP candle closing well off its high = buyers rejected.
        if direction == "down" and off_low >= 60.0:
            note = "closed %.0f%% off its low — downside rejected" % off_low
        elif direction == "up" and off_high >= 60.0:
            note = "closed %.0f%% off its high — upside rejected" % off_high
        elif direction == "down":
            note = "closed %.0f%% off its low" % off_low
        else:
            note = "closed %.0f%% off its high" % off_high
        return "Last %s candle: %.1fx ATR range, %s %s\n" % (tf, mult, direction, note)
    except Exception:
        return ""


def _atr(sym: str, tf: str = "15min", period: int = 14) -> Optional[float]:
    """ATR (volatility, not direction) for the gate prompt (peterpt 2026-07-28,
    ChatGPT's ATR input). Standard true-range average over `period` bars."""
    candles = _recent_candles(sym, tf=tf, n=period + 2)
    if len(candles) < period + 1:
        return None
    trs = []
    prev_close = None
    for c in candles:
        try:
            h = float(c.get("h", c.get("c")))
            l = float(c.get("l", c.get("c")))
            cl = float(c.get("c"))
        except Exception:
            continue
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = cl
    if len(trs) < period:
        return None
    atr = sum(trs[-period:]) / period
    return round(atr, 6)


def build_gate_prompt(engine: str, sym: str, direction: str, entry_price: float,
                      prices: dict, extra_context: Optional[str] = None) -> str:
    """The gate prompt (peterpt 2026-07-26 spec): tell the AI the engine's
    proposed trade + the 4h swings + sentiment(age) + MACD state + nearest
    neural levels + current price, and ask for a -1..+1 rating on THIS trade.

    extra_context (peterpt 2026-07-29): optional line injected into the prompt —
    e.g. when E2 fires from a detected chart pattern, the pattern name/direction
    is passed so the AI knows WHY the engine wants the trade (e.g. 'Detected
    pattern: bullish engulfing (buy)'). Kept short to preserve the token budget."""
    px = _price_of(prices, sym) or entry_price
    # MACD needs >=26 bars for its 26-EMA + 9 signal, so fetch 40. Swings are
    # the recent-journey read, so limit those to the last ~17 bars (=4h).
    candles = _recent_candles(sym, tf="15min", n=40)
    swings = _swing_map(candles[-17:], length=2)
    swing_txt = ", ".join(
        f"{s['type']}={s['price']}(vol {s['volume']})" for s in swings
    ) or "no clear swings"
    sent = _sentiment_with_age(sym)
    sent_txt = (f"{sent['value']} (age {sent['age_secs']}s)"
                if sent else "none")
    macd_txt = _macd_state(candles)   # now has 40 bars — enough for MACD
    neu = _closest_neurals(sym, px)
    htf = _htf_trend(sym)
    mtf = _month_trend(sym)
    # Multi-TF dirs computed once, used for both the stack line and alignment %.
    _t15 = _tf_dir(sym, "15min", 16, 0.15)
    _t1h = _tf_dir(sym, "1hour", 12, 0.30)
    _t4h = _tf_dir(sym, "4hour", 12, 0.60)
    _t1d = _tf_dir(sym, "1day", 10, 1.00)
    tf_stack = f"15m {_t15} | 1h {_t1h} | 4h {_t4h} | 1d {_t1d}"
    # Explicit bull/bear/neutral counts (ChatGPT 2026-07-29: "67% feels
    # arbitrary" — show the actual tally so the AI knows exactly how aligned).
    _dirs = [s.split()[0] for s in (_t15, _t1h, _t4h, _t1d)]
    _bull = _dirs.count("UP")
    _bear = _dirs.count("DOWN")
    _neut = _dirs.count("flat")
    align_line = f"TF alignment: {_bull} bull / {_bear} bear / {_neut} neutral\n"
    atr = _atr(sym)
    # ── price formatting (peterpt + Claude 2026-08-03) ──────────────────────
    # Raw floats were emitted at full repr precision: "buy 4038.31052666" on a
    # metal quoted to one decimal, "ATR: 4.815714". Two costs, and the second
    # matters more than the tokens:
    #   1) ~25 wasted chars per call, on every engine wake-up, on 6 instruments.
    #   2) Spurious precision reads as MEANINGFUL precision. The model cannot
    #      tell that 4038.31052666 is a computed fractal artifact while 4075.7
    #      is an observed, tested level — the longer number looks better
    #      measured. Every other figure in this prompt is already rounded to a
    #      sensible scale; these were the outliers.
    # Same scale ladder neural_reliability.summary_line uses, so a level prints
    # identically in both places and the AI sees one consistent format.
    def _pxs(v) -> str:
        try:
            v = float(v)
        except Exception:
            return str(v)
        _d = (5 if abs(v) < 1 else 4 if abs(v) < 10 else 3 if abs(v) < 100
              else 2 if abs(v) < 1000 else 1)
        return f"{v:.{_d}f}"

    atr_txt = _pxs(atr) if atr is not None else "n/a"
    # Exhaustion read on the last closed candle (empty when unremarkable).
    exh_line = _exhaustion(sym, atr)
    session = _market_session()
    note = _instrument_note(sym)
    note_line = f"Instrument note: {note}\n" if note else ""
    _spr = _spread_pct(prices, sym)
    spread_line = f"Spread: {_spr:.3f}%\n" if _spr is not None else ""
    # VWAP distance (ChatGPT #2): where price sits vs volume-weighted average.
    _vwap = _vwap_from_candles(sym)
    vwap_line = ""
    if _vwap and px and px > 0:
        _vd = (px - _vwap) / _vwap * 100.0
        _side = "above" if _vd >= 0 else "below"
        vwap_line = f"VWAP: {_vwap:.2f} (price {_vd:+.2f}% {_side})\n"
    _trig = _engine_trigger(engine)
    trig_line = f"{_trig}\n" if _trig else ""
    # AI-PRICE MODE (peterpt 2026-08-02): when "Allow AI price suggestion" is
    # enabled, give the AI the STRONGEST confirmed neurals near current price so it
    # can anchor a suggested ENTRY on a proven level instead of guessing. Only
    # added when the toggle is ON — otherwise the prompt is unchanged (feature
    # inert by default). Read the toggle from config here so the signature stays.
    aiprice_line = ""
    try:
        _gs = {}
        if os.path.isfile(GUI_SETTINGS_PATH):
            with open(GUI_SETTINGS_PATH, "r", encoding="utf-8") as _f:
                _gs = json.load(_f)
        _use_ai = bool(_gs.get("e5_use_ai_price", False))
    except Exception:
        _use_ai = False
    aiprice_rule = ""
    if _use_ai:
        try:
            import neural_reliability as _nr
            _strong = _nr.strongest_near_price(sym, px, band_pct=3.0, limit=4)
            if _strong:
                aiprice_line = ("Strongest nearby neurals (for entry-price choice):\n"
                                + "\n".join("  " + s for s in _strong) + "\n")
        except Exception:
            aiprice_line = ""
        # ── ASK for the price (peterpt + Claude 2026-08-03) ─────────────────
        # Without this the feature is inert. The prompt used to offer the AI a
        # "price" field in the reply schema and NOTHING else — no statement that
        # suggesting an entry was allowed, and no constraints. A model shown a
        # null field with no instruction simply leaves it null, so the AI-price
        # path almost never fired; and on the rare occasion it did, the number
        # was unconstrained and _validate_ai_price would often reject it, which
        # looks identical to "the AI didn't suggest one".
        #
        # The stated rule matches the validator's LIMIT rule (buy at/below
        # current, sell at/above) — deliberately the STRICTER of the two, since
        # E1's LEVEL mode accepts a superset. So anything the AI is told to
        # produce here is valid for every engine.
        try:
            _maxd = float(_gs.get("e5_ai_price_max_dist_pct", 2.0) or 2.0)
        except Exception:
            _maxd = 2.0
        aiprice_rule = (
            f"Entry price: you MAY name a better entry than {px} in \"price\" "
            f"(else null to accept the engine's level). A BUY entry must be at "
            f"or below {px}; a SELL entry at or above {px}; either way within "
            f"{_maxd}% of {px}. Prefer anchoring it on one of the tested levels "
            f"listed above — a level that has held repeatedly is a better entry "
            f"than a round number.\n")

    dir_u = str(direction).upper()
    buy_n = neu['closest_buy'] if neu['closest_buy'] is not None else "n/a"
    sell_n = neu['closest_sell'] if neu['closest_sell'] is not None else "n/a"

    # Neural reliability (peterpt/ChatGPT 2026-07-29, "huge"): how well these
    # levels have HELD over the rolling 3-day window. Read through the accessor
    # so storage can later move to SQLite without touching E5. Line omitted when
    # no history yet (graceful — appears once data accumulates).
    rel_line = ""
    try:
        import neural_reliability as _nr
        _parts = []
        # De-duplicate against the strongest-nearby block (peterpt + Claude
        # 2026-08-03). When e5_use_ai_price is ON, aiprice_line already lists
        # the 4 strongest levels within 3% of price — and the fractal level
        # being traded is usually one of them, so the identical summary line
        # was printed twice (~110 wasted chars per call, and repetition in a
        # prompt can also over-weight whatever is repeated).
        # Both come from neural_reliability.summary_line() for the same level,
        # so the strings match exactly apart from the block's leading indent.
        # When the toggle is OFF aiprice_line is "" and nothing is dropped —
        # the prompt is unchanged in that state.
        def _add(_s):
            if _s and _s not in aiprice_line:
                _parts.append(_s)
        if isinstance(buy_n, (int, float)):
            _add(_nr.summary_line(sym, float(buy_n), "support"))
        if isinstance(sell_n, (int, float)):
            _add(_nr.summary_line(sym, float(sell_n), "resistance"))
        if _parts:
            rel_line = "\n".join(_parts) + "\n"
    except Exception:
        rel_line = ""

    # Relative distances + range position (peterpt/ChatGPT 2026-07-29, #2/#4).
    # GPT reasons better on RELATIVE distances ("0.5% from support") than raw
    # numbers it must subtract. Pure arithmetic on values already computed — no
    # new data. Position = how far price sits between the buy/sell fractals
    # (0%=at support, 100%=at resistance).
    dist_txt = ""
    rr_txt = ""
    try:
        if isinstance(buy_n, (int, float)) and isinstance(sell_n, (int, float)) \
                and px and px > 0:
            d_buy = (px - float(buy_n)) / px * 100.0
            d_sell = (float(sell_n) - px) / px * 100.0
            span = float(sell_n) - float(buy_n)
            pos = ((px - float(buy_n)) / span * 100.0) if span > 0 else 50.0
            pos = max(0.0, min(100.0, pos))
            _closer = "BUY" if pos < 50 else "SELL"
            dist_txt = (f"Distance: {d_buy:+.2f}% above buy-level, "
                        f"{d_sell:+.2f}% below sell-level\n"
                        f"Price location: {pos:.0f}% of BUY→SELL range "
                        f"(closer to {_closer} neural)\n")
            # Risk/Reward (ChatGPT #4): for the proposed direction, the opposite
            # fractal is the natural target and the near one the stop. R:R lets
            # the AI reject "good setup, poor reward". Uses the fractal levels
            # already computed — no extra data.
            if str(direction).lower() == "buy":
                risk = abs(px - float(buy_n)); reward = abs(float(sell_n) - px)
            else:
                risk = abs(float(sell_n) - px); reward = abs(px - float(buy_n))
            if risk > 0:
                rr = reward / risk
                _rr_note = ("favorable" if rr >= 1.5 else
                            "risk exceeds reward" if rr < 1.0 else "modest")
                rr_txt = (f"Risk/Reward: ~{rr:.2f} ({_rr_note}; "
                          f"risk {risk/px*100:.2f}%, reward {reward/px*100:.2f}%)\n")
    except Exception:
        dist_txt = ""; rr_txt = ""

    # Volume ratio (ChatGPT #4): current candle vol vs recent average. High
    # volume on a move = conviction; low = weak. Uses candles already fetched.
    vol_txt = ""
    try:
        vols = [float(c.get("v")) for c in candles
                if isinstance(c, dict) and c.get("v") is not None]
        if len(vols) >= 6 and vols[-1] > 0:
            avg = sum(vols[:-1]) / max(1, len(vols) - 1)
            if avg > 0:
                ratio = vols[-1] / avg
                vol_txt = f"Volume: {ratio:.2f}x recent average\n"
    except Exception:
        vol_txt = ""

    # Optional extra context (e.g. detected chart pattern from E2). One short
    # line so the AI knows WHY the engine wants this trade (peterpt 2026-07-29).
    ctx_line = ""
    if extra_context:
        ctx_line = f"{str(extra_context).strip()}\n"

    # Tight prompt — minimal tokens per request (peterpt 2026-07-28). The
    # engine name is NOT sent (it was only an example); the AI judges the
    # instrument's data on its own merits, direction-agnostic.
    return (
        f"Evaluate a proposed {dir_u} trade on {sym} @ {px}. "
        f"The direction comes from a trading engine — assess independently; "
        f"if the evidence disagrees, choose the opposite or neutral.\n"
        f"{trig_line}"
        f"HTF trend: {htf}\n"
        f"1-month trend: {mtf}\n"
        f"Multi-TF trend: {tf_stack}\n"
        f"{align_line}"
        f"4h swings(price,vol): {swing_txt}\n"
        f"Sentiment: {sent_txt}\n"
        f"MACD: {macd_txt}\n"
        f"ATR: {atr_txt}\n"
        f"{exh_line}"
        f"Session: {session}\n"
        f"{note_line}"
        f"Nearest fractal levels: buy {_pxs(buy_n)}, sell {_pxs(sell_n)}\n"
        f"{rel_line}"
        f"{aiprice_line}"
        f"{dist_txt}"
        f"{vol_txt}"
        f"{spread_line}"
        f"{vwap_line}"
        f"{rr_txt}"
        f"{ctx_line}"
        f"Priority: HTF/1M > Multi-TF > VWAP/Neurals > MACD/Volume. ATR = volatility "
        f"only, but an exhaustion candle may override a continuation read.\n"
        f"Choose buy, sell, or neutral. Confidence 0..1 (never negative) for your "
        f"chosen direction. No clear edge => neutral, 0. Reasoning <=12 words.\n"
        f"{aiprice_rule}"
        f'Reply ONLY: {{"direction":"","confidence":0,"price":null,"reasoning":""}}'
    )


def parse_gate(raw: str) -> Optional[dict]:
    """Parse the gate response: {direction 'buy'|'sell'|'neutral', confidence
    -1..1, reasoning}. 'neutral' means the AI declines the trade. Returns None
    if unparseable (→ engine cancels, fail-closed).

    Robust to: markdown code fences, prose before/after the JSON, and empty
    responses (peterpt 2026-07-28)."""
    if not raw or not str(raw).strip():
        return None
    t = str(raw).strip()
    # strip markdown fences
    if t.startswith("```"):
        parts = t.split("```")
        t = (parts[-2] if len(parts) >= 3 else t).replace("json", "", 1).strip()
    # try direct parse first
    d = None
    try:
        d = json.loads(t)
    except Exception:
        # extract the first {...} block from mixed prose+json
        import re
        m = re.search(r"\{[^{}]*\}", t, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if not isinstance(d, dict):
        return None
    direction = str(d.get("direction", "")).lower().strip()
    if direction not in ("buy", "sell", "neutral"):
        return None
    try:
        conf = float(d.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    # Confidence is now a 0..1 conviction in the chosen `direction` (peterpt
    # 2026-07-29). If the AI still emits a negative (old-style sign convention),
    # its magnitude is the conviction — the sign was only ever meant to indicate
    # direction, which `direction` already carries. So fold to absolute value.
    conf = abs(conf)
    conf = max(0.0, min(1.0, conf))     # clamp 0..1
    # Optional entry price the AI suggests (null/absent = use the neural price).
    price = None
    if d.get("price") is not None:
        try:
            price = float(d.get("price"))
            if price <= 0:
                price = None
        except Exception:
            price = None
    return {"direction": direction, "confidence": round(conf, 3),
            "price": price,
            "reasoning": str(d.get("reasoning", ""))[:400]}


def _validate_ai_price(ai_price, current_price, ai_dir, toggle_on: bool,
                       max_dist_pct: float = 2.0, order_type: str = "LIMIT"):
    """Decide whether the AI's suggested entry price is safe/sane to use
    (peterpt 2026-07-31). Returns a dict:
      {"use": bool, "price": float|None, "reason": str}

    Guards (ALL must pass to use the AI price):
      • the toggle (e5_use_ai_price) is ON
      • ai_price is a real number, > 0
      • it's on the CORRECT side: a BUY suggestion must be <= current (buy the
        dip), a SELL must be >= current (sell the rally). A buy ABOVE current or
        sell BELOW makes no sense as a limit and is rejected.
      • it's within max_dist_pct of current — reject wild/hallucinated prices
        (e.g. 'buy at 3500' when price is 4000).
    Even with the toggle OFF this is computed so the observation log can show
    what WOULD have been used. Execution wiring is a later, deliberate step."""
    if ai_price is None or current_price is None:
        return {"use": False, "price": None, "reason": "no AI price"}
    try:
        aip = float(ai_price); cur = float(current_price)
    except Exception:
        return {"use": False, "price": None, "reason": "unparseable price"}
    if aip <= 0 or cur <= 0:
        return {"use": False, "price": None, "reason": "non-positive price"}

    dist_pct = abs(aip - cur) / cur * 100.0
    if dist_pct > max_dist_pct:
        return {"use": False, "price": aip,
                "reason": f"too far ({dist_pct:.2f}% > {max_dist_pct}%)"}

    # ── side check depends on the ORDER TYPE (peterpt + Claude 2026-08-03) ──
    # The original check assumed every order is a LIMIT. E1 also places STOP
    # working orders, and for those the correct side is the OPPOSITE:
    #
    #   LIMIT  BUY  fills when ask FALLS to level  -> level must be <= current
    #   LIMIT  SELL fills when bid RISES to level  -> level must be >= current
    #   STOP   BUY  fills when ask RISES to level  -> level must be >= current
    #   STOP   SELL fills when bid FALLS to level  -> level must be <= current
    #
    # E1's case mapping (from the trader's PLACE handler):
    #   Case A SELL touched from below -> SELL STOP  at price-T  (below current)
    #   Case B SELL touched from above -> SELL LIMIT at N+T      (above current)
    #   Case C BUY  touched from above -> BUY  STOP  at price+T  (above current)
    #   Case D BUY  touched from below -> BUY  LIMIT at N-T      (below current)
    #
    # Cases A and C are exactly what the old LIMIT-only rule rejected. Left
    # unfixed, routing E1 through this validator would silently kill AI prices
    # for half its touch cases, and the log would say "invalid limit" about an
    # order that was never a limit — looking like the AI simply stopped
    # suggesting prices. E2/E3/E4 rest LIMIT orders, so they are unaffected.
    d  = str(ai_dir).lower()
    ot = str(order_type or "LIMIT").upper().strip()
    if ot not in ("LIMIT", "STOP", "LEVEL"):
        ot = "LIMIT"

    # ── "LEVEL": no side check (peterpt + Claude 2026-08-03) ────────────────
    # E1 does NOT use the AI price as an order level. It uses it as a REFERENCE
    # LEVEL replacing the thinker's neural, then places at (level -/+ offset)
    # and TRAILS from there (pt_engine1 ~2910: SELL STOP at neural-offset trail
    # up, BUY STOP at neural+offset trail down).
    #
    # So no side rule applies to the AI's number itself. peterpt's own case —
    # E1 proposes SELL at 4000, price 3997, AI answers "sell at 4010" — is
    # exactly the intent (sell higher), yet it fails a naive STOP check and a
    # naive LIMIT check alike, because 4010 is not where the order goes: the
    # order goes at 4010-offset and then trails up.
    #
    # LEVEL therefore keeps the checks that DO apply — positive number, within
    # max_dist_pct of current, toggle on — and drops the one that does not.
    # The engine's own offset/trail logic governs the actual fill side.
    if ot == "LEVEL":
        pass
    elif ot == "LIMIT":
        if d == "buy" and aip > cur:
            return {"use": False, "price": aip,
                    "reason": "buy above current (invalid LIMIT)"}
        if d == "sell" and aip < cur:
            return {"use": False, "price": aip,
                    "reason": "sell below current (invalid LIMIT)"}
    else:
        if d == "buy" and aip < cur:
            return {"use": False, "price": aip,
                    "reason": "buy below current (invalid STOP)"}
        if d == "sell" and aip > cur:
            return {"use": False, "price": aip,
                    "reason": "sell above current (invalid STOP)"}

    # passes all sanity checks — usable IF the toggle is on
    if not toggle_on:
        return {"use": False, "price": aip, "reason": "valid but toggle off"}
    return {"use": True, "price": aip,
            "reason": f"AI entry {dist_pct:.2f}% from current ({ot})"}


def e5_gate(engine: str, sym: str, direction: str, entry_price: float,
            hub_dir: Optional[str] = None,
            extra_context: Optional[str] = None,
            order_type: str = "LIMIT",
            gap_mode: bool = False) -> dict:
    """THE GATE (peterpt 2026-07-26). An engine calls this before sending a
    signal to the trader, when its ai_trade_decision switch is ON.

    Flow: assemble the gate feature vector → ask the AI for a -1..+1 rating on
    THIS proposed trade → compare against the GLOBAL threshold in config →
    return YES/NO.

      returns {"decision": "YES"|"NO", "rating": float|None,
               "threshold": float, "reasoning": str, "reason": str}

    FAIL-CLOSED (peterpt's rule): if the AI is unavailable, rate-limited, quota-
    blocked, errors, or returns an unparseable answer → decision is "NO". When
    the engine asked for an AI decision and the AI cannot give one, the trade is
    NOT taken.

    The global threshold lives in gui_settings as e5_decision_threshold (one
    number governing ALL engines — E5 is the single decider). rating >=
    threshold → YES; below → NO. Every call is logged to engine5_reads.jsonl
    (tagged kind='gate') so the dashboard shows it.
    """
    global HUB_DIR, RAMFS_DIR, CANDLES_DIR
    if hub_dir:
        HUB_DIR = os.path.abspath(hub_dir)
        RAMFS_DIR = os.path.join(HUB_DIR, "ramfs")
        CANDLES_DIR = os.path.join(HUB_DIR, "candles")

    gs = load_settings()
    cfg = e5_config(gs)
    threshold = float(gs.get("e5_decision_threshold", 0.3))
    _sent = {"prompt": None, "raw": None, "price": None}   # filled once the call is made

    def _verdict(decision, ai_dir, confidence, reasoning, reason):
        rec = {
            "ts": round(time.time(), 3), "kind": "gate", "engine": engine,
            "symbol": sym, "direction": str(direction).lower(),
            "ai_direction": ai_dir, "entry_price": entry_price,
            "confidence": confidence, "threshold": threshold,
            "decision": decision, "reasoning": reasoning, "reason": reason,
            "ai_price": _sent["price"],     # price the AI suggested (may be None)
            "provider": cfg["provider"],
            "prompt": _sent["prompt"],      # exact prompt sent (None if no call)
            "ai_raw": _sent["raw"],         # raw AI response text
        }
        try:
            _write_read(rec)
        except Exception:
            pass
        # Schedule +15m/+30m/+1h price follow-ups for accuracy scoring.
        # peterpt 2026-07-29 spec: score the AI's DIRECTIONAL OPINION, not just
        # executed trades. Whenever the AI expressed a direction (buy or sell) —
        # even if the trade was CANCELLED for low/negative confidence, or FLIPPED
        # — we track whether that opinion was right (did price move its way?).
        # Only a genuine NEUTRAL (ai_dir == "neutral", conf 0.0) is left untracked
        # ("-"), since it expressed no directional view to score.
        if ai_dir in ("buy", "sell"):
            base_px = _sent.get("price") or _price_of(_live_prices(), sym) \
                or entry_price
            try:
                _schedule_followup(rec["ts"], sym, ai_dir, base_px)
            except Exception:
                pass
        log(f"[E5-GATE] {engine} wanted {str(direction).upper()} {sym} @ "
            f"{entry_price} → AI says {str(ai_dir).upper() if ai_dir else '—'} "
            f"conf={confidence}"
            + (f" @ {_sent['price']}" if _sent.get("price") else "")
            + f" vs thr={threshold} → {decision}"
            + (f" ({reason})" if reason else ""))
        return {"decision": decision, "direction": ai_dir,
                "confidence": confidence, "threshold": threshold,
                "reasoning": reasoning, "reason": reason,
                "ai_price": _sent["price"],
                "use_ai_price": _validate_ai_price(
                    _sent["price"], entry_price, ai_dir,
                    # Gap mode forces the AI-price ON for THIS evaluation only —
                    # it does not change the stored e5_use_ai_price toggle or
                    # affect any other engine/call. A weekend gap is the critical
                    # case where the neural is stale and we MUST use a fresh AI
                    # price. Outside gap mode the real toggle applies unchanged.
                    True if gap_mode else cfg.get("use_ai_price", False),
                    max_dist_pct=(0.2 if gap_mode else
                                  float(gs.get("e5_ai_price_max_dist_pct", 2.0) or 2.0)),
                    order_type=order_type),
                "prompt": _sent["prompt"], "ai_raw": _sent["raw"]}

    # DEFENSIVE INTERLOCK: if E5's master switch is OFF, refuse regardless.
    if not cfg["enabled"]:
        return _verdict("CANCEL", None, None, "", "E5 disabled — gate inactive")
    # Fail-closed: quota-blocked → CANCEL.
    if _quota_blocked():
        return _verdict("CANCEL", None, None, "", "AI quota-blocked (fail-closed)")

    prices = _live_prices()
    provider = cfg["provider"]
    key = cfg["keys"].get(provider, "")
    # ── PLACEHOLDER-KEY PRE-CHECK (peterpt + Claude 2026-08-06) ──────────────
    # _resolve_key already returns "" for a missing or "YOUR_..." placeholder
    # key. Calling ask_ai with it wastes a full prompt build and, worse, logs a
    # junk "no-read: missing or placeholder API key" row in the decisions table
    # every cycle. Skip cleanly BEFORE building the prompt: no prompt recorded,
    # no AI call, no no-read row. This is not the same as E5-disabled — it means
    # "E5 is on but this provider has no usable key", which is a config problem
    # the user fixes by pasting a key, not a decision worth logging repeatedly.
    if not key:
        return _verdict("CANCEL", None, None, "",
                        f"no API key for {provider} — set one in Config "
                        f"(gate skipped, not logged as a read)")
    prompt = build_gate_prompt(engine, sym, direction, entry_price, prices,
                               extra_context=extra_context)
    _sent["prompt"] = prompt                 # record what we send
    res = providers.ask_ai(provider, key, cfg["model"], prompt)
    _sent["raw"] = res.get("text")           # and what came back

    if not res.get("ok"):
        if res.get("quota"):
            _trip_quota(provider)
        return _verdict("CANCEL", None, None, "",
                        f"AI unavailable: {res.get('error')} (fail-closed)")

    _clear_quota()
    parsed = parse_gate(res.get("text", ""))
    if parsed is None:
        return _verdict("CANCEL", None, None, "",
                        "AI response unparseable (fail-closed)")

    ai_dir = parsed["direction"]
    conf = parsed["confidence"]
    reasoning = parsed.get("reasoning", "")
    _sent["price"] = parsed.get("price")     # AI's suggested entry (may be None)

    # THE DECISION (peterpt 2026-07-26 spec):
    #   • AI says "neutral"        → CANCEL (AI explicitly declines the trade)
    #   • confidence < threshold   → CANCEL (not sure enough — e.g. 0.1, or any
    #                                 low/negative value; "ignore the signal")
    #   • confidence >= threshold, ai_dir == engine's direction → PROCEED
    #   • confidence >= threshold, ai_dir opposite → FLIP (trade AI's side)
    # The AI's judgement is authoritative on DIRECTION — the engine's proposed
    # direction was only the trigger; if the AI favours the other side with
    # enough confidence, the trade flips to the AI's direction.
    if ai_dir == "neutral":
        return _verdict("CANCEL", "neutral", conf, reasoning,
                        "AI judged neutral — no trade")
    if conf < threshold:
        return _verdict("CANCEL", ai_dir, conf, reasoning,
                        f"confidence {conf} below threshold {threshold}")
    if ai_dir == str(direction).lower():
        return _verdict("PROCEED", ai_dir, conf, reasoning, "")
    return _verdict("FLIP", ai_dir, conf, reasoning,
                    f"AI favours {ai_dir.upper()} over engine's {str(direction).upper()}")


def build_prompt(feat: dict) -> str:
    """A tightly-constrained prompt: fixed menu, capped confidence, reasoning
    must cite only the given features, permission to abstain. This is the
    'don't be crazy' guardrail set, in the request itself."""
    return (
        "You are a trading-setup reviewer. You are given pre-computed signals for "
        "one instrument at a decision moment. Do NOT predict future prices. Judge "
        "only whether THIS is a good moment to act, using ONLY the fields provided.\n\n"
        "Field notes:\n"
        "- range_4h: the high/low corridor price traveled in the last 4 hours; "
        "'position' is where current price sits in it (0=at the low, 1=at the high).\n"
        "- swings_4h: turning points over 4h, each with the VOLUME traded there — "
        "a swing with high volume is a level price fought at (liquidity/interest); "
        "these often act as support/resistance.\n"
        "- structure_choch: CHoCH=trend reversal confirmed, BOS=continuation.\n"
        "- regime: 'choppy' means momentum signals are unreliable.\n"
        "- sentiment_estimate: itself an AI guess (low reliability) — weight lightly.\n\n"
        f"SIGNALS:\n{json.dumps(feat, indent=2)}\n\n"
        "Rules:\n"
        "- Choose exactly one read: \"buy\", \"sell\", \"neutral\", or \"avoid\".\n"
        "- confidence is 0.0-1.0. Never exceed 0.7 unless multiple signals clearly agree.\n"
        "- If signals conflict or are insufficient, return read \"neutral\" with low confidence.\n"
        "- reasoning must reference only the provided fields; invent nothing.\n\n"
        "Respond ONLY with valid JSON, no prose, no markdown:\n"
        '{"read": "buy|sell|neutral|avoid", "confidence": <float 0-1>, '
        '"reasoning": "<one sentence citing the signals>"}'
    )


def parse_read(raw: str) -> Optional[dict]:
    """Defensive parse — strips code fences, json.loads, validates the menu and
    clamps confidence. Returns None if it can't be trusted (an unparseable AI
    answer is treated as 'no read', not guessed at)."""
    if not raw:
        return None
    t = raw.strip()
    if t.startswith("```"):
        # take the fenced body
        parts = t.split("```")
        t = parts[-2] if len(parts) >= 3 else t
        t = t.replace("json", "", 1).strip()
    try:
        d = json.loads(t)
    except Exception:
        return None
    read = str(d.get("read", "")).lower().strip()
    if read not in ("buy", "sell", "neutral", "avoid"):
        return None
    try:
        conf = float(d.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))          # clamp
    return {"read": read, "confidence": round(conf, 3),
            "reasoning": str(d.get("reasoning", ""))[:400]}


# ── Circuit breaker (time-based quota recovery, peterpt 2026-07-26) ───────────
# When the AI provider reports quota/rate-limit exhaustion, we stop calling so we
# don't hammer a provider that's out of quota. But the block must not be
# permanent — quotas refill. So:
#   • On trip: record the timestamp in the quota file.
#   • While blocked: retry a probe every RETRY_EVERY_S (30s) — if that probe
#     succeeds, the block clears immediately.
#   • Hard ceiling: after MAX_BLOCK_S (20 min) the block auto-clears regardless,
#     so a stale block can never strand E5 forever.
QUOTA_RETRY_EVERY_S = 30      # attempt a call again every 30s while blocked
QUOTA_MAX_BLOCK_S   = 1200    # 20 minutes — auto-clear ceiling


def _quota_state() -> dict:
    """Read the quota breaker: {blocked, tripped_at, last_retry}."""
    try:
        with open(QUOTA_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}


def _quota_blocked() -> bool:
    """True if we should SKIP calling right now. Auto-clears after the 20-min
    ceiling, and allows a retry probe every 30s (returns False to let one call
    through, updating last_retry)."""
    st = _quota_state()
    if not st or not st.get("tripped_at"):
        return False
    now = time.time()
    tripped = float(st.get("tripped_at", 0))
    # Hard ceiling: 20 min elapsed → clear the block entirely.
    if now - tripped >= QUOTA_MAX_BLOCK_S:
        _clear_quota()
        log("[E5] quota block auto-cleared after 20 min — resuming AI reads")
        return False
    # Retry window: if 30s since last probe, allow ONE call through to test.
    last_retry = float(st.get("last_retry", tripped))
    if now - last_retry >= QUOTA_RETRY_EVERY_S:
        st["last_retry"] = now
        try:
            with open(QUOTA_FILE, "w", encoding="utf-8") as f:
                json.dump(st, f)
        except Exception:
            pass
        log("[E5] quota retry probe (every 30s) — attempting one AI read")
        return False          # let this call through as a probe
    return True               # still blocked, within the 30s window


def _trip_quota(provider: str) -> None:
    """Record a quota trip with timestamp (starts the 20-min ceiling + 30s
    retry cadence). If already tripped, keep the original tripped_at so the
    ceiling counts from the FIRST failure, not each retry."""
    st = _quota_state()
    now = time.time()
    if not st.get("tripped_at"):
        st = {"tripped_at": now, "last_retry": now, "provider": provider}
    else:
        # a retry probe failed again — refresh last_retry, keep tripped_at
        st["last_retry"] = now
        st["provider"] = provider
    try:
        with open(QUOTA_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:
        pass


def _clear_quota() -> None:
    try:
        if os.path.isfile(QUOTA_FILE):
            os.remove(QUOTA_FILE)
    except Exception:
        pass


# ── The read + log ────────────────────────────────────────────────────────────
def do_read(sym: str, trigger: str, cfg: dict, prices: dict) -> Optional[dict]:
    """Assemble features → ask the selected AI → parse → LOG. Test mode: never
    emits a trade signal. Returns the logged record (or None if skipped)."""
    if _quota_blocked():
        return None
    feat = assemble_features(sym, trigger, prices)
    if feat is None:
        return None

    provider = cfg["provider"]
    key = cfg["keys"].get(provider, "")
    # Same placeholder-key pre-check as the gate path: a missing/placeholder key
    # should not build a prompt or emit a no-read row every cycle. Return None
    # (no read) rather than logging the failure. (peterpt + Claude 2026-08-06)
    if not key:
        return None
    prompt = build_prompt(feat)

    res = providers.ask_ai(provider, key, cfg["model"], prompt)
    rec = {
        "ts":        round(time.time(), 3),
        "symbol":    sym,
        "trigger":   trigger,
        "provider":  provider,
        "price":     feat.get("price"),
        "features":  feat,
    }
    if not res.get("ok"):
        rec["read"] = None
        rec["error"] = res.get("error")
        if res.get("quota"):
            _trip_quota(provider)
            log(f"[E5] {sym}: {provider} quota tripped — pausing AI reads")
        else:
            log(f"[E5] {sym}: AI error — {res.get('error')}")
    else:
        _clear_quota()
        parsed = parse_read(res.get("text", ""))
        if parsed is None:
            rec["read"] = None
            rec["error"] = "unparseable AI response"
            rec["raw"] = (res.get("text") or "")[:200]
            log(f"[E5] {sym}: AI response unparseable — logged as no-read")
        else:
            rec.update(parsed)
            rec["latency_ms"] = res.get("latency_ms")
            rec["model"] = res.get("model")
            log(f"[E5] {sym}: {trigger} → AI says {parsed['read'].upper()} "
                f"(conf {parsed['confidence']:.2f}) — {parsed['reasoning'][:80]} "
                f"[TEST MODE — logged, no action]")

    _write_read(rec)
    return rec


# ===========================================================================
# ─── AI SCORECARD — was the AI actually right? (peterpt + Claude 2026-08-03) ──
# engine5_reads.jsonl already holds every decision plus the +15m/+30m/+1h prices
# and a yes/no verdict patched in once an hour has passed. That is real ground
# truth, and it answers "is the AI any good" far better than the two or three
# trades anyone happens to remember.
#
# Lives HERE rather than in a companion module (peterpt: "we already have 17
# scripts running at startup, the last thing I need is another one"). pt_api's
# /api/e5_scorecard imports this function, so the dashboard panel and the CLI
# share ONE implementation — reimplementing the maths in pt_api is exactly how
# powertrader.py and pt_api came to disagree about enable_engine3 and silently
# stopped E3 from launching.
#
#     python3 pt_engine5.py --scorecard
#     python3 pt_engine5.py --scorecard --days 3 --sc-symbol SILVER

def _sc_load(path: str, since_ts: float) -> list:
    rows = []
    if not os.path.isfile(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if float(r.get("ts", 0) or 0) < since_ts:
                    continue
                rows.append(r)
    except Exception:
        pass
    return rows


def _sc_dir_right(ai_dir, base, later):
    """Did price move the AI's way? None when not yet known."""
    if later is None or not base:
        return None
    if ai_dir == "sell":
        return later < base
    if ai_dir == "buy":
        return later > base
    return None


def scorecard(hub_dir: Optional[str] = None, days: float = 7.0,
              symbol: Optional[str] = None,
              engine: Optional[str] = None) -> dict:
    """Grade every scored AI decision. Read-only; never calls an AI.

    THE NUMBER THAT MATTERS is `flips`. When the AI overrides an engine exactly
    one of them is right, and the log records both opinions plus what price did.
    Below 50% the AI is destroying signal, not adding it.

    `horizon` separates a direction problem from an exit problem: a decision can
    be right at +15m and wrong at +1h, which no prompt change will fix.
    """
    _hub = os.path.abspath(hub_dir or HUB_DIR)
    path = os.path.join(_hub, "engine5_reads.jsonl")
    rows = _sc_load(path, time.time() - float(days) * 86400.0)
    if symbol:
        rows = [r for r in rows
                if str(r.get("symbol", "")).upper() == str(symbol).upper()]
    if engine:
        rows = [r for r in rows
                if str(r.get("engine", "")).upper() == str(engine).upper()]

    scored = [r for r in rows if r.get("accurate") in ("yes", "no")]
    pending = [r for r in rows if r.get("accurate") not in ("yes", "no")
               and r.get("ai_direction") in ("buy", "sell")]
    n = len(scored)
    ok = sum(1 for r in scored if r["accurate"] == "yes")

    def _rate(a, b):
        return int(round(100.0 * a / b)) if b else None

    flips = [r for r in scored if str(r.get("decision", "")).upper() == "FLIP"]
    ai_won = sum(1 for r in flips if r["accurate"] == "yes")

    horizon = {}
    for k in ("p15m", "p30m", "p1h"):
        good = tot = 0
        for r in scored:
            v = _sc_dir_right(r.get("ai_direction"),
                              r.get("entry_price") or 0, r.get(k))
            if v is None:
                continue
            tot += 1
            good += 1 if v else 0
        horizon[k] = {"right": good, "total": tot, "pct": _rate(good, tot)}

    calib = []
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
        sel = [r for r in scored if lo <= float(r.get("confidence") or 0) < hi]
        g = sum(1 for r in sel if r["accurate"] == "yes")
        calib.append({"lo": lo, "hi": hi, "right": g, "total": len(sel),
                      "pct": _rate(g, len(sel))})

    # Threshold sweep — peterpt 2026-08-03: "0.70 is already high enough; if I
    # set it higher the app will not trade anything all day." Correct, and the
    # trade-off is measurable: `kept` is the volume each setting costs,
    # `delta_pp` the accuracy it buys over taking every decision. If delta stays
    # near zero everywhere, confidence carries no information and the current
    # threshold throttles trading for nothing.
    base_rate = (ok / n) if n else 0.0
    sweep = []
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        sel = [r for r in scored if float(r.get("confidence") or 0) >= thr]
        g = sum(1 for r in sel if r["accurate"] == "yes")
        rate = (g / len(sel)) if sel else 0.0
        sweep.append({"threshold": thr, "kept": len(sel), "of": n,
                      "right": g, "pct": _rate(g, len(sel)),
                      "delta_pp": round((rate - base_rate) * 100.0, 1)
                      if sel else None})

    def _group(field):
        out = []
        for k in sorted({str(r.get(field, "?")) for r in scored}):
            sel = [r for r in scored if str(r.get(field, "?")) == k]
            g = sum(1 for r in sel if r["accurate"] == "yes")
            fl = [r for r in sel
                  if str(r.get("decision", "")).upper() == "FLIP"]
            fg = sum(1 for r in fl if r["accurate"] == "yes")
            out.append({"key": k, "right": g, "total": len(sel),
                        "pct": _rate(g, len(sel)), "flip_right": fg,
                        "flip_total": len(fl), "flip_pct": _rate(fg, len(fl))})
        return out

    wrong = sorted([r for r in scored if r["accurate"] == "no"],
                   key=lambda r: r.get("ts", 0), reverse=True)[:10]

    return {
        "ok": True, "path": path, "days": days,
        "logged": len(rows), "scored": n, "awaiting": len(pending),
        # Under ~20 the percentages swing wildly — the same trap as a
        # 100%-hold level with a single test. Consumers must surface this.
        "small_sample": n < 20,
        "overall": {"right": ok, "total": n, "pct": _rate(ok, n)},
        "flips": {"ai_right": ai_won, "engine_right": len(flips) - ai_won,
                  "total": len(flips), "ai_pct": _rate(ai_won, len(flips))},
        "horizon": horizon, "calibration": calib, "sweep": sweep,
        "by_symbol": _group("symbol"), "by_engine": _group("engine"),
        "ai_prices": {"with": sum(1 for r in rows if r.get("ai_price")),
                      "of": len(rows)},
        "recent_wrong": [
            {"ts": r.get("ts"), "symbol": r.get("symbol"),
             "engine": r.get("engine"), "wanted": r.get("direction"),
             "ai": r.get("ai_direction"), "confidence": r.get("confidence"),
             "decision": r.get("decision"), "entry": r.get("entry_price"),
             "p15m": r.get("p15m"), "p30m": r.get("p30m"), "p1h": r.get("p1h"),
             "reasoning": r.get("reasoning", "")} for r in wrong],
    }


def scorecard_text(d: dict) -> str:
    """Render scorecard() for the terminal."""
    if not d or not d.get("ok"):
        return "scorecard unavailable"
    def pc(v):
        return "—" if v is None else "%d%%" % v
    L = []
    L.append("=" * 72)
    L.append("E5 AI SCORECARD — last %g days" % d["days"])
    L.append(d["path"])
    L.append("=" * 72)
    L.append("logged %d · scored %d · awaiting +1h %d"
             % (d["logged"], d["scored"], d["awaiting"]))
    if d["small_sample"]:
        L.append("")
        L.append("!! only %d scored decisions — percentages will swing wildly."
                 % d["scored"])
        L.append("   A direction to look in, not a conclusion.")
    L.append("")
    L.append("OVERALL at +1h   %d/%d  %s   (50%% = coin flip)"
             % (d["overall"]["right"], d["overall"]["total"],
                pc(d["overall"]["pct"])))
    L.append("")
    f = d["flips"]
    L.append("FLIPS — AI vs ENGINE")
    L.append("  AI right     %d/%d  %s" % (f["ai_right"], f["total"],
                                           pc(f["ai_pct"])))
    L.append("  ENGINE right %d/%d" % (f["engine_right"], f["total"]))
    if f["total"] and f["ai_pct"] is not None:
        if f["ai_pct"] < 50:
            L.append("  >> AI overriding engines and being WRONG more often "
                     "than right — flipping is costing money.")
        elif f["ai_pct"] > 50:
            L.append("  >> Flips are net positive.")
        else:
            L.append("  >> Dead even.")
    L.append("")
    L.append("HORIZON")
    for k, lab in (("p15m", "+15m"), ("p30m", "+30m"), ("p1h", " +1h")):
        h = d["horizon"][k]
        L.append("  %s  %d/%d  %s" % (lab, h["right"], h["total"], pc(h["pct"])))
    e, l = d["horizon"]["p15m"]["pct"], d["horizon"]["p1h"]["pct"]
    if e is not None and l is not None:
        if e - l > 15:
            L.append("  >> Right early, wrong later — an EXIT / holding-period "
                     "problem. A better prompt will not fix it.")
        elif l - e > 15:
            L.append("  >> Wrong early, right later — entries early, view sound.")
        else:
            L.append("  >> Consistent across horizons.")
    L.append("")
    L.append("THRESHOLD SWEEP   (kept = your trade volume)")
    L.append("  %-6s %-10s %-9s %s" % ("thr", "kept", "accuracy", "vs all"))
    for r in d["sweep"]:
        cur = " <- current" if abs(r["threshold"] - 0.70) < 1e-9 else ""
        L.append("  %-6.2f %-10s %-9s %s%s"
                 % (r["threshold"], "%d/%d" % (r["kept"], r["of"]),
                    pc(r["pct"]),
                    "—" if r["delta_pp"] is None else "%+.1fpp" % r["delta_pp"],
                    cur))
    L.append("")
    for key, title in (("by_symbol", "BY INSTRUMENT"), ("by_engine", "BY ENGINE")):
        if not d[key]:
            continue
        L.append(title)
        for r in d[key]:
            L.append("  %-12s %d/%-3d %-5s  flips %d/%d %s"
                     % (r["key"], r["right"], r["total"], pc(r["pct"]),
                        r["flip_right"], r["flip_total"], pc(r["flip_pct"])))
        L.append("")
    L.append("AI named an entry price in %d of %d decisions."
             % (d["ai_prices"]["with"], d["ai_prices"]["of"]))
    return "\n".join(L)


# AI decision follow-up tracking & accuracy scoring (peterpt 2026-07-28)
# ---------------------------------------------------------------------------
# After a buy/sell gate decision, record the instrument price at +15m, +30m,
# +1h, then judge whether the AI's direction paid off (sell right if price fell,
# buy right if price rose, vs the decision-time price). Neutral/cancel decisions
# are NOT tracked — their timeframe fields stay "-".
#
# Pending follow-ups live in hub_data/engine5_followups.json; E5's main loop
# calls _process_followups() each cycle to fill any that are due and patch the
# matching row in engine5_reads.jsonl.
# ===========================================================================
FOLLOWUPS_PATH = None   # set in main() alongside the other hub paths
_FU_WINDOWS = [("p15m", 15 * 60), ("p30m", 30 * 60), ("p1h", 60 * 60)]


def _followups_path() -> str:
    return FOLLOWUPS_PATH or os.path.join(HUB_DIR, "engine5_followups.json")


def _load_followups() -> list:
    d = _rj(_followups_path())
    return d if isinstance(d, list) else []


def _save_followups(items: list) -> None:
    try:
        p = _followups_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f)
        os.replace(tmp, p)
    except Exception:
        pass


def _schedule_followup(ts: float, sym: str, ai_dir: str,
                       decision_price: float) -> None:
    """Queue the +15m/+30m/+1h price checks for a buy/sell decision. Skipped
    for neutral/cancel (ai_dir not buy/sell) or when we have no base price.

    Captures the accuracy THRESHOLD at decision time (peterpt + Claude
    2026-08-06): a call only counts as correct if price later moved the AI's way
    by more than a meaningful margin — not just any tick. The margin is
    max(spread, 0.5 x ATR) in PRICE units, both measured NOW (they describe the
    conditions the call was made under; ATR an hour later is a different number).
    Stored on the follow-up so grading is reproducible and not recomputed from
    stale live data."""
    if ai_dir not in ("buy", "sell") or not decision_price or decision_price <= 0:
        return
    # Threshold components in absolute price. ATR is already a price range;
    # spread_pct is a percentage of price, so convert to absolute.
    _thr = 0.0
    try:
        _atr_v = _atr(sym, tf="15min", period=14) or 0.0
        _sp_pct = _spread_pct(_live_prices(), sym) or 0.0
        _sp_abs = (float(_sp_pct) / 100.0) * float(decision_price)
        _thr = max(_sp_abs, 0.5 * float(_atr_v))
    except Exception:
        _thr = 0.0
    items = _load_followups()
    items.append({
        "ts": round(ts, 3), "symbol": sym, "ai_direction": ai_dir,
        "decision_price": float(decision_price),
        "move_threshold": round(_thr, 8),   # min price move to count as correct
        "p15m": None, "p30m": None, "p1h": None, "done": False,
    })
    _save_followups(items)


def _accuracy_from_prices(ai_dir: str, base: float, p1h,
                         threshold: float = 0.0) -> str:
    """Verdict once the 1h price is in. A call is CORRECT only if price moved the
    AI's way by MORE THAN `threshold` (max of spread and 0.5xATR, captured at
    decision time). This stops a move of a fraction of a tick — smaller than the
    spread you'd pay to trade it — from counting as a win. (peterpt + Claude
    2026-08-06)

    buy  correct: p1h - base >  threshold
    sell correct: base - p1h >  threshold
    A move that did not clear the threshold in EITHER direction is "no" — the AI
    called a direction and price effectively went nowhere, which is not a win."""
    if p1h is None or not base:
        return "-"
    try:
        move = float(p1h) - float(base)     # +ve = price rose
        thr = float(threshold or 0.0)
    except Exception:
        return "-"
    if ai_dir == "buy":
        return "yes" if move > thr else "no"
    if ai_dir == "sell":
        return "yes" if (-move) > thr else "no"
    return "-"


def _patch_read_row(ts: float, sym: str, updates: dict) -> None:
    """Find the gate row in engine5_reads.jsonl matching (ts, sym) and merge in
    the follow-up prices/accuracy. Rewrites the file (it's small — trimmed)."""
    try:
        if not os.path.exists(READS_LOG_PATH):
            return
        lines = open(READS_LOG_PATH, "r", encoding="utf-8").read().splitlines()
        out = []
        for ln in lines:
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                out.append(ln)
                continue
            if (rec.get("kind") == "gate"
                    and abs(float(rec.get("ts", 0)) - ts) < 0.5
                    and rec.get("symbol") == sym):
                rec.update(updates)
                out.append(json.dumps(rec))
            else:
                out.append(ln)
        tmp = READS_LOG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, READS_LOG_PATH)
    except Exception:
        pass


def _process_followups() -> None:
    """Called each E5 loop cycle. Fills any due +15m/+30m/+1h prices, patches the
    matching reads row, and marks the accuracy verdict once 1h is in."""
    items = _load_followups()
    if not items:
        return
    now = time.time()
    prices = _live_prices()
    changed = False
    still_pending = []
    for it in items:
        if it.get("done"):
            continue
        sym = it["symbol"]
        base = it.get("decision_price")
        patch = {}
        for key, delay in _FU_WINDOWS:
            if it.get(key) is None and now >= it["ts"] + delay:
                px = _price_of(prices, sym)
                if px:
                    it[key] = round(float(px), 8)
                    patch[key] = it[key]
                    changed = True
        # once 1h price is in, compute accuracy and finish this follow-up
        if it.get("p1h") is not None:
            acc = _accuracy_from_prices(it["ai_direction"], base, it["p1h"],
                                        it.get("move_threshold", 0.0))
            it["done"] = True
            patch["accurate"] = acc
            changed = True
        if patch:
            _patch_read_row(it["ts"], sym, patch)
        if not it.get("done"):
            still_pending.append(it)
    if changed:
        # keep only unfinished follow-ups (finished ones are patched into the log)
        _save_followups(still_pending)


def _write_read(rec: dict) -> None:
    # per-symbol latest
    try:
        p = os.path.join(HUB_DIR, f"engine5_latest_{rec['symbol']}.json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, p)
    except Exception:
        pass
    # append-only log (trimmed)
    try:
        with open(READS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["ts"] % 25 < 1:
            try:
                with open(READS_LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > 800:
                    with open(READS_LOG_PATH, "w", encoding="utf-8") as f:
                        f.writelines(lines[-600:])
            except Exception:
                pass
    except Exception:
        pass


# ── Trigger source ────────────────────────────────────────────────────────────
# In test mode, since E5 is NOT wired into the engines yet, it watches the shared
# engine event stream (engine_events.jsonl) for wake-up events and reads on those.
# This is the same "watch for new input, then call" shape as sentiment_fetcher —
# here the input is our own engines' events rather than news. When Phase 2 comes,
# the engines will call E5 directly instead; this watcher is the test-mode stand-in.
EVENTS_PATH = os.path.join(HUB_DIR, "engine_events.jsonl")
_ACTIONABLE = ("touch", "cross", "signal", "arm", "choch", "vwap", "setup")


def _tail_new_events(state: dict) -> List[dict]:
    """Return event dicts appended since last check. Tracks byte offset."""
    out = []
    try:
        size = os.path.getsize(EVENTS_PATH)
    except Exception:
        return out
    last = state.get("offset", 0)
    if size < last:          # file rotated/trimmed
        last = 0
    if size == last:
        return out
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            f.seek(last)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        state["offset"] = size
    except Exception:
        pass
    return out


def _is_actionable(ev: dict) -> Optional[str]:
    """If this event is a decision-point wake-up, return a short trigger label,
    else None."""
    blob = json.dumps(ev).lower()
    for kw in _ACTIONABLE:
        if kw in blob:
            return kw
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine 5 — AI advisory read (test mode)")
    parser.add_argument("--hub-dir", default=None)
    parser.add_argument("--once", action="store_true", help="one pass then exit")
    parser.add_argument("--interval", type=float, default=5.0)
    # Debug: build & print the gate prompt for a symbol WITHOUT calling the AI.
    # Wastes zero tokens — lets us inspect exactly what would be sent. (peterpt)
    #   python3 pt_engine5.py --dry-prompt GOLD:buy --hub-dir hub_data
    # Scorecard from the same binary — no extra script in the tree
    # (peterpt 2026-08-03).
    parser.add_argument("--scorecard", action="store_true",
                        help="grade logged AI decisions against what price did")
    parser.add_argument("--days", type=float, default=7.0,
                        help="scorecard window in days (default 7)")
    parser.add_argument("--sc-symbol", default=None,
                        help="scorecard: limit to one instrument")
    parser.add_argument("--sc-engine", default=None,
                        help="scorecard: limit to one engine")
    parser.add_argument("--dry-prompt", default=None, metavar="SYM:DIR",
                        help="show the gate prompt for SYM:DIR (e.g. GOLD:buy) "
                             "without calling the AI, then exit")
    # Live single-gate test (peterpt 2026-07-31): makes ONE real AI call for
    # SYM:DIR and prints the full result — direction, confidence, PRICE, and the
    # use_ai_price verdict. Costs one API call. For verifying the end-to-end
    # path (does the AI reply with a usable price?) before wiring the trader.
    #   python3 pt_engine5.py --live-gate QTUMUSD:sell --hub-dir hub_data
    parser.add_argument("--live-gate", default=None, metavar="SYM:DIR",
                        help="make ONE real AI gate call for SYM:DIR and print "
                             "the full result including price (costs 1 API call)")
    args = parser.parse_args()

    global HUB_DIR, RAMFS_DIR, READS_LOG_PATH, CANDLES_DIR, EVENTS_PATH, FOLLOWUPS_PATH
    if args.hub_dir:
        HUB_DIR = os.path.abspath(args.hub_dir)
        RAMFS_DIR = os.path.join(HUB_DIR, "ramfs")
        READS_LOG_PATH = os.path.join(HUB_DIR, "engine5_reads.jsonl")
        CANDLES_DIR = os.path.join(HUB_DIR, "candles")
        EVENTS_PATH = os.path.join(HUB_DIR, "engine_events.jsonl")
        FOLLOWUPS_PATH = os.path.join(HUB_DIR, "engine5_followups.json")

    # ── Dry-prompt debug mode (no AI call, no tokens) ────────────────────────
    if args.scorecard:
        try:
            print(scorecard_text(scorecard(hub_dir=args.hub_dir,
                                           days=args.days,
                                           symbol=args.sc_symbol,
                                           engine=args.sc_engine)))
        except Exception as e:
            print(f"scorecard error: {type(e).__name__}: {e}")
        sys.exit(0)

    if args.dry_prompt:
        try:
            sym, _, d = args.dry_prompt.partition(":")
            d = (d or "buy").lower().strip()
            sym = sym.strip()
            prices = _live_prices()
            px = _price_of(prices, sym)
            print("=" * 60)
            print(f"DRY PROMPT for {sym} {d.upper()}  (NO AI CALL — 0 tokens)")
            print("=" * 60)
            # Minimal sanity check: candle availability (the one thing NOT
            # visible in the prompt itself). Everything else — trend, ATR,
            # neurals, MACD, sentiment — is shown IN the prompt below, so we
            # don't duplicate it here.
            _c = _recent_candles(sym, tf="15min", n=40)
            print("--- data sanity ---")
            print(f"live price: {px}")
            print(f"15min candles available: {len(_c)}  (MACD needs >=26)")
            print("--- exact prompt that WILL be sent to AI ---")
            print(build_gate_prompt("DEBUG", sym, d, px or 0, prices))
            print("=" * 60)
            print(f"prompt length: {len(build_gate_prompt('DEBUG', sym, d, px or 0, prices))} chars "
                  f"(~{len(build_gate_prompt('DEBUG', sym, d, px or 0, prices))//4} tokens)")
        except Exception as e:
            print(f"dry-prompt error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        return

    # ── Live single-gate test (ONE real AI call) ─────────────────────────────
    if args.live_gate:
        try:
            sym, _, d = args.live_gate.partition(":")
            d = (d or "buy").lower().strip()
            sym = sym.strip()
            prices = _live_prices()
            px = _price_of(prices, sym)
            print("=" * 60)
            print(f"LIVE GATE for {sym} {d.upper()}  (ONE REAL AI CALL)")
            print("=" * 60)
            print(f"current price: {px}")
            print("calling e5_gate ... (this makes a real API request)")
            res = e5_gate("LIVE-TEST", sym, d, px or 0, hub_dir=HUB_DIR)
            print("--- AI RESULT ---")
            if not isinstance(res, dict):
                print(f"unexpected result type: {type(res)} -> {res}")
            else:
                print(f"decision   : {res.get('decision')}")
                print(f"direction  : {res.get('direction')}")
                print(f"confidence : {res.get('confidence')}")
                print(f"ai_price   : {res.get('ai_price')}   <-- the suggested entry price")
                _uap = res.get("use_ai_price") or {}
                print(f"use_ai_price: use={_uap.get('use')} price={_uap.get('price')} "
                      f"reason={_uap.get('reason')}")
                print(f"reasoning  : {res.get('reasoning')}")
                if res.get("reason"):
                    print(f"gate reason: {res.get('reason')}")
            print("=" * 60)
        except Exception as e:
            print(f"live-gate error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
        return

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    log("=" * 56)
    log("pt_engine5.py — AI advisory read (TEST MODE — logs only, no trades)")
    log(f"HUB   = {HUB_DIR}")
    _seed_e5_defaults()          # write default config if missing (first run)
    gs = load_settings()
    cfg = e5_config(gs)
    log(f"PROVIDER = {cfg['provider']}  MODEL = {cfg['model'] or '(default)'}  "
        f"ENABLED = {cfg['enabled']}  MODE = {cfg['mode']}")
    if not providers.provider_available(cfg["provider"]):
        log(f"  ⚠ SDK for '{cfg['provider']}' not installed — reads will log as errors.")
    log("=" * 56)

    _state = {"offset": 0, "cooldown": {}}   # per-symbol last-read time
    _last_heartbeat = 0.0
    _reads_done = 0

    # On first start, skip to the end of the events file (don't replay history).
    try:
        _state["offset"] = os.path.getsize(EVENTS_PATH)
    except Exception:
        _state["offset"] = 0

    while _RUN:
        gs = load_settings()
        cfg = e5_config(gs)

        # Fill any due +15m/+30m/+1h price follow-ups and score AI accuracy.
        # Runs BEFORE the enabled-check so decisions already in flight keep
        # getting tracked even if E5 is toggled off mid-window (peterpt).
        try:
            _process_followups()
        except Exception:
            pass

        if not cfg["enabled"]:
            if args.once:
                break
            time.sleep(args.interval)
            continue

        # Heartbeat every ~60s so the log visibly shows E5 is alive and waiting
        # for engine wake-ups (peterpt 2026-07-26). Without this a correctly-idle
        # E5 looks indistinguishable from a dead one.
        _now = time.time()
        if _now - _last_heartbeat >= 60:
            _last_heartbeat = _now
            log(f"[E5] waiting for engine setups — provider={cfg['provider']} "
                f"mode={cfg['mode']} reads_so_far={_reads_done} "
                f"(watching {os.path.basename(EVENTS_PATH)})")

        prices = _live_prices()
        for ev in _tail_new_events(_state):
            trig = _is_actionable(ev)
            if not trig:
                continue
            sym = ev.get("symbol") or ev.get("sym")
            if not sym:
                continue
            # per-symbol cooldown (debounce repeated touches)
            _now = time.time()
            _last = _state["cooldown"].get(sym, 0)
            if _now - _last < cfg["cooldown_s"]:
                continue
            _state["cooldown"][sym] = _now
            # Legacy passive observation (do_read) is OFF by default now — the
            # gate (e5_gate) is the real decision path and already asks the AI.
            # Running do_read too meant a SECOND duplicate AI call per signal
            # (the paired no-read + read rows) and double token cost. Only run it
            # if e5_observe is explicitly turned on. The rest of this loop —
            # crucially _process_followups() which scores the scorecard — keeps
            # running regardless. (peterpt 2026-07-31)
            if not cfg.get("observe", False):
                continue
            try:
                _rec = do_read(sym, trig, cfg, prices)
                if _rec is not None:
                    _reads_done += 1
            except Exception as e:
                log(f"[E5] {sym}: read failed — {type(e).__name__}: {e}")

        if args.once:
            break
        time.sleep(args.interval)

    log("pt_engine5 stopped.")


if __name__ == "__main__":
    main()
