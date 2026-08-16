"""
AUTO-LEARNING MARKET RESEARCH LAB
=================================

VERSION 0.1 - RESEARCH / PAPER TRADING ONLY

Purpose
-------
This program is NOT a live trading bot.

Its first job is to answer:

    "Is there any repeatable, cost-adjusted, out-of-sample
     information in the market state?"

Architecture:

    Market data
        |
        v
    Feature engine
        |
        v
    Historical-neighbour learner
        |
        v
    Forward outcome measurement
        |
        v
    Transaction-cost model
        |
        v
    Virtual broker
        |
        v
    Performance / evidence engine
        |
        v
    Research database

IMPORTANT
---------
DO NOT connect real order execution here.

The real broker integration should remain outside this research engine.

Claude:
-------
Please connect the marked integration points to the existing application
instead of inventing a second broker architecture.
"""


from __future__ import annotations

import math
import os
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    # --------------------------------------------------------
    # Instrument
    # --------------------------------------------------------

    instrument: str = "GOLD"

    timeframe: str = "5m"

    # --------------------------------------------------------
    # Research horizons
    # --------------------------------------------------------
    #
    # If timeframe = 5m:
    #
    # 1  = 5 minutes
    # 3  = 15 minutes
    # 6  = 30 minutes
    # 12 = 1 hour
    # 24 = 2 hours
    # 48 = 4 hours
    #

    horizons = (1, 3, 6, 12, 24, 48)

    # --------------------------------------------------------
    # Historical-neighbour learner
    # --------------------------------------------------------

    k_neighbors: int = 50

    min_neighbors: int = 20

    # --------------------------------------------------------
    # Walk-forward configuration
    # --------------------------------------------------------

    minimum_training_bars: int = 1000

    retrain_every_bars: int = 100

    # --------------------------------------------------------
    # COST MODEL
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # These are deliberately configurable.
    #
    # Claude should eventually populate these from the actual
    # Capital.com instrument specifications / existing app.
    #
    # Costs are expressed in PRICE units.
    #

    spread_cost: float = 0.00

    commission_cost: float = 0.00

    slippage_cost: float = 0.00

    financing_cost: float = 0.00

    # Cost-model provenance (ChatGPT #2/#14 + earlier cost_model_version ask).
    # While costs are all zero the model is UNCALIBRATED and the auditor MUST
    # block promotion regardless of results. When batch-3 fills real Capital
    # costs it also sets these to e.g. "CONFIGURED" / "capital_meta_2026-08-15".
    cost_model_status: str = "UNCALIBRATED"   # honest default; never blank
    cost_model_version: str = "uncalibrated_zero_v0"

    # Paths to the real broker-derived cost config + market-hours calendar
    # (both consumed by pt_cost_and_calendar). None -> module defaults.
    cost_config_path: str = None
    calendar_path: str = None
    market_calendar_version: str = "cal_none"
    target_validity_rule: str = ("exclude targets whose [t, t+h] holding path "
                                 "crosses a market closure (BAR_CLOSE)")
    require_calendar: bool = False         # True in real fleet; False for demo

    # ── Real-data mode (fleet workers) ───────────────────────────────────
    # When use_real_data=True, load_market_data fetches from the app's
    # UniversalDataProvider (Capital-only), cached to disk with nightly top-up.
    use_real_data: bool = False
    research_timeframe: str = "1hour"      # first-deployment TF (ChatGPT ruling)
    history_limit: int = 6000             # ~12 months of 1h bars
    history_cache_dir: str = None
    data_snapshot_hash: str = "unwired"

    # Data provenance — stamped into every report/experiment so a synthetic
    # demo run can never be mistaken for market evidence (ChatGPT #14).
    data_source: str = "UNKNOWN"           # e.g. "DEMO_SYNTHETIC","CAPITAL_LIVE"
    is_synthetic_demo: bool = False

    # --------------------------------------------------------
    # Minimum edge required before a virtual trade
    # --------------------------------------------------------

    minimum_net_edge: float = 0.0001

    # --------------------------------------------------------
    # Virtual account
    # --------------------------------------------------------

    starting_equity: float = 10_000.0

    risk_per_trade: float = 0.01

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    historical_file: str = "market_data.csv"

    database_file: str = "market_research.db"

    # --------------------------------------------------------
    # Demo mode
    # --------------------------------------------------------

    generate_demo_data_if_missing: bool = True

    demo_bars: int = 20_000


CONFIG = Config()


# ============================================================
# BROKER INTEGRATION PLACEHOLDERS
# ============================================================

def request_current_price(instrument: str):
    """
    ============================================================
    CAPITAL.COM INTEGRATION POINT
    ============================================================

    Claude:

    Replace the body of this function with the existing
    application's price-request function.

    DO NOT create a second broker client if the existing app
    already has one.

    Required conceptual operation:

        current_price = request_current_price("GOLD")

    PLACEHOLDER:

        "Price request on instrument HERE"

    Expected result:

        float

    Example:

        return existing_capital_api.get_current_price(instrument)

    ============================================================
    """

    # Real integration: read the app's WebSocket-backed live price store via
    # the UniversalDataProvider (no REST, no broker session opened here).
    prov = _get_data_provider()
    if prov is None:
        raise NotImplementedError(
            "UniversalDataProvider unavailable — cannot fetch current price.")
    try:
        return float(prov.get_current_price(instrument))
    except Exception as e:
        raise RuntimeError(f"current price fetch failed for {instrument}: {e}")


# ── real-data provider plumbing (lazy singleton) ───────────────────────────
_DATA_PROVIDER = None


def _get_data_provider():
    """Lazy import + singleton of the app's UniversalDataProvider. Returns None
    if the app module isn't importable (e.g. running standalone in research
    env), so callers can fall back to demo/cache."""
    global _DATA_PROVIDER
    if _DATA_PROVIDER is not None:
        return _DATA_PROVIDER
    try:
        import universal_data_provider as _udp
        _DATA_PROVIDER = _udp.UniversalDataProvider()
    except Exception:
        _DATA_PROVIDER = None
    return _DATA_PROVIDER


def _history_cache_path(instrument: str, timeframe: str,
                        cache_dir: str = None) -> str:
    import os
    d = cache_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "research_history_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"hist_{instrument}_{timeframe}.csv")


def request_market_history(instrument: str, timeframe: str,
                           limit: int = 6000, cache_dir: str = None):
    """Historical candles as a DataFrame[timestamp,open,high,low,close,volume].

    Strategy (ChatGPT ruling): fetch history ONCE to disk, then top up only new
    bars on later calls — never re-pull the full window every run. Uses the
    app's UniversalDataProvider (Capital-only, clean provenance). If the
    provider is unavailable, returns any cached history, else raises.
    """
    import os
    path = _history_cache_path(instrument, timeframe, cache_dir)

    cached = None
    if os.path.isfile(path):
        try:
            cached = pd.read_csv(path)
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        except Exception:
            cached = None

    prov = _get_data_provider()
    if prov is None:
        if cached is not None and len(cached):
            return cached
        raise NotImplementedError(
            "UniversalDataProvider unavailable and no cached history.")

    # Fetch (deep history via the provider's to= pagination). On a top-up we
    # still request `limit`, then merge/dedup with cache — cheap and robust.
    try:
        candles = prov.fetch_candles(instrument, timeframe, limit)
    except Exception as e:
        if cached is not None and len(cached):
            print(f"  [history] fetch failed ({e}); using cached history.")
            return cached
        raise

    rows = []
    for c in (candles or []):
        d = c.to_dict() if hasattr(c, "to_dict") else c
        rows.append({
            "timestamp": d.get("timestamp"),
            "open": d.get("open"), "high": d.get("high"),
            "low": d.get("low"), "close": d.get("close"),
            "volume": d.get("volume", 0.0),
        })
    fresh = pd.DataFrame(rows)
    if len(fresh):
        fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], utc=True)

    if cached is not None and len(cached):
        merged = (pd.concat([cached, fresh], ignore_index=True)
                  .drop_duplicates(subset=["timestamp"])
                  .sort_values("timestamp")
                  .reset_index(drop=True))
    else:
        merged = fresh.sort_values("timestamp").reset_index(drop=True)

    try:
        merged.to_csv(path, index=False)
    except Exception:
        pass
    return merged


def data_snapshot_hash(df: pd.DataFrame) -> str:
    """Provenance hash of the exact data a run used (first/last timestamp + row
    count + close checksum) — frozen into the spec so 'the data changed under a
    freeze' is detectable."""
    import hashlib
    try:
        if df is None or not len(df):
            return "data_empty"
        key = (f"{df['timestamp'].iloc[0]}|{df['timestamp'].iloc[-1]}|"
               f"{len(df)}|{float(df['close'].sum()):.4f}")
        return "data_" + hashlib.sha256(key.encode()).hexdigest()[:16]
    except Exception:
        return "data_unknown"


# ============================================================
# DEMO DATA
# ============================================================

def generate_demo_market_data(
    bars: int = 20_000,
    start_price: float = 2_000.0,
) -> pd.DataFrame:
    """
    Generates synthetic market data.

    THIS IS NOT MARKET DATA.

    It exists only so Claude can run the complete pipeline
    before connecting the real Capital.com data feed.
    """

    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=bars,
        freq="5min",
        tz="UTC",
    )

    # Create changing volatility and weak synthetic regimes.
    regime = np.zeros(bars)

    regime_length = 500

    current_regime = 0.0

    for i in range(0, bars, regime_length):

        current_regime = rng.choice(
            [-1.0, 0.0, 1.0]
        )

        end = min(i + regime_length, bars)

        regime[i:end] = current_regime

    volatility = rng.uniform(
        0.00015,
        0.0012,
        bars,
    )

    drift = regime * 0.00002

    noise = rng.normal(
        0,
        volatility,
        bars,
    )

    returns = drift + noise

    prices = (
        start_price *
        np.exp(np.cumsum(returns))
    )

    close = prices

    open_ = np.concatenate(
        [
            [start_price],
            close[:-1],
        ]
    )

    ranges = np.abs(
        rng.normal(
            0,
            volatility * close,
        )
    )

    high = np.maximum(
        open_,
        close,
    ) + ranges

    low = np.minimum(
        open_,
        close,
    ) - ranges

    volume = rng.lognormal(
        mean=10,
        sigma=0.4,
        size=bars,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_market_data(config: Config) -> pd.DataFrame:

    # ── REAL DATA (preferred when config.use_real_data) ──────────────────
    # Fetch from the app's UniversalDataProvider (Capital-only, clean
    # provenance), cached to disk with incremental top-up. This is what the
    # fleet runner uses for live workers.
    if getattr(config, "use_real_data", False):
        print(f"Fetching real history: {config.instrument} "
              f"{config.research_timeframe} (limit={config.history_limit})")
        df = request_market_history(
            config.instrument, config.research_timeframe,
            limit=config.history_limit,
            cache_dir=getattr(config, "history_cache_dir", None))
        config.data_source = f"CAPITAL:{config.research_timeframe}"
        config.is_synthetic_demo = False
        config.data_snapshot_hash = data_snapshot_hash(df)
        return _ensure_columns(df)

    path = Path(config.historical_file)

    if path.exists():

        print(
            f"Loading market data from {path}"
        )

        df = pd.read_csv(path)
        # Real file on disk. We can't be certain of its origin, but it is NOT
        # our synthetic generator, so mark it accordingly.
        config.data_source = "CSV_FILE"
        config.is_synthetic_demo = False

    elif config.generate_demo_data_if_missing:

        print(
            "No market_data.csv found."
        )

        print(
            "Generating synthetic demo data."
        )
        print(
            "⚠ DEMO SYNTHETIC DATA — results are NOT market evidence. "
            "The demo generator injects a deliberate directional signal; any "
            "'edge' found here is the planted one, not a discovery."
        )

        df = generate_demo_market_data(
            config.demo_bars
        )
        # Stamp provenance so no report/DB row can be mistaken for real data.
        config.data_source = "DEMO_SYNTHETIC"
        config.is_synthetic_demo = True

    else:

        raise FileNotFoundError(
            f"Market data not found: {path}"
        )

    return _ensure_columns(df)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df


# ============================================================
# FEATURE ENGINE
# ============================================================

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_12",

    "trend",

    "volatility_12",
    "volatility_48",

    "range",

    "distance_high",
    "distance_low",

    "relative_volume",

    "time_sin",
    "time_cos",
]


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = df.copy()

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    x["return_1"] = (
        x["close"].pct_change(1)
    )

    x["return_3"] = (
        x["close"].pct_change(3)
    )

    x["return_12"] = (
        x["close"].pct_change(12)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    x["volatility_12"] = (
        x["return_1"]
        .rolling(12)
        .std()
    )

    x["volatility_48"] = (
        x["return_1"]
        .rolling(48)
        .std()
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    ema_fast = (
        x["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema_slow = (
        x["close"]
        .ewm(
            span=48,
            adjust=False,
        )
        .mean()
    )

    x["trend"] = (
        ema_fast - ema_slow
    ) / x["close"]

    # --------------------------------------------------------
    # Candle range
    # --------------------------------------------------------

    x["range"] = (
        x["high"] - x["low"]
    ) / x["close"]

    # --------------------------------------------------------
    # Position inside recent range
    # --------------------------------------------------------

    rolling_high = (
        x["high"]
        .rolling(48)
        .max()
    )

    rolling_low = (
        x["low"]
        .rolling(48)
        .min()
    )

    x["distance_high"] = (
        x["close"] - rolling_high
    ) / x["close"]

    x["distance_low"] = (
        x["close"] - rolling_low
    ) / x["close"]

    # --------------------------------------------------------
    # Relative volume
    # --------------------------------------------------------

    volume_mean = (
        x["volume"]
        .rolling(48)
        .mean()
    )

    volume_std = (
        x["volume"]
        .rolling(48)
        .std()
    )

    x["relative_volume"] = (
        x["volume"] - volume_mean
    ) / volume_std.replace(
        0,
        np.nan,
    )

    # --------------------------------------------------------
    # Time-of-day
    # --------------------------------------------------------

    minute_of_day = (
        x["timestamp"].dt.hour * 60
        + x["timestamp"].dt.minute
    )

    x["time_sin"] = np.sin(
        2 * np.pi * minute_of_day / 1440
    )

    x["time_cos"] = np.cos(
        2 * np.pi * minute_of_day / 1440
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    median_vol = (
        x["volatility_48"]
        .rolling(500)
        .median()
    )

    high_vol = (
        x["volatility_48"]
        > median_vol
    )

    x["regime"] = np.select(
        [
            (x["trend"] > 0)
            & high_vol,

            (x["trend"] > 0)
            & ~high_vol,

            (x["trend"] <= 0)
            & high_vol,
        ],

        [
            "UP_HIGH_VOL",
            "UP_LOW_VOL",
            "DOWN_HIGH_VOL",
        ],

        default="DOWN_LOW_VOL",
    )

    return x


# ============================================================
# FORWARD OUTCOMES
# ============================================================

def add_forward_outcomes(
    df: pd.DataFrame,
    horizons,
    instrument: str = None,
    calendar: dict = None,
    require_calendar: bool = False,
) -> pd.DataFrame:
    """Compute forward-return targets, EXCLUDING observations whose holding path
    crosses a market closure (ChatGPT market-hours ruling).

    An excluded target is set to NaN — the walk-forward already skips NaN
    targets, so a closed-market observation is simply not scored (not a win,
    not a loss, not a fake NO_TRADE). If no calendar is supplied, behaviour is
    unchanged (all targets kept) unless require_calendar=True.
    """
    x = df.copy()

    for horizon in horizons:

        x[
            f"forward_return_{horizon}"
        ] = (
            x["close"].shift(-horizon)
            / x["close"]
            - 1
        )

        # Market-hours target validity: NaN-out gap-crossing targets.
        if _CC is not None and calendar is not None and instrument is not None:
            try:
                mask = _CC.valid_target_mask(
                    x, instrument, horizon, calendar,
                    require_calendar=require_calendar)
                col = f"forward_return_{horizon}"
                x.loc[~mask, col] = np.nan
            except Exception:
                pass

    return x


# ============================================================
# COST MODEL
# ============================================================
#
# Real per-instrument cost comes from pt_cost_and_calendar (broker-derived,
# fail-safe to UNCALIBRATED). If no valid cost config exists for the instrument
# the cost is 0.0 AND the run is marked UNCALIBRATED so the auditor BLOCKS it —
# a zero cost never masquerades as a real (calibrated) zero.

try:
    import pt_cost_and_calendar as _CC
except Exception:
    _CC = None


def total_round_trip_cost(
    config: Config,
) -> float:

    return (
        config.spread_cost
        + config.commission_cost
        + config.slippage_cost
        + config.financing_cost
    )


def cost_as_return(
    price: float,
    config: Config,
) -> float:
    """Per-trade round-trip cost as a price-relative return.

    Priority: real broker-derived cost from pt_cost_and_calendar for this
    instrument. If unavailable (no config / uncalibrated), fall back to the
    Config's explicit cost fields (default 0.0). When the real module returns a
    CALIBRATED cost, config.cost_model_status/version are updated as a side
    effect so the experiment records genuine provenance.

    Financing is passed 0 boundaries here (intraday default); the simulator can
    supply real boundary counts later for overnight-hold analysis.
    """
    if price <= 0:
        return 0.0

    if _CC is not None:
        comp = _CC.cost_components(
            getattr(config, "instrument", "GOLD"),
            price,
            financing_boundaries_crossed=0,
            path=getattr(config, "cost_config_path", None),
        )
        if comp is not None:
            # record real provenance on the config (calibrated OR partial)
            config.cost_model_version = comp.get("cost_model_version",
                                                 config.cost_model_version)
            config.cost_model_status = (
                "CONFIGURED" if comp.get("calibrated") else
                comp.get("cost_status", "UNCALIBRATED"))
            return float(comp["cost_return"])

    # No real cost available → zero, and force UNCALIBRATED so nothing passes.
    config.cost_model_status = "UNCALIBRATED"
    return (
        total_round_trip_cost(config)
        / price
    )


# ────────────────────────────────────────────────────────────────────────────
# CANONICAL SIGNED-RETURN  (peterpt + ChatGPT + Claude 2026-08-15)
#
# THE ONE PLACE direction + cost are combined. Every consumer (walk-forward,
# event clustering, report, virtual broker, auditor) MUST route through this —
# do NOT re-implement the sign/cost math anywhere else.
#
# WHY: the previous code stored a LONG-oriented net return
#         net = raw_return - cost
# and then re-signed it downstream for shorts as `-net`, which expands to
#         -(raw - cost) = -raw + cost
# i.e. it ADDED the transaction cost to profitable shorts instead of charging
# it. The cost must be subtracted AFTER orienting the raw return to the trade
# direction, never before-then-negated.
#
# CONVENTION: raw_return is the LONG-oriented forward return (what a long makes).
#   LONG  →  +raw - cost
#   SHORT →  -raw - cost   (short profits when raw < 0; cost still a debit)
#   else  →   0.0
# The returned number is FINAL and already direction-signed and cost-charged.
# Downstream must NEVER negate it again.
# ────────────────────────────────────────────────────────────────────────────

def directional_net_return(
    raw_return: float,
    direction: str,
    cost_return: float,
) -> float:
    if direction == "LONG":
        return raw_return - cost_return
    if direction == "SHORT":
        return -raw_return - cost_return
    return 0.0


def timeframe_to_seconds(timeframe: str) -> float:
    """'5m'->300, '15m'->900, '1h'->3600, '4h'->14400, '1d'->86400.
    Falls back to 300 (5m) on an unrecognised string rather than guessing, so
    event clustering never silently uses a wildly wrong bar size."""
    try:
        tf = str(timeframe).strip().lower()
        num = int("".join(c for c in tf if c.isdigit()) or "0")
        unit = "".join(c for c in tf if c.isalpha())
        mult = {"m": 60, "min": 60, "h": 3600, "hour": 3600,
                "d": 86400, "day": 86400, "w": 604800}.get(unit, 60)
        secs = num * mult
        return float(secs) if secs > 0 else 300.0
    except Exception:
        return 300.0


# ============================================================
# HISTORICAL NEIGHBOUR MODEL
# ============================================================

class HistoricalNeighbourModel:

    def __init__(
        self,
        k: int,
        minimum_examples: int,
    ):

        self.k = k

        self.minimum_examples = (
            minimum_examples
        )

        self.mean = None
        self.std = None

        self.training_matrix = None
        self.training_indices = None

    def fit(
        self,
        training_data: pd.DataFrame,
    ):

        data = (
            training_data[
                FEATURE_COLUMNS
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if (
            len(data)
            < self.minimum_examples
        ):

            self.training_matrix = None

            return self

        self.mean = data.mean()

        self.std = (
            data.std()
            .replace(0, 1)
        )

        self.training_matrix = (
            (data - self.mean)
            / self.std
        ).to_numpy()

        self.training_indices = (
            data.index
        )

        return self

    def predict(
        self,
        current_row: pd.Series,
        future_returns: pd.Series,
    ):

        if self.training_matrix is None:

            return None

        current = (
            current_row[
                FEATURE_COLUMNS
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        if current.isna().any():

            return None

        # df.iloc[position] returns an OBJECT-dtype Series (the full row mixes
        # timestamps/strings/floats), and slicing FEATURE_COLUMNS keeps object
        # dtype even though every value is numeric. Left as-is, the normalized
        # vector stays object and np.linalg.norm() raises
        # "numpy.float64 has no sqrt method". Coerce to float64 explicitly.
        # (peterpt + Claude 2026-08-15 — caught in real-data testing)
        current_vector = (
            (current.astype(float) - self.mean)
            / self.std
        ).to_numpy(dtype=float)

        distances = np.linalg.norm(
            self.training_matrix
            - current_vector,
            axis=1,
        )

        k = min(
            self.k,
            len(distances),
        )

        positions = np.argpartition(
            distances,
            k - 1,
        )[:k]

        indices = (
            self.training_indices[
                positions
            ]
        )

        outcomes = (
            future_returns
            .reindex(indices)
            .dropna()
        )

        if (
            len(outcomes)
            < self.minimum_examples
        ):

            return None

        return {
            "expected_return":
                float(outcomes.mean()),

            "median_return":
                float(outcomes.median()),

            "win_probability":
                float(
                    (outcomes > 0).mean()
                ),

            "sample_count":
                int(len(outcomes)),

            "return_std":
                float(outcomes.std()),

            "distance":
                float(
                    distances[positions].mean()
                ),
        }


# ============================================================
# VIRTUAL BROKER
# ============================================================

@dataclass
class VirtualTrade:

    timestamp: str

    instrument: str

    direction: str

    entry_price: float

    exit_price: float

    expected_return: float

    realized_return: float

    pnl: float

    equity_after: float

    model_version: str

    regime: str


class VirtualBroker:

    """
    Simulates trades.

    NEVER sends anything to Capital.com.
    """

    def __init__(
        self,
        starting_equity: float,
    ):

        self.starting_equity = (
            starting_equity
        )

        self.equity = (
            starting_equity
        )

        self.peak_equity = (
            starting_equity
        )

        self.max_drawdown = 0.0

        self.trades = []

    def execute(
        self,
        timestamp,
        instrument,
        direction,
        entry_price,
        exit_price,
        expected_return,
        risk_fraction,
        model_version,
        regime,
        cost_return=0.0,
    ):

        if direction == "NO_TRADE":

            return

        gross_return = (
            exit_price / entry_price
            - 1
        )

        # NET return: orient to direction AND charge cost, via the SAME
        # canonical function the report/clustering use. Previously the broker
        # charged NO cost, so the equity curve was GROSS while the reported
        # stats were NET — the two disagreed. Now equity and stats are the same
        # number. (ChatGPT #3, 2026-08-15)
        net_return = directional_net_return(
            gross_return, direction, cost_return)

        capital = (
            self.equity
            * risk_fraction
        )

        pnl = (
            capital
            * net_return
        )

        self.equity += pnl

        self.peak_equity = max(
            self.peak_equity,
            self.equity,
        )

        drawdown = (
            self.peak_equity
            - self.equity
        ) / self.peak_equity

        self.max_drawdown = max(
            self.max_drawdown,
            drawdown,
        )

        self.trades.append(
            VirtualTrade(
                timestamp=str(timestamp),
                instrument=instrument,
                direction=direction,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                expected_return=float(
                    expected_return
                ),
                realized_return=float(
                    net_return
                ),
                pnl=float(pnl),
                equity_after=float(
                    self.equity
                ),
                model_version=model_version,
                regime=regime,
            )
        )


# ============================================================
# WALK-FORWARD LEARNING
# ============================================================

@dataclass
class Prediction:

    timestamp: str

    horizon: int

    regime: str

    expected_return: float

    median_return: float

    win_probability: float

    sample_count: int

    uncertainty: float

    nearest_distance: float

    actual_return: float

    cost_return: float

    net_expected_return: float

    net_actual_return: float

    decision: str


def run_walk_forward(
    df: pd.DataFrame,
    config: Config,
    horizon: int,
):

    target = (
        f"forward_return_{horizon}"
    )

    model = HistoricalNeighbourModel(
        k=config.k_neighbors,
        minimum_examples=config.min_neighbors,
    )

    predictions = []

    last_training_position = -1

    for position in range(
        config.minimum_training_bars,
        len(df) - horizon,
    ):

        # --------------------------------------------
        # TRAIN ONLY ON INFORMATION AVAILABLE BEFORE
        # THIS TIMESTAMP.
        # --------------------------------------------

        # ────────────────────────────────────────────────────────────────
        # PURGE / EMBARGO  (peterpt + ChatGPT + Claude 2026-08-15)
        #
        # PURGE:  Remove training observations whose forward-outcome window
        #         overlaps the test decision timestamp at `position`.
        #
        # REASON: forward_return_{horizon}[i] looks `horizon` bars ahead of i.
        #         A training bar at i in (position-horizon, position) therefore
        #         has an outcome that extends AT OR BEYOND `position` — i.e.
        #         information that had NOT happened yet at decision time. Using
        #         it as a neighbour leaks the future into training and inflates
        #         measured edge.
        #
        # FIX:    train_end = position - horizon. Train (and read neighbour
        #         outcomes) ONLY from bars whose full outcome window closed
        #         before the decision bar.
        #
        # DO NOT "optimize" this away to reclaim sample size. Less data here is
        # correct; contaminated data is not. The whole project exists to avoid
        # exactly this kind of self-deception.
        # ────────────────────────────────────────────────────────────────
        train_end = position - horizon
        if train_end < config.min_neighbors:
            # Not enough purged history yet to form a clean neighbour set.
            continue

        if (
            last_training_position < 0
            or
            position
            - last_training_position
            >= config.retrain_every_bars
        ):

            training = (
                df.iloc[:train_end]
                .copy()
            )

            model.fit(training)

            last_training_position = (
                position
            )

        current = df.iloc[position]

        # Outcome series is ALSO truncated to train_end: a neighbour's outcome
        # must be fully realized before `position`. reindex() in predict() will
        # drop any neighbour whose outcome is NaN/absent here, so passing the
        # purged slice is what enforces the embargo on the outcomes too.
        prediction = model.predict(
            current,
            df[target].iloc[:train_end],
        )

        if prediction is None:

            continue

        entry_price = float(
            current["close"]
        )

        actual_return = float(
            current[target]
        )

        transaction_cost = (
            cost_as_return(
                entry_price,
                config,
            )
        )

        expected_return = (
            prediction[
                "expected_return"
            ]
        )

        # ── Explicit trade economics (ChatGPT #2) ────────────────────────
        # expected_return is the LONG-oriented predicted forward return. Go
        # long only if the predicted move clears cost+edge to the upside;
        # short only if it clears cost+edge to the downside; else NO_TRADE.
        # This is mathematically identical to the old
        #   net = expected_return - cost;  long if net>edge, short if net<-edge
        # but states the economics so no future edit can silently change the
        # meaning of "net_expected_return".
        edge_floor = transaction_cost + config.minimum_net_edge

        if expected_return > edge_floor:
            decision = "LONG"
        elif expected_return < -edge_floor:
            decision = "SHORT"
        else:
            decision = "NO_TRADE"

        # net_expected_return: direction-signed predicted return, cost charged.
        # For NO_TRADE it is the long-oriented net (diagnostic only).
        if decision == "NO_TRADE":
            net_expected_return = expected_return - transaction_cost
        else:
            net_expected_return = directional_net_return(
                expected_return, decision, transaction_cost)

        # net_actual_return: the REALIZED return, already direction-signed and
        # cost-charged via the canonical function. Downstream (clustering,
        # report, broker) must NOT negate this again.
        if decision == "NO_TRADE":
            net_actual_return = actual_return - transaction_cost
        else:
            net_actual_return = directional_net_return(
                actual_return, decision, transaction_cost)

        predictions.append(
            Prediction(

                timestamp=str(
                    current["timestamp"]
                ),

                horizon=horizon,

                regime=str(
                    current["regime"]
                ),

                expected_return=(
                    expected_return
                ),

                median_return=(
                    prediction[
                        "median_return"
                    ]
                ),

                win_probability=(
                    prediction[
                        "win_probability"
                    ]
                ),

                sample_count=(
                    prediction[
                        "sample_count"
                    ]
                ),

                uncertainty=(
                    prediction[
                        "return_std"
                    ]
                ),

                nearest_distance=(
                    prediction[
                        "distance"
                    ]
                ),

                actual_return=(
                    actual_return
                ),

                cost_return=(
                    transaction_cost
                ),

                net_expected_return=(
                    net_expected_return
                ),

                net_actual_return=(
                    net_actual_return
                ),

                decision=decision,
            )
        )

    return predictions


# ============================================================
# INDEPENDENT-EVENT ACCOUNTING + BLOCK BOOTSTRAP
# (peterpt + ChatGPT + Claude 2026-08-15)
#
# WHY THIS EXISTS
# ---------------
# run_walk_forward emits ONE Prediction per bar. On a 5m series, adjacent bars
# are heavily autocorrelated and their forward-outcome windows OVERLAP, so a
# run of same-direction trades is really ONE bet expressed many times — not
# many independent observations. Treating each bar as an independent sample
# makes N look ~5-10x bigger than it is, which shrinks confidence intervals and
# makes a coin-flip look "significant". §5b: count EVENTS, not candles.
#
# CLUSTERING RULE v1 (deliberately simple, auditable, tunable later)
# ------------------------------------------------------------------
# Walk the executed trades (decision != NO_TRADE) in time order. A trade starts
# a NEW event when EITHER:
#     • its decision direction differs from the current event's, OR
#     • it begins at least `horizon` bars after the current event's LAST trade
#       (its outcome window no longer overlaps the event) .
# Otherwise it joins the current event. Each event's return is the MEAN of its
# member trades' realized net returns (one bet → one number).
#
# This is intentionally conservative: when in doubt, collapse. Fewer, cleaner
# events beat many contaminated ones. A future version may key events off setup
# triggers (e.g. a volatility breakout) instead of raw adjacency — but that is
# an upgrade, not a correctness fix, and must be measured before adoption.
#
# NO heavy deps: pure numpy/pandas + stdlib, per the fanless-box constraint.
# ============================================================

def _bars_between(ts_a: str, ts_b: str, horizon_seconds: float) -> float:
    """Best-effort gap in 'bars' between two ISO timestamps, given the bar size
    implied by horizon_seconds/horizon. Falls back to a large gap (force split)
    if timestamps can't be parsed, so unparseable data never MERGES events."""
    try:
        ta = pd.to_datetime(ts_a)
        tb = pd.to_datetime(ts_b)
        return abs((tb - ta).total_seconds()) / max(horizon_seconds, 1e-9)
    except Exception:
        return float("inf")


def cluster_events(predictions, bar_seconds: float):
    """Collapse a time-ordered list of executed Predictions into independent
    events. Returns (event_returns: np.ndarray, event_meta: list[dict]).

    bar_seconds — seconds per bar of the research timeframe (e.g. 300 for 5m),
    used to convert the time gap between trades into a 'bars' count so the
    horizon-overlap test is correct regardless of missing bars / market gaps.
    """
    trades = [p for p in predictions if p.decision != "NO_TRADE"]
    if not trades:
        return np.array([], dtype=float), []

    # keep time order (walk-forward already emits in order, but be explicit)
    trades = sorted(trades, key=lambda p: str(p.timestamp))

    events = []          # each: {"returns":[...], "dir":..., "last_ts":..., "start_ts":...}
    for p in trades:
        # net_actual_return is ALREADY direction-signed and cost-charged by
        # directional_net_return() in run_walk_forward. Use it as-is — do NOT
        # negate for shorts (that was the old sign bug). (2026-08-15)
        r = p.net_actual_return
        if not events:
            events.append({"returns": [r], "dir": p.decision,
                           "last_ts": p.timestamp, "start_ts": p.timestamp,
                           "regime": p.regime, "horizon": p.horizon})
            continue

        cur = events[-1]
        gap_bars = _bars_between(cur["last_ts"], p.timestamp,
                                 bar_seconds)
        same_dir = (p.decision == cur["dir"])
        overlaps = gap_bars < float(p.horizon)   # outcome windows still overlap

        if same_dir and overlaps:
            cur["returns"].append(r)
            cur["last_ts"] = p.timestamp
        else:
            events.append({"returns": [r], "dir": p.decision,
                           "last_ts": p.timestamp, "start_ts": p.timestamp,
                           "regime": p.regime, "horizon": p.horizon})

    event_returns = np.array([float(np.mean(e["returns"])) for e in events],
                             dtype=float)
    return event_returns, events


def block_bootstrap_ci(event_returns: np.ndarray,
                       statistic="mean",
                       n_boot: int = 5000,
                       block_size: int = None,
                       ci: float = 0.95,
                       seed: int = 12345):
    """Block bootstrap CI over INDEPENDENT EVENTS (not candles).

    We bootstrap over the event series, resampling contiguous BLOCKS so any
    residual serial structure between adjacent events is preserved rather than
    shuffled into false independence (block bootstrap, not iid). With the
    clustering above events are already near-independent, so the default block
    is small; it's kept configurable and >1 so the method degrades safely if the
    clustering is ever loosened.

    statistic: "mean" | "median" | "positive_rate"
    Returns (point_estimate, lo, hi, n_events). NaNs if too few events.
    """
    n = len(event_returns)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)

    def _stat(a):
        if statistic == "median":
            return float(np.median(a))
        if statistic == "positive_rate":
            return float((a > 0).mean())
        return float(np.mean(a))

    point = _stat(event_returns)
    if n < 2:
        return (point, float("nan"), float("nan"), n)

    if block_size is None:
        # ~sqrt(n) is a standard default for block length; clamp to [1, n].
        block_size = max(1, min(n, int(round(math.sqrt(n)))))

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_size))
    stats = np.empty(n_boot, dtype=float)
    max_start = n - block_size
    for b in range(n_boot):
        if max_start <= 0:
            idx = rng.integers(0, n, size=n)
        else:
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_size)
                                  for s in starts])[:n]
        stats[b] = _stat(event_returns[idx])

    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return (point, lo, hi, n)


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

def calculate_report(
    predictions,
    broker: VirtualBroker,
    bar_seconds: float = 300.0,
    config: "Config" = None,
):

    if config is None:
        config = Config()

    if not predictions:

        return {
            "status": "NO_DATA"
        }

    trades = [
        p
        for p in predictions
        if p.decision != "NO_TRADE"
    ]

    if not trades:

        return {
            "status": "NO_TRADES",

            "predictions":
                len(predictions),

            "trades": 0,
        }

    returns = []

    for p in trades:

        # Already direction-signed and cost-charged by directional_net_return()
        # in run_walk_forward. Use as-is — no per-direction negation (old bug).
        returns.append(p.net_actual_return)

    returns = np.array(
        returns,
        dtype=float,
    )

    wins = (
        returns > 0
    ).sum()

    mean_return = (
        returns.mean()
    )

    median_return = (
        np.median(returns)
    )

    gross_profit = (
        returns[returns > 0].sum()
    )

    gross_loss = (
        -returns[returns < 0].sum()
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    # ────────────────────────────────────────────────────────────────────
    # INDEPENDENT EVENTS + BOOTSTRAP CIs  (the honest sample size)
    #
    # raw_observations : one per executed bar (autocorrelated — DO NOT treat
    #                    as independent).
    # independent_events: same-direction overlapping-window runs collapsed to
    #                    one bet each (see cluster_events).
    # All CIs and the honest Sharpe use EVENTS, not raw bars.
    # ────────────────────────────────────────────────────────────────────
    event_returns, events = cluster_events(predictions, bar_seconds)
    n_events = len(event_returns)

    mean_pt, mean_lo, mean_hi, _ = block_bootstrap_ci(
        event_returns, statistic="mean")
    med_pt, med_lo, med_hi, _ = block_bootstrap_ci(
        event_returns, statistic="median")
    pos_pt, pos_lo, pos_hi, _ = block_bootstrap_ci(
        event_returns, statistic="positive_rate")

    # Honest Sharpe-like: scale by sqrt(EVENTS), not sqrt(bars).
    ev_std = (event_returns.std(ddof=1) if n_events > 1 else 0.0)
    sharpe_events = (
        float(event_returns.mean() / ev_std * math.sqrt(n_events))
        if ev_std > 0 else 0.0
    )

    # NOTE: the old candle-count Sharpe (sqrt(len(trades))) is deliberately
    # NOT reported — it over-stated significance ~sqrt(bars/events)x. The
    # auditor consumes independent_events and the CIs, never raw trade count.

    # ── Raw-edge vs after-cost edge, and break-even cost (ChatGPT #9/#10) ──
    # gross event edge = mean event return with cost added back in. This tells
    # us WHICH world we're in: no predictive signal at all, vs a real gross
    # edge that broker costs then destroy. Break-even = the round-trip cost at
    # which net edge hits zero (how cheap execution must be to have any edge).
    per_event_cost = float(np.mean([p.cost_return for p in predictions
                                    if p.decision != "NO_TRADE"])) \
        if any(p.decision != "NO_TRADE" for p in predictions) else 0.0
    gross_event_edge = float(mean_pt + per_event_cost)   # add cost back
    net_event_edge = float(mean_pt)
    break_even_cost = gross_event_edge   # net=0 when cost == gross edge

    # ── Cost calibration + data provenance status (ChatGPT #2/#14) ────────
    # If costs are all zero the cost model is UNCALIBRATED and NO result may be
    # trusted or promoted, however good it looks. The auditor reads this and
    # BLOCKS promotion. cost_model_version travels with the record so we never
    # compare optimistic-cost vs realistic-cost experiments (ChatGPT earlier).
    cost_status = getattr(config, "cost_model_status", None) or (
        "UNCALIBRATED" if total_round_trip_cost(config) <= 0.0
        else "CONFIGURED")
    cost_model_version = getattr(config, "cost_model_version",
                                 "uncalibrated_zero_v0")

    return {

        "status": "OK",

        # ── evidence-quality / provenance flags (read by the auditor) ─────
        "cost_model_status": cost_status,           # UNCALIBRATED blocks promo
        "cost_model_version": cost_model_version,
        "data_source": getattr(config, "data_source", "UNKNOWN"),
        "is_synthetic_demo": bool(getattr(config, "is_synthetic_demo", False)),

        # ── raw (autocorrelated) counts — reference only ──────────────────
        "predictions":
            len(predictions),

        "raw_observations":
            int(len(trades)),          # was "trades" — renamed to flag it

        "trades":                       # kept for backward-compat readers
            len(trades),

        # ── the honest sample size ────────────────────────────────────────
        # NAMED _estimate_v1 on purpose (ChatGPT #1): this is a conservative
        # overlap-based ESTIMATE of independence, NOT ground-truth independent
        # events. A future setup-id scheme may refine it; until then, do not
        # treat this as exact.
        "independent_event_estimate_v1":
            int(n_events),

        "independent_events":           # alias kept for existing readers
            int(n_events),

        "effective_sample_size":
            int(n_events),              # conservative: ESS = event count

        # ── point estimates over EVENTS, with bootstrap CIs ───────────────
        "mean_net_return":
            float(mean_pt),
        "mean_net_return_ci":
            [float(mean_lo), float(mean_hi)],

        "median_net_return":
            float(med_pt),
        "median_net_return_ci":
            [float(med_lo), float(med_hi)],

        "positive_event_rate":
            float(pos_pt),
        "positive_event_rate_ci":
            [float(pos_lo), float(pos_hi)],

        # ── raw-edge vs after-cost edge + break-even (the key diagnostic) ─
        "gross_event_edge":
            gross_event_edge,
        "net_event_edge":
            net_event_edge,
        "per_event_cost":
            per_event_cost,
        "break_even_cost":
            break_even_cost,

        # raw per-bar mean/median kept for cross-checking only
        "raw_mean_net_return":
            float(mean_return),
        "raw_median_net_return":
            float(median_return),

        "win_rate":                     # per-bar win rate (reference)
            float(wins / len(trades)),

        "sharpe_events":
            float(sharpe_events),

        "profit_factor":
            float(profit_factor),

        "max_drawdown":
            float(broker.max_drawdown),

        "starting_equity":
            broker.starting_equity,

        "ending_equity":
            broker.equity,

        "total_pnl":
            broker.equity
            - broker.starting_equity,
    }


# ============================================================
# REGIME ANALYSIS
# ============================================================

def analyze_by_regime(
    predictions,
):

    result = {}

    dataframe = pd.DataFrame(
        [
            asdict(p)
            for p in predictions
        ]
    )

    if dataframe.empty:

        return result

    for regime, group in (
        dataframe
        .groupby("regime")
    ):

        executed = group[
            group["decision"]
            != "NO_TRADE"
        ]

        if executed.empty:

            result[regime] = {
                "observations":
                    len(group),

                "trades": 0,
            }

            continue

        # net_actual_return is ALREADY direction-signed and cost-charged by
        # directional_net_return() in run_walk_forward. Do NOT re-sign shorts
        # here — that double-flipped them and made the regime report disagree
        # with the main report. (Fix: ChatGPT batch-3 review, 2026-08-15 —
        # same bug class as calculate_report/cluster_events, this was the
        # 4th consumer that got missed the first time.)
        signed_returns = executed[
            "net_actual_return"
        ].to_numpy(dtype=float)

        result[regime] = {

            "observations":
                int(len(group)),

            "trades":
                int(len(executed)),

            "win_rate":
                float(
                    (
                        signed_returns > 0
                    ).mean()
                ),

            "mean_net_return":
                float(
                    signed_returns.mean()
                ),

            "median_net_return":
                float(
                    np.median(
                        signed_returns
                    )
                ),
        }

    return result


# ============================================================
# DATABASE
# ============================================================

class ResearchDatabase:

    def __init__(
        self,
        path: str,
        run_id: str = None,
    ):

        self.connection = (
            sqlite3.connect(path)
        )

        # research_run_id isolates one invocation's rows from every other run
        # in the same DB file (ChatGPT #7). Without it, re-running GOLD 5m on
        # three days piles three histories into `predictions` and the auditor
        # would audit a BLENDED mixture across config/cost/feature versions.
        # The auditor selects a single run_id. Default: UTC timestamp + short
        # random suffix so two runs in the same second still differ.
        self.run_id = run_id or (
            datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            + "_" + os.urandom(3).hex()
        )

        self.create_tables()
        self._migrate_add_run_id()

    def _migrate_add_run_id(self):
        """Add research_run_id to any pre-existing table that lacks it, so an
        old DB from before this change keeps working (ALTER is a no-op if the
        column already exists). Never raises."""
        for tbl in ("predictions", "experiments", "virtual_trades"):
            try:
                cols = [r[1] for r in self.connection.execute(
                    f"PRAGMA table_info({tbl})").fetchall()]
                if "research_run_id" in cols:
                    continue
                self.connection.execute(
                    f"ALTER TABLE {tbl} "
                    f"ADD COLUMN research_run_id TEXT")
            except Exception:
                pass
        try:
            self.connection.commit()
        except Exception:
            pass

    def create_tables(self):

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            predictions (

                id INTEGER PRIMARY KEY,

                research_run_id TEXT,

                timestamp TEXT,

                instrument TEXT,

                horizon INTEGER,

                regime TEXT,

                expected_return REAL,

                median_return REAL,

                win_probability REAL,

                sample_count INTEGER,

                uncertainty REAL,

                nearest_distance REAL,

                actual_return REAL,

                cost_return REAL,

                net_expected_return REAL,

                net_actual_return REAL,

                decision TEXT,

                model_version TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            experiments (

                id INTEGER PRIMARY KEY,

                created_at TEXT,

                instrument TEXT,

                timeframe TEXT,

                horizon INTEGER,

                model_version TEXT,

                report_json TEXT,

                regime_report_json TEXT,

                research_run_id TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            virtual_trades (

                id INTEGER PRIMARY KEY,

                research_run_id TEXT,

                timestamp TEXT,

                instrument TEXT,

                direction TEXT,

                entry_price REAL,

                exit_price REAL,

                expected_return REAL,

                realized_return REAL,

                pnl REAL,

                equity_after REAL,

                model_version TEXT,

                regime TEXT
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            run_spec (

                research_run_id TEXT PRIMARY KEY,

                created_at TEXT,

                instrument TEXT,

                spec_json TEXT
            )
            """
        )

        self.connection.commit()

    def save_run_spec(self, instrument: str, spec: dict) -> None:
        """Persist the CANONICAL research spec for THIS run, keyed by
        research_run_id. This is what --freeze freezes — NOT whatever CONFIG
        happens to contain later (ChatGPT 🔴 fix 2026-08-15). Freezing live
        CONFIG could record k=75 for a run actually executed at k=50; pinning
        the saved spec makes the freeze describe the experiment it confirms.
        """
        try:
            self.connection.execute(
                "INSERT OR REPLACE INTO run_spec "
                "(research_run_id, created_at, instrument, spec_json) "
                "VALUES (?,?,?,?)",
                (self.run_id, datetime.utcnow().isoformat(), instrument,
                 json.dumps(spec, default=str)))
            self.connection.commit()
        except Exception:
            pass

    def save_experiment(
        self,
        config: Config,
        horizon: int,
        report: dict,
        regime_report: dict,
    ):

        self.connection.execute(
            """
            INSERT INTO experiments (

                created_at,
                instrument,
                timeframe,
                horizon,
                model_version,
                report_json,
                regime_report_json,
                research_run_id

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (

                datetime.utcnow().isoformat(),

                config.instrument,

                config.timeframe,

                horizon,

                "knn_baseline_v1",

                # json.dumps (not str()) so report_json is valid JSON that
                # round-trips with json.loads — str() of a dict with numpy
                # floats produces Python-repr, not JSON. default=str keeps any
                # stray non-serialisable value from crashing the write.
                json.dumps(report, default=str),

                json.dumps(regime_report, default=str),

                self.run_id,
            ),
        )

        self.connection.commit()

    def save_predictions(
        self,
        predictions,
        config: Config,
    ):

        rows = []

        for p in predictions:

            rows.append(
                (
                    self.run_id,

                    p.timestamp,

                    config.instrument,

                    p.horizon,

                    p.regime,

                    p.expected_return,

                    p.median_return,

                    p.win_probability,

                    p.sample_count,

                    p.uncertainty,

                    p.nearest_distance,

                    p.actual_return,

                    p.cost_return,

                    p.net_expected_return,

                    p.net_actual_return,

                    p.decision,

                    "knn_baseline_v1",
                )
            )

        self.connection.executemany(
            """
            INSERT INTO predictions (

                research_run_id,
                timestamp,
                instrument,
                horizon,
                regime,
                expected_return,
                median_return,
                win_probability,
                sample_count,
                uncertainty,
                nearest_distance,
                actual_return,
                cost_return,
                net_expected_return,
                net_actual_return,
                decision,
                model_version

            )

            VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,

            rows,
        )

        self.connection.commit()

    def save_trades(
        self,
        trades,
    ):

        rows = [
            (
                self.run_id,
                t.timestamp,
                t.instrument,
                t.direction,
                t.entry_price,
                t.exit_price,
                t.expected_return,
                t.realized_return,
                t.pnl,
                t.equity_after,
                t.model_version,
                t.regime,
            )

            for t in trades
        ]

        self.connection.executemany(
            """
            INSERT INTO virtual_trades (

                research_run_id,
                timestamp,
                instrument,
                direction,
                entry_price,
                exit_price,
                expected_return,
                realized_return,
                pnl,
                equity_after,
                model_version,
                regime

            )

            VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,

            rows,
        )

        self.connection.commit()


# ============================================================
# MAIN RESEARCH ENGINE
# ============================================================

def run_research():

    print()
    print("=" * 72)
    print("AUTO-LEARNING MARKET RESEARCH LAB")
    print("=" * 72)

    print(
        f"Instrument : {CONFIG.instrument}"
    )

    print(
        f"Timeframe  : {CONFIG.timeframe}"
    )

    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_market_data(
        CONFIG
    )

    print(
        f"Bars loaded: {len(df):,}"
    )

    print(
        f"Start: {df['timestamp'].iloc[0]}"
    )

    print(
        f"End  : {df['timestamp'].iloc[-1]}"
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    df = build_features(
        df
    )

    # --------------------------------------------------------
    # Build forward outcomes
    # --------------------------------------------------------

    # Load the market-hours calendar (real fleet supplies one; demo may not).
    _calendar = None
    if _CC is not None:
        try:
            _calendar = _CC.load_calendar(CONFIG.calendar_path)
            CONFIG.market_calendar_version = _CC.calendar_version(
                CONFIG.calendar_path)
        except Exception:
            _calendar = None

    df = add_forward_outcomes(
        df,
        CONFIG.horizons,
        instrument=CONFIG.instrument,
        calendar=_calendar,
        require_calendar=CONFIG.require_calendar,
    )

    database = ResearchDatabase(
        CONFIG.database_file
    )

    # Persist the CANONICAL spec for THIS run so a later --freeze freezes the
    # exact experiment that produced these results, not whatever CONFIG holds
    # at freeze time (ChatGPT 🔴 fix). Uses the same builder --freeze reads.
    database.save_run_spec(
        CONFIG.instrument,
        build_research_config_for_freeze(CONFIG),
    )
    print(f"[run_spec] saved canonical spec for run_id={database.run_id}")

    # --------------------------------------------------------
    # Test every prediction horizon
    # --------------------------------------------------------

    for horizon in CONFIG.horizons:

        print()
        print("-" * 72)

        print(
            f"HORIZON: {horizon} candles"
        )

        print("-" * 72)

        predictions = run_walk_forward(
            df,
            CONFIG,
            horizon,
        )

        # ----------------------------------------------------
        # Virtual broker
        # ----------------------------------------------------

        virtual_broker = (
            VirtualBroker(
                CONFIG.starting_equity
            )
        )

        for prediction in predictions:

            # Locate the current timestamp.
            row = df[
                df["timestamp"]
                == pd.Timestamp(
                    prediction.timestamp
                )
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            entry_price = float(
                row["close"]
            )

            # Exit price is known historically
            # ONLY because this is the simulator.
            #
            # The live system must never use this future
            # price when generating the prediction.

            exit_price = (
                entry_price
                * (
                    1
                    + prediction.actual_return
                )
            )

            virtual_broker.execute(

                timestamp=
                    prediction.timestamp,

                instrument=
                    CONFIG.instrument,

                direction=
                    prediction.decision,

                entry_price=
                    entry_price,

                exit_price=
                    exit_price,

                expected_return=
                    prediction.net_expected_return,

                risk_fraction=
                    CONFIG.risk_per_trade,

                model_version=
                    "knn_baseline_v1",

                regime=
                    prediction.regime,

                cost_return=
                    prediction.cost_return,
            )

        # ----------------------------------------------------
        # Reports
        # ----------------------------------------------------

        report = calculate_report(
            predictions,
            virtual_broker,
            bar_seconds=timeframe_to_seconds(CONFIG.timeframe),
            config=CONFIG,
        )

        regime_report = (
            analyze_by_regime(
                predictions
            )
        )

        print()
        print("OVERALL REPORT")
        print(
            "-------------"
        )

        for key, value in report.items():

            print(
                f"{key:25} {value}"
            )

        print()
        print("REGIME REPORT")
        print(
            "-------------"
        )

        for regime, values in (
            regime_report.items()
        ):

            print()
            print(regime)

            for key, value in (
                values.items()
            ):

                print(
                    f"    {key:20} {value}"
                )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        database.save_predictions(
            predictions,
            CONFIG,
        )

        database.save_trades(
            virtual_broker.trades
        )

        database.save_experiment(
            CONFIG,
            horizon,
            report,
            regime_report,
        )

    print()
    print("=" * 72)
    print("RESEARCH RUN COMPLETE")
    print("=" * 72)

    print()
    print(
        "NO REAL BROKER ORDERS WERE PLACED."
    )

    print(
        f"Database: {CONFIG.database_file}"
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Replace the demo data source with the existing "
        "Capital.com market-data integration."
    )

    print(
        "Then replace the placeholder cost values with "
        "actual instrument-specific execution costs."
    )


# ============================================================
# LIVE ADVISORY INTERFACE
# ============================================================

def generate_live_research_advice(
    instrument: str,
):
    """
    Future live-paper-trading interface.

    This is intentionally incomplete.

    It should eventually:

        1. Request current market state.
        2. Build the same features used during training.
        3. Load the CURRENT APPROVED model.
        4. Produce a prediction.
        5. Return advice.
        6. NEVER place a real order.

    Claude should connect the existing application here.
    """

    # Price request on instrument HERE

    current_price = (
        request_current_price(
            instrument
        )
    )

    return {
        "instrument": instrument,

        "price": current_price,

        "status":
            "LIVE_RESEARCH_ONLY",

        "decision":
            "NO_TRADE",

        "message":
            "Live feature/model integration "
            "has not yet been connected.",
    }


# ============================================================
# FROZEN-OOS FREEZE COMMAND  (deliberate, explicit — NOT automatic)
# ============================================================
#
# ChatGPT ruling (2026-08-15): a freeze is a scientific COMMITMENT to a
# specific specification + immutable OOS cutoff. It is NEVER a side-effect of
# run_research() (that pollutes the ledger and blurs exploration vs
# confirmation). It is a separate, explicit command:
#
#     python research.py --freeze --cutoff 2026-06-01
#
# The DB freeze_ledger is the SINGLE source of truth (via the auditor's
# register_freeze). A JSON file, if written, is a derivative export only.
# ============================================================

def build_research_config_for_freeze(config: "Config") -> dict:
    """Assemble the EXACT code-defined recipe that the freeze will hash. Pulls
    real values from Config + module-level rules so the spec hash reflects what
    the research engine actually does — not a hand-typed guess that could drift.
    """
    return {
        "feature_set": list(FEATURE_COLUMNS),
        "feature_transformations": "zscore_on_train_mean_std",
        "k": config.k_neighbors,
        "horizons": list(config.horizons),
        "no_trade_threshold": config.minimum_net_edge,
        # ChatGPT 🟡: minimum_net_edge alone is NOT the decision threshold. The
        # actual rule is cost_return + minimum_net_edge, and recording it makes
        # the re-costing contract unambiguous.
        "decision_rule": (
            "LONG if expected_return > (cost_return + minimum_net_edge); "
            "SHORT if expected_return < -(cost_return + minimum_net_edge); "
            "else NO_TRADE"),
        "minimum_net_edge": config.minimum_net_edge,
        "regime_rules": "trend_sign x (volatility_48 vs rolling500 median)",
        "training_window_rules": (
            f"expanding to iloc[:position-horizon]; "
            f"min_training_bars={config.minimum_training_bars}"),
        "retrain_cadence": config.retrain_every_bars,
        "data_selection_rules": (
            f"timeframe={config.timeframe}; source={getattr(config,'data_source','UNKNOWN')}"),
        # ChatGPT 🟡: there are two distinct seeds. The neighbour model is
        # deterministic; these two are the reproducibility-relevant ones.
        "random_seed": 42,                # demo-data generator seed
        "bootstrap_seed": 12345,          # block_bootstrap_ci default seed
        "random_seed_semantics": "random_seed=demo_data; bootstrap_seed=block_bootstrap",
        "cost_model_version": getattr(config, "cost_model_version",
                                      "uncalibrated_zero_v0"),
        # target definition (the #1 decision we settled — explicit + auditable)
        "target": "forward_close_return",
        "target_orientation": "LONG",
        "target_cost_adjusted": False,
        # decision-timing convention: features assume the full current candle
        # is known at decision time (bar close). Prevents a future integration
        # from silently treating this as an intrabar model.
        "decision_timing": "BAR_CLOSE",
        # market-hours target validity + calendar version (frozen so "what
        # counts as a valid trade" cannot change after seeing results).
        "market_calendar_version": getattr(config, "market_calendar_version",
                                           "cal_none"),
        "target_validity_rule": getattr(config, "target_validity_rule", ""),
        # data provenance snapshot (set by the fleet runner from the cached
        # history hash; 'unwired' until real-data wiring populates it).
        "data_snapshot_hash": getattr(config, "data_snapshot_hash", "unwired"),
    }


def _load_saved_run_spec(database_file: str, run_id: str = None):
    """Load the canonical spec persisted for a research run. If run_id is None,
    use the MOST RECENT run. Returns (run_id, spec_dict) or (None, None)."""
    try:
        conn = sqlite3.connect(database_file)
        if run_id:
            row = conn.execute(
                "SELECT research_run_id, spec_json FROM run_spec "
                "WHERE research_run_id = ?", (run_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT research_run_id, spec_json FROM run_spec "
                "ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        if not row:
            return None, None
        return row[0], json.loads(row[1])
    except Exception:
        return None, None


def freeze_command(cutoff: str, model_freeze_id: str = None,
                   database_file: str = None, export_json: str = None,
                   run_id: str = None) -> int:
    """Create an immutable freeze in the auditor's freeze_ledger. Returns a
    shell exit code (0 ok, non-zero on refusal/error).

    CRITICAL (ChatGPT 🔴 fix): freezes the SAVED spec of a specific research
    run, NOT whatever CONFIG contains now. This guarantees the freeze describes
    the experiment that produced the result being confirmed."""
    if not cutoff:
        print("ERROR: --freeze requires an explicit --cutoff "
              "(e.g. --cutoff 2026-06-01). Refusing to guess a boundary.")
        return 2

    try:
        ts = pd.to_datetime(cutoff, utc=True)
        cutoff_iso = ts.isoformat()
    except Exception as e:
        print(f"ERROR: could not parse --cutoff '{cutoff}': {e}")
        return 2

    try:
        import model_auditor as MA
    except Exception as e:
        print(f"ERROR: cannot import model_auditor to register the freeze: {e}")
        return 3

    cfg = CONFIG
    db = database_file or getattr(cfg, "database_file", "market_research.db")

    # Load the SAVED spec of the run being frozen — never live CONFIG.
    frozen_run_id, saved_config = _load_saved_run_spec(db, run_id)
    if saved_config is None:
        print("ERROR: no saved research run spec found to freeze. Run "
              "`python research.py` first (it persists a canonical spec per "
              "run), then freeze a specific --run-id. Refusing to freeze live "
              "CONFIG, which may not match any actual research result.")
        return 5

    spec = MA.build_freeze_spec(cfg.instrument, cutoff_iso, saved_config)

    fid = model_freeze_id or (
        f"{cfg.instrument}_freeze_{ts.strftime('%Y%m%d')}_{frozen_run_id[:8]}")

    try:
        result = MA.register_freeze(db, fid, spec)
    except MA.InvalidFreeze as e:
        print(f"REFUSED: {e}")
        return 4

    spec_hash = MA.canonical_spec_hash(spec)
    print("=" * 68)
    print("FREEZE REGISTERED (immutable)")
    print("=" * 68)
    print(f"  model_freeze_id      : {fid}")
    print(f"  frozen research run  : {frozen_run_id}")
    print(f"  instrument           : {cfg.instrument}")
    print(f"  oos_cutoff_timestamp : {cutoff_iso}")
    print(f"  spec_hash            : {spec_hash}")
    print(f"  status               : {result.get('status')}")
    print(f"  db (source of truth) : {db}")
    print()
    print("  target      : forward_close_return (LONG-oriented, cost-free)")
    print("  decision    : BAR_CLOSE; LONG/SHORT if expected_return beyond")
    print("                (cost_return + minimum_net_edge), else NO_TRADE")
    print("  Re-cost freely; direction+cost live in the decision layer.")
    print()
    print("  Froze the SAVED spec of the run above — not live CONFIG.")
    print("  A re-tune requires a NEW id with a strictly LATER cutoff.")

    if export_json:
        try:
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump({"model_freeze_id": fid, "spec_hash": spec_hash,
                           "frozen_research_run_id": frozen_run_id,
                           "spec": spec, "_note":
                           "DERIVATIVE EXPORT of freeze_ledger — DB is "
                           "authoritative"}, f, indent=2, default=str)
            print(f"\n  (derivative JSON written to {export_json})")
        except Exception as e:
            print(f"\n  WARNING: could not write JSON export: {e}")
    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="PowerTrader research lab (offline, advisory-only).")
    parser.add_argument("--freeze", action="store_true",
                        help="Register an immutable frozen-OOS spec (deliberate "
                             "confirmatory commitment; NOT part of a normal run).")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="OOS cutoff instant, e.g. 2026-06-01. REQUIRED "
                             "with --freeze; everything after is frozen holdout.")
    parser.add_argument("--freeze-id", type=str, default=None,
                        help="Optional explicit MODEL_FREEZE_ID.")
    parser.add_argument("--db", type=str, default=None,
                        help="Research DB path (defaults to the configured DB).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Freeze the saved spec of THIS research run "
                             "(defaults to the most recent run). The freeze "
                             "records the run's spec, never live CONFIG.")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Optional derivative JSON export of the manifest "
                             "(DB remains authoritative).")
    args = parser.parse_args()

    if args.freeze:
        # Deliberate freeze — does NOT run research.
        raise SystemExit(freeze_command(
            args.cutoff, args.freeze_id, args.db, args.export_json,
            args.run_id))

    # Normal exploratory run — MUST NOT freeze.
    run_research()
