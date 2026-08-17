"""
MODEL AUDITOR
=============

The Model Auditor is deliberately hostile to the trading learner.

It does NOT train models.
It does NOT place trades.
It does NOT decide market direction.

Its job is to answer:

    "Does this model have enough evidence to deserve promotion?"

The auditor compares:

    1. Model performance
    2. No-trade baseline
    3. Always-long baseline
    4. Always-short baseline
    5. Random-direction baseline
    6. Different market regimes
    7. Different time periods
    8. Sample size
    9. Parameter stability
    10. Drawdown
    11. Cost sensitivity
    12. Out-of-sample performance

IMPORTANT:

A profitable backtest does NOT automatically pass.

This auditor's possible verdicts are:

    BLOCKED_UNCALIBRATED_COST   (costs not yet real → nothing may promote)
    REJECT                      (enough evidence, and it failed)
    INCONCLUSIVE                (NOT enough evidence to judge — not a failure)
    INVESTIGATE                 (passes basics but something needs a human look)
    PAPER_TEST                  (the STRONGEST verdict this auditor can issue)

PROMOTE is deliberately NOT a verdict this auditor can return. Promotion to
live trading requires evidence this backtest auditor cannot provide — namely
real paper-trading performance — and must be decided by a SEPARATE promotion
system layered above this one:

    REJECT → INVESTIGATE → PAPER_TEST → [separate paper-perf auditor] → PROMOTE

Only PAPER_TEST should ever reach that paper-trading layer.

Real broker execution is deliberately absent.
"""


from __future__ import annotations

import sqlite3
import json
import math
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# CANONICAL SIGNED-RETURN — MUST STAY IN SYNC WITH research.py
# (peterpt + ChatGPT + Claude 2026-08-15)
#
# raw_return is the LONG-oriented forward return.
#   LONG  →  +raw - cost
#   SHORT →  -raw - cost
# Orient to direction FIRST, then charge cost. Never negate a value that has
# already been through this function. Duplicated (not imported) so the auditor
# has zero coupling to research.py's import side-effects; if you change one,
# change both.
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


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_FILE = "market_research.db"

# ── Sample-size floors ────────────────────────────────────────────────────
# The HONEST unit is independent events, not candles/trades (ChatGPT #4/#6 of
# the earlier review). Raw-trade floors are kept only as a secondary guard.
MIN_TRADES = 50                 # raw executed rows (reference floor)

# Independent-event floors drive the INCONCLUSIVE gate. These are OPERATIONAL
# gates, not statistical laws (ChatGPT). Tunable.
MIN_EVENTS_INCONCLUSIVE = 30    # < this -> INCONCLUSIVE (not REJECT)
MIN_EVENTS_WEAK        = 100    # 30-99  -> at most WEAK / INVESTIGATE
MIN_EVENTS_MEANINGFUL  = 250    # 250+   -> eligible for the strongest verdicts

MIN_REGIME_TRADES = 20
MIN_PERIOD_TRADES = 20

# Minimum acceptable improvement over random direction (now ACTUALLY enforced).
MIN_EDGE_OVER_RANDOM = 0.00005

# Economic hurdle: the 95% CI LOWER bound must clear this, not merely be >0
# (ChatGPT hostile #6). A statistically detectable but economically trivial
# edge is not tradeable after spread uncertainty/slippage/fees/degradation.
# Per-observation return units. Tunable; will be revisited with real costs.
MIN_ECONOMIC_HURDLE = 0.0000   # start at 0 (CI must be strictly >0); raise
                               # to a real hurdle once Capital costs are in.

# Random/permutation significance: model must beat random at this level.
MAX_PROB_RANDOM_BEATS_MODEL = 0.05   # p-value style gate

# Maximum acceptable equity drawdown for the strongest verdict (now computed
# as true peak-to-trough equity drawdown, ChatGPT #7).
MAX_ALLOWED_DRAWDOWN = 0.25

# ── FROZEN-OOS CONTRACT (ChatGPT ruling 2026-08-15) ───────────────────────
# The OOS mechanism is a CALENDAR cutoff frozen at freeze-creation time, NOT a
# fraction. Everything AT/BEFORE the cutoff is development; everything AFTER is
# the untouched final holdout. The cutoff is immutable per MODEL_FREEZE_ID.
#
#   OOS_CUTOFF_TIMESTAMP  — fixed ISO instant, chosen once, never moved.
#   MODEL_FREEZE_ID       — pins the RECIPE (spec), not a weight blob.
#
# If the holdout has too few events -> INCONCLUSIVE. NEVER move the cutoff to
# make a result appear. A re-tune after peeking requires a NEW freeze with a
# strictly LATER cutoff (enforced by the freeze ledger, ruling #5).
AUDITOR_VERSION       = "auditor_2026-08-15_frozen_oos_v1"
EVENT_CLUSTERING_VERSION = "cluster_events_v1+block_floor"
MIN_OOS_EVENTS        = MIN_EVENTS_INCONCLUSIVE   # holdout must clear this too
FREEZE_LEDGER_TABLE   = "freeze_ledger"

# Legacy label kept only for the diagnostic walk-forward report section.
OOS_POLICY = "frozen_calendar_cutoff_v1"

# Cost stress tests.
COST_MULTIPLIERS = [
    1.0,
    1.25,
    1.50,
    2.0,
    3.0,
]

# Minimum percentage of cost scenarios that must remain profitable.
MIN_COST_SURVIVAL = 0.60

# Number of random-direction simulations.
RANDOM_SIMULATIONS = 2000

# Shuffle (permutation) test iterations.
SHUFFLE_ITERATIONS = 2000


# ============================================================
# DATABASE
# ============================================================

def load_predictions(
    database_file: str,
    model_version: str | None = None,
    research_run_id: str | None = None,
) -> pd.DataFrame:

    connection = sqlite3.connect(
        database_file
    )

    # Pull research_run_id too (may be absent in very old DBs; guarded below).
    query = """
        SELECT
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
            model_version,
            research_run_id

        FROM predictions
    """

    clauses = []
    parameters = []

    if model_version is not None:
        clauses.append("model_version = ?")
        parameters.append(model_version)

    if research_run_id is not None:
        clauses.append("research_run_id = ?")
        parameters.append(research_run_id)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += """
        ORDER BY timestamp
    """

    try:
        dataframe = pd.read_sql_query(
            query,
            connection,
            params=parameters,
        )
    except Exception:
        # Old DB without research_run_id column — retry without it.
        connection.close()
        connection = sqlite3.connect(database_file)
        q2 = query.replace(",\n            research_run_id", "")
        dataframe = pd.read_sql_query(q2, connection, params=parameters)

    connection.close()

    if dataframe.empty:

        raise ValueError(
            "No predictions found in database."
        )

    dataframe["timestamp"] = (
        pd.to_datetime(
            dataframe["timestamp"],
            utc=True,
        )
    )

    return dataframe


def persist_audit_verdicts(database_file: str, reports: list,
                           research_run_id: str = None) -> None:
    """Persist each (instrument, horizon) verdict to an authoritative table the
    research advisor reads. The advisor consumes THIS — never the raw
    predictions — so 'is the model allowed to advise?' has one source of truth
    (the auditor's verdict), enforcing the invariant that an uncalibrated /
    unqualified model can send NOTHING to the engines.

    Stores the WORST-case is fine to be strict: the advisor will treat anything
    below PAPER_TEST as SILENT anyway. We store per (instrument,horizon)."""
    try:
        conn = sqlite3.connect(database_file)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_verdict (
                instrument        TEXT,
                horizon           INTEGER,
                verdict           TEXT,
                cost_model_status TEXT,
                auditor_ceiling   TEXT,
                effective_events  INTEGER,
                event_mean        REAL,
                research_run_id   TEXT,
                audited_at        TEXT,
                PRIMARY KEY (instrument, horizon)
            )
        """)
        for r in reports:
            dec = r.get("decision", {})
            verdict = dec.get("verdict") if isinstance(dec, dict) else str(dec)
            ev = r.get("event_stats", {}) or {}
            conn.execute(
                "INSERT OR REPLACE INTO audit_verdict "
                "(instrument,horizon,verdict,cost_model_status,auditor_ceiling,"
                " effective_events,event_mean,research_run_id,audited_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (str(r.get("instrument")), int(r.get("horizon", 0)),
                 verdict, r.get("cost_model_status"),
                 r.get("auditor_ceiling"),
                 int(ev.get("effective_events") or 0),
                 float(ev.get("event_mean") or 0.0),
                 research_run_id,
                 datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
    except Exception:
        pass


def latest_run_id(database_file: str) -> str | None:
    """Return the research_run_id of the most recent run, so the auditor
    audits ONE run by default instead of a blended mixture of every run ever
    stored (ChatGPT #15 / run-isolation). None if the column is absent."""
    try:
        conn = sqlite3.connect(database_file)
        row = conn.execute(
            "SELECT research_run_id FROM predictions "
            "WHERE research_run_id IS NOT NULL "
            "ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def latest_run_id_per_instrument(database_file: str) -> dict:
    """Return {instrument: latest research_run_id} — the most recent run for
    EACH instrument. The supervisor researches instruments sequentially, each
    into its own run_id, so a single global 'latest run' only covers the last
    instrument researched. Auditing per-instrument-latest ensures ALL six
    instruments are actually audited daily (ChatGPT CRITICAL finding)."""
    out = {}
    try:
        conn = sqlite3.connect(database_file)
        rows = conn.execute(
            "SELECT instrument, research_run_id, MAX(id) "
            "FROM predictions WHERE research_run_id IS NOT NULL "
            "GROUP BY instrument").fetchall()
        conn.close()
        for inst, rid, _maxid in rows:
            if inst and rid:
                out[inst] = rid
    except Exception:
        pass
    return out


# ============================================================
# DATA INTEGRITY  (ChatGPT #3/#15 — do not trust the DB blindly)
# ============================================================

def verify_data_integrity(dataframe: pd.DataFrame) -> dict:
    """Independently RECONSTRUCT net_actual_return from raw actual_return +
    decision + cost_return, and compare to the stored value. A hostile auditor
    must not faithfully audit bad data — and this codebase has ALREADY had two
    signed-return bugs. If the stored field disagrees with the reconstruction,
    that is a DATA_INTEGRITY_FAILURE and the model is REJECTed before any
    performance claim is considered.

    Returns {"ok": bool, "max_abs_diff": float, "n_mismatch": int, "n": int}.
    """
    executed = dataframe[dataframe["decision"] != "NO_TRADE"]
    if executed.empty:
        return {"ok": True, "max_abs_diff": 0.0, "n_mismatch": 0, "n": 0}

    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)
    dirs = executed["decision"].astype(str).to_numpy()
    stored = executed["net_actual_return"].to_numpy(dtype=float)

    recon = np.array([
        directional_net_return(float(r), str(d), float(c))
        for r, d, c in zip(raw, dirs, cost)
    ], dtype=float)

    diff = np.abs(recon - stored)
    tol = 1e-9
    n_mismatch = int((diff > tol).sum())
    return {
        "ok": n_mismatch == 0,
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "n_mismatch": n_mismatch,
        "n": int(len(diff)),
    }


# ============================================================
# FROZEN-OOS CONTRACT  (ChatGPT ruling — freeze protocol)
# ============================================================
#
# This is NOT merely a split. It is a small experiment-lifecycle protocol that
# closes the loop where 'OOS' evidence gets repeatedly fed back into model
# selection. Four machine-enforced guarantees:
#   1. cutoff is a fixed calendar instant, immutable per freeze (ruling #1)
#   2. the FROZEN RECIPE (spec) is hashed; drift is detected (ruling #2/#3)
#   3. only the post-cutoff holdout can gate PAPER_TEST (ruling #4)
#   4. each MODEL_FREEZE_ID may be holdout-audited EXACTLY ONCE (ruling #5)
#
# A hash cannot stop a human peeking-then-retuning; the ledger's one-audit rule
# is what turns 'please don't peek twice' into an enforced state transition.
# ============================================================

class FrozenHoldoutAlreadyConsumed(Exception):
    """Raised when a MODEL_FREEZE_ID's single holdout audit has already been
    spent. A re-tune requires a NEW freeze with a strictly LATER cutoff."""


class InvalidFreeze(Exception):
    """Raised when a freeze manifest fails verification (missing, spec-hash
    mismatch, moved cutoff, or version mismatch)."""


def canonical_spec_hash(spec: dict) -> str:
    """Deterministic hash of the FROZEN RECIPE. Canonical JSON (sorted keys,
    no whitespace) so logically-identical specs hash identically and any
    change to a frozen field is detectable."""
    canon = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                       default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def build_freeze_spec(instrument: str,
                      oos_cutoff_timestamp: str,
                      research_config: dict) -> dict:
    """Assemble the spec that MODEL_FREEZE_ID pins. Freezes the RECIPE, not a
    weight blob (ruling #2). research_config carries the fields the research
    lab actually used (feature set, k, horizons, thresholds, cost/clustering
    versions, seeds, retraining cadence). Missing fields are recorded as null
    so the hash still reflects 'this was unspecified at freeze time'."""
    return {
        "instrument": instrument,
        "oos_cutoff_timestamp": oos_cutoff_timestamp,
        "feature_set": research_config.get("feature_set"),
        "feature_transformations": research_config.get("feature_transformations"),
        "k": research_config.get("k"),
        "horizons": research_config.get("horizons"),
        "no_trade_threshold": research_config.get("no_trade_threshold"),
        "regime_rules": research_config.get("regime_rules"),
        "training_window_rules": research_config.get("training_window_rules"),
        "retrain_cadence": research_config.get("retrain_cadence"),
        "data_selection_rules": research_config.get("data_selection_rules"),
        "random_seed": research_config.get("random_seed"),
        "cost_model_version": research_config.get("cost_model_version"),
        # ChatGPT ruling (2026-08-15): the frozen spec MUST capture the target
        # definition and the decision-timing convention, not just model params.
        # target: the learner predicts the RAW, LONG-oriented, COST-FREE forward
        # return; direction + costs are applied afterward in the decision layer
        # (so the same predictions can be re-costed). decision_timing=BAR_CLOSE
        # records that features assume the full current candle is known at
        # decision time — preventing a future integration from silently
        # treating this as an intrabar model.
        "target": research_config.get("target", "forward_close_return"),
        "target_orientation": research_config.get("target_orientation", "LONG"),
        "target_cost_adjusted": research_config.get("target_cost_adjusted",
                                                    False),
        "decision_timing": research_config.get("decision_timing", "BAR_CLOSE"),
        "event_clustering_version": EVENT_CLUSTERING_VERSION,
        "auditor_version": AUDITOR_VERSION,
    }


def _ensure_freeze_ledger(conn) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {FREEZE_LEDGER_TABLE} (
            model_freeze_id         TEXT PRIMARY KEY,
            spec_hash               TEXT,
            oos_cutoff_timestamp    TEXT,
            created_at              TEXT,
            status                  TEXT,
            holdout_audit_count     INTEGER DEFAULT 0,
            holdout_audit_timestamp TEXT,
            holdout_result_hash     TEXT,
            superseded_by           TEXT
        )
    """)
    conn.commit()


def register_freeze(database_file: str,
                    model_freeze_id: str,
                    spec: dict) -> dict:
    """Create an immutable freeze manifest row. If the id already exists, the
    stored spec_hash and cutoff MUST match (a freeze is immutable) — otherwise
    someone is trying to redefine a freeze in place, which is rejected."""
    conn = sqlite3.connect(database_file)
    _ensure_freeze_ledger(conn)
    spec_hash = canonical_spec_hash(spec)
    cutoff = spec["oos_cutoff_timestamp"]
    row = conn.execute(
        f"SELECT spec_hash, oos_cutoff_timestamp FROM {FREEZE_LEDGER_TABLE} "
        f"WHERE model_freeze_id = ?", (model_freeze_id,)).fetchone()
    if row is not None:
        if row[0] != spec_hash or row[1] != cutoff:
            conn.close()
            raise InvalidFreeze(
                f"freeze '{model_freeze_id}' already exists with a DIFFERENT "
                f"spec/cutoff — a freeze is immutable; create a new id with a "
                f"later cutoff instead")
        conn.close()
        return {"model_freeze_id": model_freeze_id, "spec_hash": spec_hash,
                "oos_cutoff_timestamp": cutoff, "status": "exists"}
    conn.execute(
        f"INSERT INTO {FREEZE_LEDGER_TABLE} "
        f"(model_freeze_id, spec_hash, oos_cutoff_timestamp, created_at, "
        f" status, holdout_audit_count) VALUES (?,?,?,?,?,0)",
        (model_freeze_id, spec_hash, cutoff,
         datetime.now(timezone.utc).isoformat(), "FROZEN"))
    conn.commit()
    conn.close()
    return {"model_freeze_id": model_freeze_id, "spec_hash": spec_hash,
            "oos_cutoff_timestamp": cutoff, "status": "created"}


def verify_freeze(database_file: str,
                  model_freeze_id: str,
                  spec: dict) -> dict:
    """Verify the manifest before auditing (ruling #3). Checks existence,
    spec-hash match, and cutoff immutability. Returns the ledger row + ok flag;
    raises InvalidFreeze on hard failure."""
    conn = sqlite3.connect(database_file)
    _ensure_freeze_ledger(conn)
    row = conn.execute(
        f"SELECT model_freeze_id, spec_hash, oos_cutoff_timestamp, status, "
        f"holdout_audit_count FROM {FREEZE_LEDGER_TABLE} "
        f"WHERE model_freeze_id = ?", (model_freeze_id,)).fetchone()
    conn.close()
    if row is None:
        raise InvalidFreeze(f"no freeze registered for '{model_freeze_id}'")
    expected_hash = canonical_spec_hash(spec)
    if row[1] != expected_hash:
        raise InvalidFreeze(
            f"SPEC DRIFT: current spec hash {expected_hash[:12]}… != frozen "
            f"{row[1][:12]}… — the recipe changed after freezing")
    if row[2] != spec["oos_cutoff_timestamp"]:
        raise InvalidFreeze(
            f"cutoff moved: manifest {row[2]} != requested "
            f"{spec['oos_cutoff_timestamp']}")
    return {"ok": True, "model_freeze_id": row[0], "spec_hash": row[1],
            "oos_cutoff_timestamp": row[2], "status": row[3],
            "holdout_audit_count": int(row[4] or 0)}


def consume_holdout_audit(database_file: str,
                          model_freeze_id: str,
                          result_hash: str) -> None:
    """Atomically spend this freeze's SINGLE holdout audit (ruling #5). If it
    was already consumed, raise FrozenHoldoutAlreadyConsumed — a re-tune must
    create a NEW freeze with a later cutoff, not re-audit this holdout."""
    conn = sqlite3.connect(database_file)
    _ensure_freeze_ledger(conn)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT holdout_audit_count FROM {FREEZE_LEDGER_TABLE} "
            f"WHERE model_freeze_id = ?", (model_freeze_id,)).fetchone()
        if row is None:
            conn.rollback(); conn.close()
            raise InvalidFreeze(f"no freeze '{model_freeze_id}' to consume")
        if int(row[0] or 0) >= 1:
            conn.rollback(); conn.close()
            raise FrozenHoldoutAlreadyConsumed(
                f"freeze '{model_freeze_id}' holdout already audited once — "
                f"create a NEW freeze with a strictly LATER cutoff to re-test")
        conn.execute(
            f"UPDATE {FREEZE_LEDGER_TABLE} SET holdout_audit_count = 1, "
            f"holdout_audit_timestamp = ?, holdout_result_hash = ?, "
            f"status = 'HOLDOUT_CONSUMED' WHERE model_freeze_id = ?",
            (datetime.now(timezone.utc).isoformat(), result_hash,
             model_freeze_id))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def split_frozen_oos(dataframe: pd.DataFrame,
                     oos_cutoff_timestamp: str):
    """Split into (development, frozen_holdout) by the immutable calendar
    cutoff. AT/BEFORE cutoff = development (diagnostic walk-forward); AFTER =
    the untouched holdout that alone can gate PAPER_TEST (ruling #1/#4)."""
    cutoff = pd.to_datetime(oos_cutoff_timestamp, utc=True)
    ts = pd.to_datetime(dataframe["timestamp"], utc=True)
    dev = dataframe[ts <= cutoff].copy()
    holdout = dataframe[ts > cutoff].copy()
    return dev, holdout


# ============================================================
# BASIC RETURN CALCULATION
# ============================================================

def signed_model_returns(
    dataframe: pd.DataFrame,
) -> np.ndarray:

    executed = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ].copy()

    if executed.empty:

        return np.array([])

    # net_actual_return is written by research.py's directional_net_return():
    # it is ALREADY direction-signed and cost-charged. Read it as-is. The old
    # `-net_actual_return` for shorts double-signed it and flipped the cost —
    # the exact bug fixed in research.py on 2026-08-15. Keep these consistent.
    returns = executed["net_actual_return"].to_numpy()

    return returns.astype(float)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    returns: np.ndarray,
) -> dict:

    if len(returns) == 0:

        return {
            "trades": 0
        }

    wins = (
        returns > 0
    ).sum()

    losses = (
        returns < 0
    ).sum()

    mean_return = (
        returns.mean()
    )

    median_return = (
        np.median(returns)
    )

    std = (
        returns.std(ddof=1)
        if len(returns) > 1
        else 0
    )

    sharpe_like = (

        mean_return
        / std
        * math.sqrt(len(returns))

        if std > 0
        else 0
    )

    # TRUE peak-to-trough EQUITY drawdown (compounding), not additive cumsum.
    # 0.25 now means 25% equity drawdown. (ChatGPT #7)
    equity = np.cumprod(1.0 + returns)

    running_max = np.maximum.accumulate(
        equity
    )

    dd = (
        equity - running_max
    ) / running_max

    max_drawdown = (
        float(-dd.min())
        if len(dd)
        else 0.0
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

    return {

        "trades":
            int(len(returns)),

        "win_rate":
            float(
                wins / len(returns)
            ),

        "loss_rate":
            float(
                losses / len(returns)
            ),

        "mean_return":
            float(mean_return),

        "median_return":
            float(median_return),

        "std":
            float(std),

        "sharpe_like":
            float(sharpe_like),

        "profit_factor":
            float(profit_factor),

        "max_drawdown":
            float(max_drawdown),

        "total_return":
            float(
                returns.sum()
            ),
    }


# ============================================================
# BASELINES
# ============================================================

def always_long_baseline(
    dataframe: pd.DataFrame,
) -> np.ndarray:

    executed = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ]

    if executed.empty:

        return np.array([])

    # A TRUE always-long control: go LONG on every executed opportunity, using
    # the RAW forward return + that row's cost — NOT the model's already-signed
    # net (which was signed by the model's OWN chosen direction). Deriving a
    # baseline by re-signing the model's net would contaminate the control with
    # the model's decisions. (2026-08-15)
    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)
    return raw - cost


def always_short_baseline(
    dataframe: pd.DataFrame,
) -> np.ndarray:

    executed = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ]

    if executed.empty:

        return np.array([])

    # True always-short control: -raw - cost on every executed opportunity.
    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)
    return -raw - cost


def random_baseline(
    dataframe: pd.DataFrame,
    simulations: int = RANDOM_SIMULATIONS,
):

    executed = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ]

    if executed.empty:

        return {
            "mean_random_return": 0,
            "std_random_return": 0,
            "probability_beating_model": 0,
        }

    # Random control on the SAME opportunities: pick LONG/SHORT at random per
    # row, using raw+cost. Charging cost the same way as the model keeps the
    # comparison fair (the model's edge must beat random DIRECTION, not just
    # random cost). (2026-08-15)
    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)

    model_returns = (
        signed_model_returns(
            dataframe
        )
    )

    random_results = []

    rng = np.random.default_rng(42)

    for _ in range(simulations):

        directions = rng.choice(
            [-1, 1],
            size=len(raw),
        )

        # +raw-cost when long (+1), -raw-cost when short (-1)
        simulated = (
            raw * directions - cost
        )

        random_results.append(
            simulated.mean()
        )

    random_results = np.array(
        random_results
    )

    model_mean = float(model_returns.mean()) if len(model_returns) else 0.0
    prob_random_beats = float((random_results >= model_mean).mean())

    return {

        "mean_random_return":
            float(
                random_results.mean()
            ),

        "std_random_return":
            float(
                random_results.std()
            ),

        "random_95th_percentile":
            float(np.percentile(random_results, 95)),

        "edge_over_random":
            float(model_mean - float(random_results.mean())),

        # p-value style: fraction of random runs matching/beating the model.
        "probability_random_beats_model":
            prob_random_beats,

        # kept for backward-compat readers
        "probability_beating_model":
            prob_random_beats,
    }


# ============================================================
# NO-TRADE ANALYSIS
# ============================================================

def analyze_no_trade_filter(
    dataframe: pd.DataFrame,
) -> dict:

    trade_rows = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ]

    no_trade_rows = dataframe[
        dataframe["decision"]
        == "NO_TRADE"
    ]

    trade_returns = (
        signed_model_returns(
            dataframe
        )
    )

    return {

        "total_predictions":
            int(len(dataframe)),

        "trade_count":
            int(len(trade_rows)),

        "no_trade_count":
            int(len(no_trade_rows)),

        "trade_percentage":
            float(
                len(trade_rows)
                / len(dataframe)
            ),

        "average_traded_edge":
            float(
                trade_returns.mean()
            )
            if len(trade_returns)
            else 0,

        "average_predicted_edge":
            float(
                trade_rows[
                    "net_expected_return"
                ].mean()
            )
            if not trade_rows.empty
            else 0,
    }


# ============================================================
# REGIME ANALYSIS
# ============================================================

def analyze_regimes(
    dataframe: pd.DataFrame,
) -> dict:

    result = {}

    for regime, group in (
        dataframe
        .groupby("regime")
    ):

        # Reconstruct independently (not the learner's stored net), consistent
        # with the auditor-independence principle (ChatGPT #15).
        returns = reconstruct_signed_returns(group)

        metrics = calculate_metrics(
            returns
        )

        # Flag regimes below the sample floor as INSUFFICIENT_DATA so nobody
        # reads a mean from 8 trades as meaningful (ChatGPT #9).
        if metrics.get("trades", 0) < MIN_REGIME_TRADES:
            metrics["status"] = "INSUFFICIENT_DATA"
        else:
            metrics["status"] = "OK"

        result[regime] = metrics

    return result


# ============================================================
# TIME PERIOD ANALYSIS
# ============================================================

def analyze_periods(
    dataframe: pd.DataFrame,
    periods: int = 5,
) -> dict:

    if dataframe.empty:

        return {}

    x = dataframe.sort_values(
        "timestamp"
    ).copy()

    x["period"] = pd.qcut(
        np.arange(len(x)),
        q=min(periods, len(x)),
        labels=False,
        duplicates="drop",
    )

    result = {}

    for period, group in (
        x.groupby("period")
    ):

        returns = (
            signed_model_returns(
                group
            )
        )

        result[str(period)] = (
            calculate_metrics(
                returns
            )
        )

    return result


# ============================================================
# COST STRESS TEST
# ============================================================

def cost_stress_test(
    dataframe: pd.DataFrame,
) -> dict:

    executed = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ].copy()

    if executed.empty:

        return {}

    results = {}

    for multiplier in COST_MULTIPLIERS:

        # cost is PER ROW — must be an array, not a Series scalar. The previous
        # float(cost) crashed on any multi-row input (introduced during a
        # refactor and caught by ChatGPT 2026-08-15). Zip all three per row.
        cost = (
            executed["cost_return"]
            .to_numpy(dtype=float)
            * multiplier
        )

        gross = (
            executed["actual_return"]
            .to_numpy(dtype=float)
        )

        dirs = (
            executed["decision"]
            .astype(str)
            .to_numpy()
        )

        # Canonical function per row so cost-stress uses the SAME sign/cost
        # convention as everything else — no inlined duplicate math.
        signed = np.array([
            directional_net_return(float(g), str(d), float(c))
            for g, d, c in zip(gross, dirs, cost)
        ], dtype=float)

        metrics = calculate_metrics(
            signed
        )

        results[
            str(multiplier)
        ] = metrics

    return results


# ============================================================
# PARAMETER STABILITY
# ============================================================

def parameter_stability(
    dataframe: pd.DataFrame,
) -> dict:

    """
    Checks whether the strongest predictions are actually
    stronger than weaker predictions.

    If expected edge increases but realized results do not
    improve, the model's confidence estimate is suspicious.
    """

    executed = dataframe[
        dataframe["decision"]
        != "NO_TRADE"
    ].copy()

    if len(executed) < 20:

        return {
            "status": "INSUFFICIENT_DATA"
        }

    executed["prediction_bucket"] = pd.qcut(

        executed[
            "net_expected_return"
        ],

        q=4,

        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ],

        duplicates="drop",
    )

    result = {}

    for bucket, group in (
        executed.groupby(
            "prediction_bucket",
            observed=True,
        )
    ):

        # net_actual_return is already direction-signed and cost-charged by
        # research.py. Do NOT re-sign shorts here (same bug class fixed on
        # 2026-08-15). Use it as-is.
        returns = group[
            "net_actual_return"
        ].to_numpy(dtype=float)

        result[str(bucket)] = {

            "trades":
                int(len(group)),

            "predicted_edge":
                float(
                    group[
                        "net_expected_return"
                    ].mean()
                ),

            "realized_edge":
                float(
                    returns.mean()
                ),
        }

    return result


# ============================================================
# OVERFITTING WARNING TESTS
# ============================================================

def overfitting_checks(
    dataframe: pd.DataFrame,
    metrics: dict,
    period_report: dict,
) -> list:

    warnings = []

    if metrics["trades"] < MIN_TRADES:

        warnings.append(
            "TOO_FEW_TRADES"
        )

    if (
        metrics.get(
            "max_drawdown",
            1
        )
        > MAX_ALLOWED_DRAWDOWN
    ):

        warnings.append(
            "EXCESSIVE_DRAWDOWN"
        )

    profitable_periods = 0

    total_periods = 0

    insufficient_periods = 0

    for values in period_report.values():

        # ENFORCE MIN_PERIOD_TRADES (ChatGPT #8): a 3-trade +20% period is NOT
        # evidence. Periods below the floor are INSUFFICIENT_DATA, not counted
        # as profitable/loss-making either way.
        if values.get("trades", 0) < MIN_PERIOD_TRADES:
            insufficient_periods += 1
            continue

        total_periods += 1

        if values.get(
            "mean_return",
            0
        ) > 0:

            profitable_periods += 1

    if (
        total_periods >= 3
        and
        profitable_periods
        < total_periods * 0.5
    ):

        warnings.append(
            "PERFORMANCE_NOT_STABLE_ACROSS_TIME"
        )

    if total_periods < 3:
        warnings.append(
            f"INSUFFICIENT_PERIOD_DATA "
            f"({insufficient_periods} periods below {MIN_PERIOD_TRADES} trades)"
        )

    return warnings


# ============================================================
# MODEL SCORE
# ============================================================

def calculate_model_score(
    metrics: dict,
    random_report: dict,
    regime_report: dict,
    period_report: dict,
    cost_report: dict,
) -> float:

    score = 0.0

    # --------------------------------------------------------
    # Profitability
    # --------------------------------------------------------

    score += (
        metrics.get(
            "mean_return",
            0
        )
        * 10_000
    )

    # --------------------------------------------------------
    # Sharpe-like
    # --------------------------------------------------------

    score += (
        metrics.get(
            "sharpe_like",
            0
        )
        * 2
    )

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    profit_factor = (
        metrics.get(
            "profit_factor",
            0
        )
    )

    if math.isfinite(
        profit_factor
    ):

        score += (
            profit_factor
            - 1
        )

    # --------------------------------------------------------
    # Drawdown penalty
    # --------------------------------------------------------

    score -= (
        metrics.get(
            "max_drawdown",
            1
        )
        * 10
    )

    # --------------------------------------------------------
    # Random baseline penalty
    # --------------------------------------------------------

    if (
        metrics.get(
            "mean_return",
            0
        )
        <=
        random_report.get(
            "mean_random_return",
            0
        )
    ):

        score -= 10

    # --------------------------------------------------------
    # Time stability
    # --------------------------------------------------------

    period_returns = []

    for values in period_report.values():

        if values.get("trades", 0) > 0:

            period_returns.append(
                values.get(
                    "mean_return",
                    0
                )
            )

    if period_returns:

        positive_ratio = (
            sum(
                x > 0
                for x in period_returns
            )
            / len(period_returns)
        )

        score += (
            positive_ratio
            * 5
        )

    # --------------------------------------------------------
    # Regime diversity
    # --------------------------------------------------------

    profitable_regimes = 0

    for values in regime_report.values():

        if (
            values.get("trades", 0)
            >= MIN_REGIME_TRADES
            and
            values.get(
                "mean_return",
                0
            ) > 0
        ):

            profitable_regimes += 1

    score += (
        profitable_regimes
        * 1.5
    )

    # --------------------------------------------------------
    # Cost survival
    # --------------------------------------------------------

    profitable_cost_tests = 0

    total_cost_tests = 0

    for values in cost_report.values():

        if values.get(
            "trades",
            0
        ) == 0:

            continue

        total_cost_tests += 1

        if values.get(
            "mean_return",
            0
        ) > 0:

            profitable_cost_tests += 1

    if total_cost_tests:

        cost_survival = (
            profitable_cost_tests
            / total_cost_tests
        )

        score += (
            cost_survival
            * 5
        )

    return float(score)


# ============================================================
# FINAL DECISION
# ============================================================

def final_decision(
    metrics: dict,
    random_report: dict,
    warnings: list,
    cost_report: dict,
    integrity: dict = None,
    event_stats: dict = None,
    shuffle: dict = None,
    long_metrics: dict = None,
    short_metrics: dict = None,
    stability_report: dict = None,
    cost_status: str = "UNCALIBRATED",
    selection_context: dict = None,
    freeze_mode: str = None,
) -> dict:
    """Return {"verdict":..., "reasons":[...]} — a structured decision.

    Gate order (ChatGPT's requested flow). This auditor's CEILING is
    PAPER_TEST by design (#4): PROMOTE requires real paper-trading evidence and
    belongs to a SEPARATE auditor, not another backtest. INCONCLUSIVE is a
    first-class verdict distinct from REJECT: 'not enough evidence' ≠ 'enough
    evidence and it failed'.
    """
    reasons = []

    def out(v, why):
        reasons.append(why)
        return {"verdict": v, "reasons": list(reasons)}

    # 0) COST CALIBRATION — an UNCALIBRATED cost model may not be promoted,
    #    however good it looks (ChatGPT #2/#14 earlier).
    if str(cost_status).upper() == "UNCALIBRATED":
        return out("BLOCKED_UNCALIBRATED_COST",
                   "cost model is UNCALIBRATED — no promotion path until real "
                   "instrument costs are configured")

    # 1) DATA INTEGRITY — reconstruct vs stored; mismatch => bad data (#3/#15).
    if integrity is not None and not integrity.get("ok", True):
        return out("REJECT",
                   f"DATA_INTEGRITY_FAILURE: {integrity.get('n_mismatch')} rows "
                   f"where stored net != reconstructed "
                   f"(max diff {integrity.get('max_abs_diff'):.2e})")

    # 2) SAMPLE SIZE in INDEPENDENT EVENTS — INCONCLUSIVE, not REJECT (#4/#6).
    ev = event_stats or {}
    # Clustering MUST be available to issue any positive verdict — a missing
    # clustering implementation must never let raw trades pose as independent
    # events (ChatGPT hostile #1). Block rather than fall back.
    if ev.get("clustering_available", True) is False:
        return out("BLOCKED_EVENT_CLUSTERING_UNAVAILABLE",
                   "independence definition (research.cluster_events) is "
                   "unavailable — refusing to treat raw trades as independent "
                   "events; cannot judge sample size")

    # Gate-facing count is the CONSERVATIVE effective count = min(cluster,
    # block floor) computed by the auditor itself (ChatGPT hostile #2).
    n_events = ev.get("effective_events",
                      ev.get("independent_event_estimate_v1",
                             metrics.get("trades", 0)))
    if n_events < MIN_EVENTS_INCONCLUSIVE:
        return out("INCONCLUSIVE",
                   f"only {n_events} effective independent events "
                   f"(< {MIN_EVENTS_INCONCLUSIVE}) — insufficient evidence to "
                   f"judge; not a failure")

    # secondary raw-trade floor (reference)
    if metrics.get("trades", 0) < MIN_TRADES:
        return out("INCONCLUSIVE",
                   f"only {metrics.get('trades',0)} raw trades (< {MIN_TRADES})")

    # 3) BEAT RANDOM BY A MARGIN, and SIGNIFICANTLY (#5/#8).
    edge_over_random = (metrics.get("mean_return", 0.0)
                        - random_report.get("mean_random_return", 0.0))
    if edge_over_random < MIN_EDGE_OVER_RANDOM:
        return out("REJECT",
                   f"edge over random {edge_over_random:.2e} "
                   f"< required {MIN_EDGE_OVER_RANDOM:.2e}")

    p_random = random_report.get("probability_random_beats_model",
                                 random_report.get("probability_beating_model",
                                                   1.0))
    if p_random > MAX_PROB_RANDOM_BEATS_MODEL:
        return out("REJECT",
                   f"random beats model with p={p_random:.3f} "
                   f"(> {MAX_PROB_RANDOM_BEATS_MODEL}) — not significant")

    # 3b) SHUFFLE / PERMUTATION significance (ChatGPT hostile #3): BOTH the IID
    #     null AND the block null (which preserves temporal dependence) must
    #     reject. The block null is the harder, more appropriate time-series
    #     test — passing only IID is not enough.
    if shuffle is not None and shuffle.get("iterations", 0) > 0:
        p_iid = shuffle.get("prob_iid_ge_real",
                            shuffle.get("prob_shuffled_ge_real", 1.0))
        p_block = shuffle.get("prob_block_ge_real", 1.0)
        if p_iid > MAX_PROB_RANDOM_BEATS_MODEL:
            return out("REJECT",
                       f"IID shuffle p={p_iid:.3f}: apparent edge consistent "
                       f"with unconditional structure")
        if p_block > MAX_PROB_RANDOM_BEATS_MODEL:
            return out("REJECT",
                       f"BLOCK shuffle p={p_block:.3f}: edge does not survive a "
                       f"time-series-aware null (momentum/vol-clustering "
                       f"preserved) — likely exploiting temporal structure, "
                       f"not conditional skill")

    # 4) EQUITY DRAWDOWN (#7 — now true peak-to-trough).
    if metrics.get("max_drawdown", 1.0) > MAX_ALLOWED_DRAWDOWN:
        return out("REJECT",
                   f"equity drawdown {metrics.get('max_drawdown'):.1%} "
                   f"> {MAX_ALLOWED_DRAWDOWN:.0%}")

    # 5) COST RESILIENCE.
    profitable = total = 0
    for values in cost_report.values():
        if values.get("trades", 0) == 0:
            continue
        total += 1
        if values.get("mean_return", 0) > 0:
            profitable += 1
    if total and (profitable / total) < MIN_COST_SURVIVAL:
        return out("REJECT",
                   f"survives only {profitable}/{total} cost scenarios "
                   f"(< {MIN_COST_SURVIVAL:.0%})")

    # 6) JUSTIFY vs SIMPLE CONTROLS (#11): if the model doesn't beat simply
    #    being long or short on its own trades, it must at least be justified
    #    (e.g. lower drawdown). Otherwise INVESTIGATE, not PAPER_TEST.
    m_mean = metrics.get("mean_return", 0.0)
    if long_metrics and m_mean <= long_metrics.get("mean_return", -1) \
       and metrics.get("max_drawdown", 1) >= long_metrics.get("max_drawdown", 1):
        return out("INVESTIGATE",
                   "model does not beat always-long on its own trades and is "
                   "not lower-drawdown — no clear directional edge")
    if short_metrics and m_mean <= short_metrics.get("mean_return", -1) \
       and metrics.get("max_drawdown", 1) >= short_metrics.get("max_drawdown", 1):
        return out("INVESTIGATE",
                   "model does not beat always-short on its own trades and is "
                   "not lower-drawdown")

    # 7) CONFIDENCE MONOTONICITY (#10): higher-confidence buckets should not
    #    realize worse outcomes.
    if stability_report is not None and confidence_is_inverted(stability_report):
        return out("INVESTIGATE",
                   "CONFIDENCE_MONOTONICITY_FAILURE: higher-confidence buckets "
                   "realized worse outcomes")

    # 8) GENERAL WARNINGS.
    if warnings:
        return out("INVESTIGATE", "; ".join(warnings))

    # 8b) ECONOMIC-HURDLE CI GATE (ChatGPT hostile #6). A statistically
    #     detectable but economically trivial edge is not tradeable after
    #     spread uncertainty, slippage, fees, and model degradation. Require
    #     the LOWER bound of the event-level 95% CI to clear a real hurdle,
    #     not merely be positive.
    ci = ev.get("event_mean_ci", [float("nan"), float("nan")])
    ci_low = ci[0] if ci and len(ci) == 2 else float("nan")
    if not (ci_low == ci_low):   # NaN check
        return out("INVESTIGATE",
                   "event-level CI unavailable — cannot confirm the edge clears "
                   "an economic hurdle")
    if ci_low <= MIN_ECONOMIC_HURDLE:
        return out("INVESTIGATE",
                   f"95% CI lower bound {ci_low:.2e} does not clear the economic "
                   f"hurdle {MIN_ECONOMIC_HURDLE:.2e} — edge may be real but not "
                   f"economically meaningful")

    # 8c) MULTIPLE-TESTING / SELECTION provenance (ChatGPT hostile #4). If this
    #     verdict is the winner among many candidates (horizons × variants ×
    #     params), a bare p<0.05 is selection bias. Require either a small
    #     candidate count or a Bonferroni-adjusted significance.
    sel = selection_context or {}
    n_candidates = int(sel.get("candidate_count", 1) or 1)
    if n_candidates > 1:
        p_iid = (shuffle or {}).get("prob_iid_ge_real", 1.0)
        adjusted = p_iid * n_candidates          # Bonferroni
        if adjusted > MAX_PROB_RANDOM_BEATS_MODEL:
            return out("INVESTIGATE",
                       f"selection-aware: this candidate is 1 of {n_candidates}; "
                       f"Bonferroni-adjusted p={adjusted:.3f} "
                       f"(> {MAX_PROB_RANDOM_BEATS_MODEL}) — likely a "
                       f"selection-bias winner, not a robust edge")

    # 9) STRONGEST VERDICT this auditor can issue = PAPER_TEST — and ONLY from
    #    a genuine FROZEN_HOLDOUT audit. Without a registered frozen holdout the
    #    evidence is potentially contaminated by repeated tuning, so the ceiling
    #    is INVESTIGATE (frozen-OOS contract, ruling #4).
    if freeze_mode != "FROZEN_HOLDOUT":
        return out("INVESTIGATE",
                   "passes all statistical gates BUT this is not a frozen-"
                   "holdout audit (no immutable OOS cutoff) — diagnostic only, "
                   "cannot authorize PAPER_TEST")

    if n_events < MIN_EVENTS_MEANINGFUL:
        return out("INVESTIGATE",
                   f"passes all gates but only {n_events} effective events "
                   f"(< {MIN_EVENTS_MEANINGFUL}) — promising, not yet strong")

    return out("PAPER_TEST",
               "passed integrity, clustering-available, conservative event "
               "count, IID+block significance, cost resilience, economic "
               "hurdle, selection-awareness, control, stability, AND frozen-"
               "holdout gates")


# ============================================================
# EVENT-LEVEL STATS, SIGNIFICANCE, NEW BASELINES  (batch 3)
# ============================================================
#
# The auditor computes these from its OWN independently-reconstructed signed
# returns (never the learner's stored net), but reuses research.py's clustering
# + bootstrap so "an event" and "a CI" mean the SAME thing in both files. This
# is the hybrid: independent COMPUTATION, shared DEFINITION.
# ============================================================

def _import_research():
    """Import research.py's canonical event/bootstrap functions. Kept as a
    soft import so a missing research.py degrades to raw-trade stats with a
    warning rather than crashing the auditor."""
    try:
        import research as _r
        return _r
    except Exception:
        return None


def reconstruct_signed_returns(dataframe: pd.DataFrame) -> np.ndarray:
    """Auditor's OWN signed returns from raw+decision+cost (not the stored
    net). This is what makes the audit independent of the learner's math."""
    executed = dataframe[dataframe["decision"] != "NO_TRADE"]
    if executed.empty:
        return np.array([], dtype=float)
    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)
    dirs = executed["decision"].astype(str).to_numpy()
    return np.array([
        directional_net_return(float(r), str(d), float(c))
        for r, d, c in zip(raw, dirs, cost)
    ], dtype=float)


def conservative_block_events(dataframe: pd.DataFrame,
                              bar_seconds: float,
                              horizon: int) -> int:
    """The auditor's OWN, independent lower-bound on independent events — it
    does NOT delegate independence to research.py (ChatGPT hostile #2).

    Method: partition the executed timeline into NON-OVERLAPPING time blocks of
    length = horizon bars. Every trade whose timestamp falls in the same block
    collapses to ONE event, regardless of direction. This is deliberately more
    aggressive than research.cluster_events (which only merges same-direction
    overlapping runs), so it is a conservative floor: two adjacent opposite-
    direction trades within one horizon window count as ONE event here.

    A hostile constructor who splits economically-dependent observations into
    many 'events' under the overlap rule cannot inflate THIS count, because it
    keys purely on wall-clock block occupancy.
    """
    executed = dataframe[dataframe["decision"] != "NO_TRADE"].copy()
    if executed.empty:
        return 0
    ts = pd.to_datetime(executed["timestamp"], utc=True)
    # seconds since the first executed trade — unambiguous regardless of the
    # underlying epoch unit (a previous version guessed nanoseconds and broke
    # when the index was microseconds, collapsing everything into 1 block).
    t0 = ts.min()
    secs = (ts - t0).dt.total_seconds().to_numpy()
    block_width = float(max(1, horizon) * max(bar_seconds, 1.0))
    blocks = np.floor(secs / block_width).astype("int64")
    return int(len(np.unique(blocks)))


def event_level_stats(dataframe: pd.DataFrame,
                      bar_seconds: float = 300.0) -> dict:
    """Independent-event count + bootstrap CIs, computed by the auditor.

    Reports THREE counts (ChatGPT hostile #2):
      raw_observations              — executed bars (autocorrelated)
      cluster_events                — research.py overlap rule (primary)
      conservative_block_events     — auditor's own non-overlapping block floor
    The gate-facing number is the CONSERVATIVE MINIMUM:
      effective_events = min(cluster_events, conservative_block_events)

    If clustering is UNAVAILABLE (research.py not importable), we DO NOT fall
    back to raw trades — that would let 500 correlated bars pose as 500
    independent events. We return clustering_available=False and 0 events, and
    the decision layer turns that into a BLOCK (never PAPER_TEST).
    """
    r = _import_research()
    executed = dataframe[dataframe["decision"] != "NO_TRADE"].copy()
    n_raw = int(len(executed))
    horizon = int(executed["horizon"].iloc[0]) if not executed.empty else 1

    # Auditor's own conservative floor — computed regardless of research.py.
    n_block = conservative_block_events(dataframe, bar_seconds, horizon)

    if r is None or executed.empty:
        # UNSAFE to proceed — clustering is the independence definition and it
        # is missing. Do NOT invent independence from raw trades.
        signed = reconstruct_signed_returns(dataframe)
        n = len(signed)
        return {
            "clustering_available": False,
            "raw_observations": n_raw,
            "cluster_events": None,
            "conservative_block_events": int(n_block),
            "independent_event_estimate_v1": 0,   # gate-facing: 0 => BLOCK
            "effective_events": 0,
            "event_mean": float(signed.mean()) if n else 0.0,
            "event_mean_ci": [float("nan"), float("nan")],
            "event_positive_rate": float((signed > 0).mean()) if n else 0.0,
            "event_positive_rate_ci": [float("nan"), float("nan")],
            "clustering": "UNAVAILABLE_blocked",
        }

    class _P:
        __slots__ = ("decision", "timestamp", "horizon",
                     "net_actual_return", "regime")

    recon = reconstruct_signed_returns(dataframe)
    preds = []
    for (_, row), sr in zip(executed.iterrows(), recon):
        p = _P()
        p.decision = str(row["decision"])
        p.timestamp = str(row["timestamp"])
        p.horizon = int(row["horizon"])
        p.net_actual_return = float(sr)     # OUR reconstruction
        p.regime = str(row.get("regime", ""))
        preds.append(p)

    event_returns, _ = r.cluster_events(preds, bar_seconds)
    n_ev = len(event_returns)

    # Gate-facing count is the conservative minimum across dependence
    # assumptions (ChatGPT hostile #2). CIs are computed on the clustered
    # series but reported alongside the conservative count so the reader sees
    # both.
    effective = int(min(n_ev, n_block)) if n_block > 0 else int(n_ev)

    mean_pt, mean_lo, mean_hi, _ = r.block_bootstrap_ci(
        event_returns, statistic="mean")
    pos_pt, pos_lo, pos_hi, _ = r.block_bootstrap_ci(
        event_returns, statistic="positive_rate")

    return {
        "clustering_available": True,
        "raw_observations": n_raw,
        "cluster_events": int(n_ev),
        "conservative_block_events": int(n_block),
        # gate-facing effective count = conservative minimum
        "independent_event_estimate_v1": effective,
        "effective_events": effective,
        "event_mean": float(mean_pt),
        "event_mean_ci": [float(mean_lo), float(mean_hi)],
        "event_positive_rate": float(pos_pt),
        "event_positive_rate_ci": [float(pos_lo), float(pos_hi)],
        "clustering": "min(research.cluster_events_v1, block_floor)",
    }


def equity_drawdown(returns: np.ndarray) -> float:
    """TRUE peak-to-trough EQUITY drawdown via compounding (ChatGPT #7), not
    the additive cumsum version. 0.25 => 25% peak-to-trough."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(-dd.min())


def momentum_baseline(dataframe: pd.DataFrame) -> np.ndarray:
    """Dumb control (ChatGPT): go LONG when recent momentum is up, SHORT when
    down, on the SAME executed opportunities, using raw+cost. If KNN can't beat
    'trade with the recent move', KNN hasn't demonstrated much.

    Momentum proxy = sign of expected_return is NOT allowed (that's the model).
    We use the realized SHORT-TERM prior return if present; else fall back to
    the sign of the row's own trend feature is unavailable here, so we use a
    causal proxy: the sign of the previous executed bar's actual_return."""
    executed = dataframe[dataframe["decision"] != "NO_TRADE"].copy()
    if executed.empty:
        return np.array([], dtype=float)
    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)
    # causal momentum: previous bar's raw return sign (shifted, no look-ahead).
    prev = np.roll(raw, 1)
    prev[0] = 0.0
    dirs = np.where(prev >= 0, "LONG", "SHORT")
    return np.array([
        directional_net_return(float(r), str(d), float(c))
        for r, d, c in zip(raw, dirs, cost)
    ], dtype=float)


def shuffle_test(dataframe: pd.DataFrame,
                 iterations: int = SHUFFLE_ITERATIONS,
                 seed: int = 20260815) -> dict:
    """Permutation tests (ChatGPT hostile #3). Two nulls, BOTH must be passed:

    1. IID permutation — permute individual outcomes against fixed directions.
       Null: 'are the model's directions associated with THESE outcomes?'
       Weakness: destroys temporal structure (momentum/vol-clustering), so it
       tests against an exchangeable/IID assumption.

    2. BLOCK permutation — permute CONTIGUOUS BLOCKS of outcomes, preserving
       local dependence (momentum, volatility clustering, autocorrelation).
       Null: 'does the model still look special when realistic time-series
       structure is retained?' This is the harder, more appropriate null.

    Returns both p-values. The decision layer requires BOTH < threshold.
    """
    executed = dataframe[dataframe["decision"] != "NO_TRADE"].copy()
    if len(executed) < 10:
        return {"real_mean": 0.0,
                "prob_iid_ge_real": 1.0,
                "prob_block_ge_real": 1.0,
                "iterations": 0}

    raw = executed["actual_return"].to_numpy(dtype=float)
    cost = executed["cost_return"].to_numpy(dtype=float)
    sign = np.where(executed["decision"].astype(str).to_numpy() == "LONG",
                    1.0, -1.0)

    real = float((sign * raw - cost).mean())
    rng = np.random.default_rng(seed)
    n = len(raw)

    # 1) IID permutation
    iid = np.empty(iterations, dtype=float)
    for i in range(iterations):
        iid[i] = float((sign * raw[rng.permutation(n)] - cost).mean())
    p_iid = float((iid >= real).mean())

    # 2) Block permutation — preserve local dependence. Block ~ sqrt(n),
    #    clamped to [5, n]. Resample contiguous blocks of OUTCOMES, keep the
    #    directions fixed in place.
    block = int(max(5, min(n, round(math.sqrt(n)))))
    n_blocks = int(math.ceil(n / block))
    blk = np.empty(iterations, dtype=float)
    starts_max = n - block
    for i in range(iterations):
        if starts_max <= 0:
            perm = rng.permutation(n)
        else:
            starts = rng.integers(0, starts_max + 1, size=n_blocks)
            perm = np.concatenate([np.arange(s, s + block)
                                   for s in starts])[:n]
        blk[i] = float((sign * raw[perm] - cost).mean())
    p_block = float((blk >= real).mean())

    return {
        "real_mean": real,
        "shuffled_mean_iid": float(iid.mean()),
        "prob_iid_ge_real": p_iid,
        "prob_block_ge_real": p_block,
        "block_size": block,
        "iterations": int(iterations),
        # back-compat alias (old field name = the IID p-value)
        "prob_shuffled_ge_real": p_iid,
    }


def confidence_is_inverted(stability_report: dict) -> bool:
    """Detect a monotonicity failure (ChatGPT #10): the model's HIGHER-
    confidence buckets realizing WORSE outcomes than its lower-confidence ones.
    Returns True if the realized edge trends DOWN as predicted edge goes UP."""
    try:
        buckets = []
        for k, v in stability_report.items():
            if isinstance(v, dict) and v.get("trades", 0) >= MIN_PERIOD_TRADES:
                pe = v.get("predicted_edge")
                re = v.get("mean_return", v.get("realized_edge"))
                if pe is not None and re is not None:
                    buckets.append((float(pe), float(re)))
        if len(buckets) < 3:
            return False
        buckets.sort(key=lambda x: x[0])          # by predicted edge ascending
        realized = [b[1] for b in buckets]
        # inverted if the top-predicted bucket realizes less than the bottom,
        # AND the sequence is net-decreasing.
        decreasing = sum(1 for a, b in zip(realized, realized[1:]) if b < a)
        return (realized[-1] < realized[0]
                and decreasing > len(realized) // 2)
    except Exception:
        return False


# ============================================================
# FULL AUDIT
# ============================================================

def audit_model(
    dataframe: pd.DataFrame,
    cost_status: str = "UNCALIBRATED",
    bar_seconds: float = 300.0,
    apply_oos: bool = True,
    selection_context: dict = None,
    freeze: dict = None,
) -> dict:
    """Audit under the FROZEN-OOS CONTRACT.

    freeze (dict) enables the real contract:
        {"model_freeze_id":..., "oos_cutoff_timestamp":..., "spec":{...},
         "database_file":...}
    When present, the auditor:
      • verifies the freeze manifest (spec hash + cutoff immutable),
      • splits on the CALENDAR cutoff into development vs frozen holdout,
      • computes a DIAGNOSTIC walk-forward summary on development (cannot gate),
      • gates PAPER_TEST ONLY on the frozen holdout,
      • consumes the freeze's single holdout audit (one per freeze).

    When freeze is None the auditor runs in DIAGNOSTIC-ONLY mode: it still
    reports everything but its ceiling is INVESTIGATE — it may NOT issue
    PAPER_TEST, because without a registered frozen holdout there is no
    uncontaminated evidence to authorize paper testing.
    """
    full_n = len(dataframe)

    # ── HOSTILE INPUT SANITATION (ChatGPT HIGH) ─────────────────────────────
    # A hostile auditor must not trust that prediction rows are clean. Even
    # though research.py now skips NaN targets at write time, the auditor
    # independently rejects any row it cannot honestly score: non-finite
    # actual_return / cost_return, invalid decision label, or unparseable
    # timestamp. Rejected rows never enter any statistic. This is defense in
    # depth — the producer fix and this consumer fix are independent.
    if dataframe is not None and len(dataframe):
        dataframe = dataframe.copy()
        _valid = pd.Series(True, index=dataframe.index)
        for _col in ("actual_return", "cost_return"):
            if _col in dataframe.columns:
                _num = pd.to_numeric(dataframe[_col], errors="coerce")
                _valid &= np.isfinite(_num.to_numpy(dtype=float))
        if "decision" in dataframe.columns:
            _valid &= dataframe["decision"].isin(
                ["LONG", "SHORT", "NO_TRADE"]).to_numpy()
        if "timestamp" in dataframe.columns:
            _ts = pd.to_datetime(dataframe["timestamp"], errors="coerce", utc=True)
            _valid &= _ts.notna().to_numpy()
        _rejected = int((~_valid).sum())
        if _rejected:
            print(f"  [auditor] rejected {_rejected} invalid evidence row(s) "
                  f"(non-finite/NaN/malformed) before scoring")
            dataframe = dataframe[_valid].reset_index(drop=True)

    freeze_info = {"mode": "DIAGNOSTIC_ONLY", "reason":
                   "no freeze supplied — cannot issue PAPER_TEST"}
    wf_diagnostic = None
    verdict_override = None

    if freeze:
        db = freeze.get("database_file")
        fid = freeze.get("model_freeze_id")
        cutoff = freeze.get("oos_cutoff_timestamp")
        spec = freeze.get("spec") or {}
        try:
            v = verify_freeze(db, fid, spec)
        except InvalidFreeze as e:
            # Hard stop — a bad/altered freeze is not a model failure.
            return {
                "audit_timestamp": datetime.now(timezone.utc).isoformat(),
                "auditor_ceiling": "PAPER_TEST",
                "decision": {"verdict": "INVALID_FREEZE",
                             "reasons": [str(e)]},
                "freeze": {"mode": "INVALID", "model_freeze_id": fid,
                           "error": str(e)},
            }

        dev, holdout = split_frozen_oos(dataframe, cutoff)

        # DIAGNOSTIC walk-forward on development (causal history). This can
        # NEVER authorize PAPER_TEST — it is the data the model was tuned on.
        if not dev.empty:
            wf_returns = reconstruct_signed_returns(dev)
            wf_diagnostic = calculate_metrics(wf_returns)
            wf_diagnostic["events"] = event_level_stats(
                dev, bar_seconds).get("effective_events")

        freeze_info = {
            "mode": "FROZEN_HOLDOUT",
            "model_freeze_id": fid,
            "spec_hash": v["spec_hash"],
            "oos_cutoff_timestamp": cutoff,
            "development_rows": int(len(dev)),
            "holdout_rows": int(len(holdout)),
        }

        # The audit now runs on the HOLDOUT ONLY.
        dataframe = holdout

        # Too few holdout events -> INCONCLUSIVE (NEVER move the cutoff).
        if len(dataframe) == 0:
            verdict_override = {
                "verdict": "INCONCLUSIVE",
                "reasons": [f"frozen holdout after {cutoff} is EMPTY — "
                            f"insufficient post-cutoff evidence; do NOT move "
                            f"the cutoff, gather more data"]}
    elif apply_oos:
        # No freeze: diagnostic-only. Audit the whole frame but cap at
        # INVESTIGATE (enforced in final_decision via freeze_mode).
        pass

    oos_n = len(dataframe)

    # ── DATA INTEGRITY FIRST (ChatGPT #3/#15) ────────────────────────────
    integrity = verify_data_integrity(dataframe)

    # Model returns: reconstructed independently (auditor never trusts stored
    # net for its own stats).
    model_returns = reconstruct_signed_returns(dataframe)

    metrics = calculate_metrics(
        model_returns
    )

    long_metrics = calculate_metrics(
        always_long_baseline(
            dataframe
        )
    )

    short_metrics = calculate_metrics(
        always_short_baseline(
            dataframe
        )
    )

    momentum_metrics = calculate_metrics(
        momentum_baseline(dataframe)
    )

    random_report = random_baseline(
        dataframe
    )

    event_stats = event_level_stats(dataframe, bar_seconds)

    shuffle_report = shuffle_test(dataframe)

    no_trade_report = (
        analyze_no_trade_filter(
            dataframe
        )
    )

    regime_report = (
        analyze_regimes(
            dataframe
        )
    )

    period_report = (
        analyze_periods(
            dataframe
        )
    )

    cost_report = (
        cost_stress_test(
            dataframe
        )
    )

    stability_report = (
        parameter_stability(
            dataframe
        )
    )

    warnings = overfitting_checks(
        dataframe,
        metrics,
        period_report,
    )

    score = calculate_model_score(
        metrics,
        random_report,
        regime_report,
        period_report,
        cost_report,
    )

    decision = final_decision(
        metrics,
        random_report,
        warnings,
        cost_report,
        integrity=integrity,
        event_stats=event_stats,
        shuffle=shuffle_report,
        long_metrics=long_metrics,
        short_metrics=short_metrics,
        stability_report=stability_report,
        cost_status=cost_status,
        selection_context=selection_context,
        freeze_mode=freeze_info.get("mode"),
    )

    # Empty-holdout override (INCONCLUSIVE, never move the cutoff).
    if verdict_override is not None:
        decision = verdict_override

    # ── CONSUME THE SINGLE HOLDOUT AUDIT (ruling #5) ─────────────────────
    # Only a real frozen-holdout audit that produced a DECISION-QUALITY verdict
    # spends the freeze. Diagnostic-only and INVALID_FREEZE do not. If it was
    # already consumed, surface that as the verdict rather than silently
    # re-auditing.
    if freeze and freeze_info.get("mode") == "FROZEN_HOLDOUT" \
       and verdict_override is None:
        result_hash = canonical_spec_hash({
            "verdict": decision.get("verdict"),
            "event_mean": event_stats.get("event_mean"),
            "effective_events": event_stats.get("effective_events"),
        })
        try:
            consume_holdout_audit(freeze.get("database_file"),
                                  freeze.get("model_freeze_id"),
                                  result_hash)
        except FrozenHoldoutAlreadyConsumed as e:
            decision = {"verdict": "HOLDOUT_ALREADY_CONSUMED",
                        "reasons": [str(e)]}

    return {

        "audit_timestamp":
            datetime.now(timezone.utc).isoformat(),

        "auditor_ceiling":
            ("PAPER_TEST" if freeze_info.get("mode") == "FROZEN_HOLDOUT"
             else "INVESTIGATE"),   # diagnostic-only cannot reach PAPER_TEST

        "freeze":
            freeze_info,

        "walk_forward_diagnostic":
            wf_diagnostic,   # development-set causal summary; NEVER gates

        "oos_policy":
            OOS_POLICY,
        "sample_full_rows":
            int(full_n),
        "sample_oos_rows":
            int(oos_n),

        "cost_model_status":
            cost_status,

        "data_integrity":
            integrity,

        "event_stats":
            event_stats,

        "shuffle_test":
            shuffle_report,

        "momentum_baseline":
            momentum_metrics,

        "model_version":
            str(
                dataframe[
                    "model_version"
                ].iloc[0]
            ),

        "instrument":
            str(
                dataframe[
                    "instrument"
                ].iloc[0]
            ),

        "horizon":
            int(
                dataframe[
                    "horizon"
                ].iloc[0]
            ),

        "model_metrics":
            metrics,

        "always_long":
            long_metrics,

        "always_short":
            short_metrics,

        "random_baseline":
            random_report,

        "no_trade_analysis":
            no_trade_report,

        "regime_analysis":
            regime_report,

        "period_analysis":
            period_report,

        "cost_stress_test":
            cost_report,

        "confidence_stability":
            stability_report,

        "warnings":
            warnings,

        "model_score":
            score,

        "decision":
            decision,
    }


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    report: dict,
):

    print()
    print("=" * 72)

    print(
        "MODEL AUDIT REPORT"
    )

    print("=" * 72)

    print()

    print(
        f"Model       : "
        f"{report['model_version']}"
    )

    print(
        f"Instrument  : "
        f"{report['instrument']}"
    )

    print(
        f"Horizon     : "
        f"{report['horizon']}"
    )

    print()

    metrics = (
        report["model_metrics"]
    )

    print(
        "MODEL PERFORMANCE"
    )

    print(
        "-----------------"
    )

    for key, value in metrics.items():

        print(
            f"{key:25} {value}"
        )

    print()

    print(
        "BASELINES"
    )

    print(
        "---------"
    )

    print(
        "Always LONG:"
    )

    print(
        json.dumps(
            report["always_long"],
            indent=4,
        )
    )

    print()

    print(
        "Always SHORT:"
    )

    print(
        json.dumps(
            report["always_short"],
            indent=4,
        )
    )

    print()

    print(
        "Random:"
    )

    print(
        json.dumps(
            report["random_baseline"],
            indent=4,
        )
    )

    print()

    print(
        "NO-TRADE FILTER"
    )

    print(
        "---------------"
    )

    print(
        json.dumps(
            report["no_trade_analysis"],
            indent=4,
        )
    )

    print()

    print(
        "REGIMES"
    )

    print(
        "-------"
    )

    for regime, values in (
        report[
            "regime_analysis"
        ].items()
    ):

        print()
        print(regime)

        for key, value in (
            values.items()
        ):

            print(
                f"    {key:20} "
                f"{value}"
            )

    print()

    print(
        "COST STRESS TEST"
    )

    print(
        "----------------"
    )

    for multiplier, values in (
        report[
            "cost_stress_test"
        ].items()
    ):

        print(
            f"\nCost x {multiplier}"
        )

        print(
            f"    mean return: "
            f"{values.get('mean_return')}"
        )

        print(
            f"    trades: "
            f"{values.get('trades')}"
        )

    print()

    print(
        "CONFIDENCE STABILITY"
    )

    print(
        "--------------------"
    )

    print(
        json.dumps(
            report[
                "confidence_stability"
            ],
            indent=4,
        )
    )

    print()

    print(
        "WARNINGS"
    )

    print(
        "--------"
    )

    if report["warnings"]:

        for warning in (
            report["warnings"]
        ):

            print(
                f"!!! {warning}"
            )

    else:

        print(
            "No major automatic warnings."
        )

    print()

    print(
        f"MODEL SCORE: "
        f"{report['model_score']:.4f}"
    )

    print()

    print(
        "FINAL DECISION"
    )

    print(
        "--------------"
    )

    dec = report.get("decision", {})
    if isinstance(dec, dict):
        print(f"   VERDICT: {dec.get('verdict')}")
        print(f"   (auditor ceiling: {report.get('auditor_ceiling','PAPER_TEST')})")
        for why in dec.get("reasons", []):
            print(f"     - {why}")
    else:
        print(f"   {dec}")

    ev = report.get("event_stats", {})
    sh = report.get("shuffle_test", {})
    print()
    print(f"   effective events   : {ev.get('effective_events')}  "
          f"(cluster {ev.get('cluster_events')}, "
          f"block {ev.get('conservative_block_events')}, "
          f"raw {ev.get('raw_observations')})")
    print(f"   clustering available: {ev.get('clustering_available')}")
    print(f"   event mean + 95% CI: {ev.get('event_mean')}  "
          f"{ev.get('event_mean_ci')}")
    print(f"   shuffle p (IID/block): {sh.get('prob_iid_ge_real')} / "
          f"{sh.get('prob_block_ge_real')}")
    print(f"   cost model status  : {report.get('cost_model_status')}")
    print(f"   OOS rows / full    : {report.get('sample_oos_rows')} / "
          f"{report.get('sample_full_rows')}  ({report.get('oos_policy')})")

    print()

    print("=" * 72)


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    report: dict,
    filename: str = "latest_model_audit.json",
):

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()
    print(
        f"Audit saved to: {filename}"
    )


# ============================================================
# MAIN
# ============================================================

def _cost_status_and_bar_seconds(database_file: str):
    """Read the most recent experiment's cost_model_status and timeframe from
    the experiments table. GLOBAL fallback only — prefer the per-instrument
    version below so GOLD's CALIBRATED status is never applied to QTUM's
    UNCALIBRATED evidence (ChatGPT isolation finding)."""
    status, bar_seconds = "UNCALIBRATED", 300.0
    try:
        conn = sqlite3.connect(database_file)
        row = conn.execute(
            "SELECT report_json, timeframe FROM experiments "
            "ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            rep = json.loads(row[0])
            status = rep.get("cost_model_status", status)
            bar_seconds = _tf_to_seconds(row[1])
    except Exception:
        pass
    return status, bar_seconds


def _tf_to_seconds(tf) -> float:
    try:
        tf = str(tf or "5m").lower()
        num = int("".join(c for c in tf if c.isdigit()) or "5")
        unit = "".join(c for c in tf if c.isalpha())
        mult = {"m": 60, "min": 60, "h": 3600, "hour": 3600,
                "d": 86400}.get(unit, 60)
        return float(num * mult) or 300.0
    except Exception:
        return 300.0


def cost_status_and_bar_seconds_per_instrument(database_file: str) -> dict:
    """{instrument: (cost_status, bar_seconds)} from EACH instrument's most
    recent experiment. Isolates cost status per instrument so one instrument's
    calibration can never leak onto another's audit (ChatGPT CRITICAL/HIGH:
    global cost status was applied to all instruments). Instruments absent here
    fall back to the global reader (UNCALIBRATED-safe)."""
    out = {}
    try:
        conn = sqlite3.connect(database_file)
        # newest experiment row per instrument
        rows = conn.execute(
            "SELECT e.instrument, e.report_json, e.timeframe "
            "FROM experiments e "
            "JOIN (SELECT instrument, MAX(id) AS mid FROM experiments "
            "      GROUP BY instrument) m "
            "ON e.instrument = m.instrument AND e.id = m.mid"
        ).fetchall()
        conn.close()
        for inst, report_json, tf in rows:
            try:
                rep = json.loads(report_json)
                status = rep.get("cost_model_status", "UNCALIBRATED")
            except Exception:
                status = "UNCALIBRATED"
            out[inst] = (status, _tf_to_seconds(tf))
    except Exception:
        pass
    return out


def main():

    print(
        "Loading predictions..."
    )

    # Audit the latest run PER INSTRUMENT — not just the single global latest
    # run, which would only cover the last instrument the supervisor researched
    # (ChatGPT CRITICAL finding). We load each instrument's most-recent run and
    # concatenate, so all instruments are audited every day. Each still gets its
    # own isolated (model,instrument,horizon) groups below; no blending across
    # instruments occurs.
    per_inst = latest_run_id_per_instrument(DATABASE_FILE)
    if per_inst:
        frames = []
        for _inst, _rid in per_inst.items():
            f = load_predictions(DATABASE_FILE, research_run_id=_rid)
            if f is not None and len(f):
                frames.append(f)
        if frames:
            import pandas as _pd
            dataframe = _pd.concat(frames, ignore_index=True)
        else:
            dataframe = load_predictions(DATABASE_FILE,
                                         research_run_id=latest_run_id(DATABASE_FILE))
        run_id = f"per_instrument_latest({len(per_inst)})"
    else:
        # fallback: single latest run (old behavior) if per-instrument query
        # returns nothing (e.g. empty DB)
        run_id = latest_run_id(DATABASE_FILE)
        dataframe = load_predictions(
            DATABASE_FILE,
            research_run_id=run_id,
        )

    # Per-instrument cost status/bar-seconds — each instrument judged under ITS
    # OWN cost calibration, never a global one (ChatGPT isolation fix). Global
    # reader kept as the fallback for any instrument missing an experiment row.
    _cost_by_inst = cost_status_and_bar_seconds_per_instrument(DATABASE_FILE)
    cost_status, bar_seconds = _cost_status_and_bar_seconds(DATABASE_FILE)

    print(
        f"Loaded "
        f"{len(dataframe):,} predictions."
        f"  (run_id={run_id}, per-instrument cost isolation on)"
    )

    # --------------------------------------------------------
    # Audit each model/horizon independently
    # --------------------------------------------------------

    grouping = [
        "model_version",
        "instrument",
        "horizon",
    ]

    reports = []

    groups = list(dataframe.groupby(grouping))
    # Multiple-testing surface: every (model,instrument,horizon) group audited
    # in this run is a candidate. A verdict that only wins because it was the
    # best of many must be discounted (ChatGPT hostile #4).
    candidate_count = len(groups)

    for keys, group in groups:

        print()
        print(
            "Auditing:"
        )

        print(
            f"Model={keys[0]} "
            f"Instrument={keys[1]} "
            f"Horizon={keys[2]}"
        )

        # This group's instrument gets ITS OWN cost status + bar size.
        _inst_key = keys[1]
        _cs, _bs = _cost_by_inst.get(_inst_key, (cost_status, bar_seconds))

        report = audit_model(
            group.copy(),
            cost_status=_cs,
            bar_seconds=_bs,
            selection_context={"candidate_count": candidate_count},
        )

        reports.append(
            report
        )

        print_report(
            report
        )

    # --------------------------------------------------------
    # MULTI-HORIZON / MULTIPLE-TESTING AWARENESS (ChatGPT — don't treat N
    # horizons as N independent chances to find an edge). Flag when a positive
    # verdict appears in only a minority of horizons.
    # --------------------------------------------------------
    passing = [r for r in reports
               if r.get("decision", {}).get("verdict") == "PAPER_TEST"]
    if reports and 0 < len(passing) < max(2, len(reports) // 2):
        print()
        print("⚠ MULTIPLE-TESTING WARNING: a PAPER_TEST verdict appears in "
              f"only {len(passing)}/{len(reports)} horizons. With several "
              f"horizons tested, isolated positives are expected by chance — "
              f"treat single-horizon edges with strong suspicion.")

    # --------------------------------------------------------
    # Save all reports (via save_report — no more dead helper, ChatGPT #12)
    # --------------------------------------------------------

    output = {
        "generated_at":
            datetime.now(timezone.utc).isoformat(),
        "research_run_id":
            run_id,
        "cost_model_status":
            cost_status,
        "reports":
            reports,
    }

    save_report(output, "model_audit_report.json")

    # Persist authoritative per-instrument verdicts for the research advisor.
    # The advisor reads THIS table only — never the raw predictions — so an
    # uncalibrated/unqualified model can send nothing to the engines.
    persist_audit_verdicts(DATABASE_FILE, reports, run_id)

    print()
    print(
        "=" * 72
    )

    print(
        "ALL MODEL AUDITS COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    for report in reports:

        dec = report.get("decision", {})
        verdict = dec.get("verdict") if isinstance(dec, dict) else dec
        print(
            f"{report['model_version']} "
            f"| horizon={report['horizon']} "
            f"| verdict={verdict}"
        )


if __name__ == "__main__":

    main()
