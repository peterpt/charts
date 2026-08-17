"""
pt_ai_gate.py — shared AI trade-decision gate helper (peterpt 2026-07-26)

One function every engine (E1-E4) calls before sending a signal, when its
<engine>_ai_decision toggle is ON. Keeps the gate logic in ONE place so all
engines behave identically.

    final_dir = ai_gate_direction(engine, sym, direction, price, hub_dir,
                                  toggle_key)

Returns:
    • the engine's original direction   → trade as proposed (toggle off, or AI PROCEED)
    • the OPPOSITE direction            → AI flipped it (AI FLIP)
    • None                              → do not trade (AI CANCEL / neutral /
                                          below threshold / unavailable — fail-closed)

The heavy lifting (assemble features, call AI, apply threshold, decide
PROCEED/FLIP/CANCEL) lives in pt_engine5.e5_gate — this is just the thin,
uniform entry point engines use, plus the per-engine toggle check.
"""

from __future__ import annotations
import json
import os
from typing import Optional


def _toggle_on(hub_dir: str, toggle_key: str) -> bool:
    """Read <engine>_ai_decision from gui_settings.json (install root is the
    parent of hub_dir)."""
    try:
        root = os.path.dirname(os.path.abspath(hub_dir.rstrip(os.sep)))
        # hub_dir is usually <root>/hub_data, so root is its parent; but if a
        # bare 'hub_data' was passed, fall back to cwd.
        p = os.path.join(root, "gui_settings.json")
        if not os.path.isfile(p):
            p = os.path.join(os.getcwd(), "gui_settings.json")
        with open(p, "r", encoding="utf-8") as f:
            gs = json.load(f)
        return bool(gs.get(toggle_key, False))
    except Exception:
        return False


def _ai_mode(hub_dir: str, toggle_key: str) -> str:
    """Read <engine>_ai_mode from gui_settings.json — "gate" (default) or
    "log_only". (peterpt + Claude 2026-08-06)

    gate     — the AI verdict controls the trade (PROCEED/FLIP/CANCEL). Current,
               default behaviour; unchanged when the key is absent.
    log_only — the AI is still called and its decision logged to the scorecard,
               but the engine trades its OWN direction regardless. Lets you
               build up a per-engine track record of the AI's calls vs reality
               before trusting it to actually gate that engine.

    Key name is derived from the toggle key: e1_ai_decision -> e1_ai_mode.
    """
    mode_key = toggle_key.replace("_ai_decision", "_ai_mode")
    try:
        root = os.path.dirname(os.path.abspath(hub_dir.rstrip(os.sep)))
        pth = os.path.join(root, "gui_settings.json")
        if not os.path.isfile(pth):
            pth = os.path.join(os.getcwd(), "gui_settings.json")
        with open(pth, "r", encoding="utf-8") as f:
            gs = json.load(f)
        m = str(gs.get(mode_key, "gate")).lower().strip()
        return "log_only" if m == "log_only" else "gate"
    except Exception:
        return "gate"


def _live_price(hub_dir: str, sym: str, fallback: float) -> float:
    for p in (os.path.join(hub_dir, "ramfs", "live_prices.json"),
              os.path.join(hub_dir, "live_prices.json")):
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    lp = json.load(f)
                v = lp.get(sym)
                if isinstance(v, dict):
                    return float(v.get("mid") or v.get("price") or fallback)
        except Exception:
            pass
    return fallback


def ai_gate_direction(engine: str, sym: str, direction: str,
                      entry_price: float, hub_dir: str,
                      toggle_key: str,
                      logger=None,
                      extra_context: Optional[str] = None) -> Optional[str]:
    """Return the final direction to trade (possibly flipped), or None to cancel.

    engine     — label for logs/prompt ("E1","3A","3B","E2","E4")
    toggle_key — the gui_settings key for this engine's switch
                 ("e1_ai_decision", "e3_ai_decision", ...)
    logger     — optional callable(str) for the engine's own log
    extra_context — optional prompt line (e.g. E2's detected pattern) so the AI
                    knows WHY the engine wants the trade (peterpt 2026-07-29)
    """
    def _log(m):
        if logger:
            try: logger(m)
            except Exception: pass

    # Toggle off → engine behaves normally, no AI call.
    if not _toggle_on(hub_dir, toggle_key):
        return direction

    try:
        import pt_engine5 as _e5
        px = _live_price(hub_dir, sym, entry_price)
        res = _e5.e5_gate(engine, sym, direction, px, hub_dir=hub_dir,
                          extra_context=extra_context)
        decision = res.get("decision")
        ai_dir = res.get("direction")
        _log(f"[{engine}-AI] {sym} wanted {direction.upper()} @ {px} → "
             f"{decision} (AI={str(ai_dir).upper() if ai_dir else '-'}, "
             f"conf={res.get('confidence')}, thr={res.get('threshold')})"
             + (f" — {res['reasoning']}" if res.get('reasoning') else ""))
        # ── LOG-ONLY MODE (peterpt + Claude 2026-08-06) ──────────────────
        # The AI was called and e5_gate already logged the decision to the
        # scorecard above. In log_only we DO NOT let that verdict gate the
        # trade — the engine proceeds with its own direction. The point is to
        # accumulate the AI's opinion vs reality per engine before trusting it.
        if _ai_mode(hub_dir, toggle_key) == "log_only":
            _log(f"[{engine}-AI] {sym}: LOG-ONLY — decision {decision} recorded "
                 f"but not enforced; trading engine's own {direction.upper()}")
            return direction
        if decision == "PROCEED":
            return direction
        if decision == "FLIP":
            return ai_dir
        return None                      # CANCEL
    except Exception as e:
        _log(f"[{engine}-AI] {sym}: gate error {type(e).__name__}: {e} — "
             f"BLOCKING (fail-closed)")
        return None


def ai_gate_direction_price(engine: str, sym: str, direction: str,
                            entry_price: float, hub_dir: str,
                            toggle_key: str, logger=None,
                            extra_context: Optional[str] = None,
                            order_type: str = "LIMIT",
                            gap_mode: bool = False):
    """Like ai_gate_direction but also returns the AI-suggested entry price.
    Returns (final_direction, ai_price):
      • (direction, None)  toggle off — normal behaviour, no AI price
      • (dir, price|None)  PROCEED/FLIP — price is the AI's suggested entry
                           (None if the AI didn't give one → caller uses its own)
      • (None, None)       CANCEL / unavailable (fail-closed)
    Used by E1, which places a pre-order at the AI's price (+ threshold) when
    given, else at the neural price (peterpt 2026-07-28).

    order_type — "LIMIT" or "STOP". The price is only valid on the correct side
    of current price, and which side that is DEPENDS on the order type. E1 must
    pass the type for the touch case it is handling (A/C are STOP, B/D LIMIT);
    engines that rest LIMIT orders can leave the default.

    ── 2026-08-03 (peterpt + Claude): NOW RETURNS THE *VALIDATED* PRICE ───────
    This used to return res["ai_price"] — the RAW number the model produced,
    with no sanity check and NOT gated by e5_use_ai_price. E1 consumed it
    directly (pt_engine1 ~line 2858, `_lv = float(_ai_price)`), so E1 has been
    re-basing pre-orders onto unvalidated AI prices even with the toggle off,
    while E3 — which reads res["use_ai_price"] — correctly ignored them.

    It now returns res["use_ai_price"], the same verdict E3 uses, which
    enforces: the e5_use_ai_price toggle, a positive real number, the correct
    side for the order type, and a maximum distance from current price
    (e5_ai_price_max_dist_pct, default 2%) to reject hallucinated levels.

    CONSEQUENCE, deliberate: E1 will use AI prices LESS often than before, and
    not at all while e5_use_ai_price is off. That is the intended behaviour —
    the thinker's neural is an untested reference, and an AI price replacing it
    should be at least as well checked, not less."""
    def _log(m):
        if logger:
            try: logger(m)
            except Exception: pass
    if not _toggle_on(hub_dir, toggle_key):
        return direction, None
    try:
        import pt_engine5 as _e5
        px = _live_price(hub_dir, sym, entry_price)
        res = _e5.e5_gate(engine, sym, direction, px, hub_dir=hub_dir,
                          extra_context=extra_context, order_type=order_type,
                          gap_mode=gap_mode)
        decision = res.get("decision")
        ai_dir = res.get("direction")
        # Validated verdict {use, price, reason} — NOT the raw res["ai_price"].
        _v = res.get("use_ai_price") or {}
        ai_price = _v.get("price") if _v.get("use") else None
        if _v and not _v.get("use") and _v.get("price"):
            # The AI named a price and it was refused. Say so explicitly —
            # otherwise this looks identical to the AI not suggesting one.
            _log(f"[{engine}-AI] {sym}: AI price {_v.get('price')} NOT used "
                 f"({_v.get('reason')}) — falling back to the engine's own level")
        _log(f"[{engine}-AI] {sym} wanted {direction.upper()} @ {px} → "
             f"{decision} (AI={str(ai_dir).upper() if ai_dir else '-'}, "
             f"conf={res.get('confidence')}"
             + (f", price={ai_price}" if ai_price else "")
             + f", thr={res.get('threshold')})"
             + (f" — {res['reasoning']}" if res.get('reasoning') else ""))
        # LOG-ONLY: decision already logged by e5_gate; do not let it gate the
        # trade, and do NOT feed the AI price into the order — the engine uses
        # its own direction and its own level. (peterpt + Claude 2026-08-06)
        if _ai_mode(hub_dir, toggle_key) == "log_only":
            _log(f"[{engine}-AI] {sym}: LOG-ONLY — decision {decision} recorded "
                 f"but not enforced; trading engine's own {direction.upper()} "
                 f"at its own price")
            return direction, None
        if decision == "PROCEED":
            return direction, ai_price
        if decision == "FLIP":
            return ai_dir, ai_price
        return None, None
    except Exception as e:
        _log(f"[{engine}-AI] {sym}: gate error {type(e).__name__}: {e} — "
             f"BLOCKING (fail-closed)")
        return None, None
