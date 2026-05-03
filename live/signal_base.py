"""
signal_base.py — Base strategy scanner (THOR / HULK / IRON MAN / CAPTAIN AMERICA)
===================================================================================
THIS FILE NEEDS TO BE IMPLEMENTED.
See FIFTO_AI_IMPLEMENTATION_PROMPT.md for full specification.

Strategies covered:
  THOR        — V17A pivot zones (CE and PE)
  HULK        — Camarilla L3 touch → CE sell
  IRON MAN    — Camarilla H3 / IV2 R1/R2 → PE sell
  CAPTAIN     — IV2 PDL / R1 / R2 → PE sell

Each strategy is a sub-scanner inside BaseScanner.
"""

from __future__ import annotations
import logging
from datetime import datetime
from config import LOT_SIZE, EOD_EXIT_TIME

logger = logging.getLogger(__name__)


class BaseScanner:
    """
    Base strategy scanner — NOT YET IMPLEMENTED.

    Interface (must be preserved):
    ─────────────────────────────────────────────────────────────────
    Constructor:
        BaseScanner(levels: dict, ema20: float, today_open: float,
                    fut_basis_pts: float)

        levels keys:
            pivot, tc, bc, r1, r2, r3, s1, s2, s3   ← CPR/Pivot
            cam_h3, cam_l3                            ← Camarilla
            pdh, pdl                                  ← Prev day high/low
            ib_high, ib_low                           ← IB (set after 09:46)

    Methods:
        update(spot_ticks: list[dict]) -> dict | None
            Called with latest spot ticks every time new ticks arrive.
            Returns signal dict if entry triggered, else None.
            Signal dict format:
                {
                    "strategy":   "BASE",
                    "sub":        "THOR" | "HULK" | "IRON MAN" | "CAPTAIN",
                    "signal":     "CE" | "PE",
                    "opt":        "CE" | "PE",
                    "strike":     int,
                    "entry_time": "HH:MM:SS",
                    "lots":       int,
                    "score":      int,
                    "zone":       str,
                }

        set_ib(ib_high: float, ib_low: float)
            Call after IB is confirmed at 09:46 to update IB levels.

        is_done() -> bool
            Returns True once a signal has been fired (sequential trading).

    ─────────────────────────────────────────────────────────────────
    See FIFTO_AI_IMPLEMENTATION_PROMPT.md for:
      - V17A_PARAMS full lookup table (16 entries)
      - classify_zone() logic
      - compute_score7() feature formulas
      - Quality cuts (score==6 skip, cam_h3+tc_to_pdh skip, PE basis skip)
      - detect_touch() for HULK / IRON MAN / CAPTAIN
      - Full implementation checklist
    """

    def __init__(self, levels: dict, ema20: float, today_open: float,
                 fut_basis_pts: float):
        self.levels        = levels
        self.ema20         = ema20
        self.today_open    = today_open
        self.fut_basis_pts = fut_basis_pts
        self._done         = False
        logger.info("BaseScanner initialised (NOT YET IMPLEMENTED)")

    def set_ib(self, ib_high: float, ib_low: float):
        self.levels["ib_high"] = ib_high
        self.levels["ib_low"]  = ib_low

    def update(self, spot_ticks: list) -> dict | None:
        """TODO: implement full base strategy logic."""
        return None

    def is_done(self) -> bool:
        return self._done
