"""
config.py — Angel One credentials and strategy parameters
==========================================================
Fill in YOUR credentials before running.
PAPER_TRADE = True by default — no real orders will be placed.
"""

# ── Angel One API Credentials ──────────────────────────────────────────────────
ANGEL_API_KEY     = ""          # from Angel One developer console
ANGEL_CLIENT_ID   = ""          # your Angel One login ID (e.g. A123456)
ANGEL_PASSWORD    = ""          # Angel One PIN (4-digit MPIN)
ANGEL_TOTP_SECRET = ""          # TOTP secret key (from Angel One 2FA setup)

# ── Mode ───────────────────────────────────────────────────────────────────────
PAPER_TRADE = True              # True = log only, no real orders

# ── Strategy Parameters (must match backtest exactly) ─────────────────────────
LOT_SIZE      = 65              # shares per lot (Nifty = 75 from Nov 2024, backtest uses 65)
STRIKE_INT    = 50              # Nifty strike interval
TGT_PCT       = 0.30            # 30% option premium target
EOD_EXIT_TIME = "15:20:00"      # force exit at this time

# ── IB (Initial Balance) window ───────────────────────────────────────────────
IB_START      = "09:15:00"
IB_END        = "09:45:00"      # IB is confirmed at 09:46

# ── Signal scan windows ────────────────────────────────────────────────────────
BASE_ENTRY_START  = "09:16:00"  # earliest base signal entry
BASE_ENTRY_END    = "15:15:00"
CRT_SCAN_START    = "09:15:00"  # 15M candles from open
CRT_SCAN_END      = "12:00:00"  # last 15M pattern C3 close ≤ 12:00
MRC_SCAN_START    = "09:15:00"  # HA 5M from open
MRC_SCAN_END      = "12:00:00"  # last HA candle ≤ 12:00

# ── Score7 lot sizing (must match backtest) ────────────────────────────────────
# score 0-1 → 1 lot, 2-3 → 2 lots, 4+ → 3 lots
# inside_cpr: reduce by 1 (min 1)
SCORE_LOT_MAP = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}

# ── MRC PE lot override (from script 127 decision) ────────────────────────────
MRC_PE_LOTS = 2                 # MRC PE always 2 lots (WR 80.6%, approved)
MRC_CE_LOTS = 0                 # MRC CE removed (net negative P&L)

# ── Output paths ───────────────────────────────────────────────────────────────
LOG_CSV     = "live/paper_trades.csv"
STATE_FILE  = "live/state.json"
OHLC_CACHE  = "live/ohlc_cache.csv"    # 45-day NIFTY daily OHLC cache
