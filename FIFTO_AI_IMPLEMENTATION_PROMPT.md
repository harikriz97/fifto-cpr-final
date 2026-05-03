# FIFTO Live Trading System — AI Implementation Prompt
## Complete handoff document for Claude Code / VS Code

---

## CONTEXT

You are implementing a **live paper trading system** for FIFTO — a NIFTY weekly options intraday selling strategy that has been backtested over 5 years (2021–2026) with:
- **949 trades | Win Rate 74.5% | Rs.16,96,299 | Max DD 2.93%**

The backtest is complete and verified bias-free. The task is to implement the **live execution layer** using Angel One SmartAPI so the system can paper trade in real market hours.

---

## REPOSITORY STRUCTURE

```
fifto-live/
├── CLAUDE.md                    ← project rules (THIS FILE when placed there)
├── live/
│   ├── config.py                ← credentials + parameters (FILL BEFORE RUN)
│   ├── angel_client.py          ← Angel One API wrapper (COMPLETE)
│   ├── indicators.py            ← CPR, EMA, HA, IB, MRC levels (COMPLETE)
│   ├── signal_crt.py            ← CRT scanner (COMPLETE — Spider-Man)
│   ├── signal_mrc.py            ← MRC scanner (COMPLETE — Black Widow)
│   ├── signal_base.py           ← BASE scanner (NEEDS IMPLEMENTATION)
│   ├── trade_manager.py         ← Entry/SL/target/exit (COMPLETE)
│   ├── paper_trader.py          ← Main orchestrator (COMPLETE — needs base hooked in)
│   └── requirements.txt
├── data/
│   └── 127_all_trades.csv       ← 5yr backtest trades for reference
└── FIFTO_Intraday_Selling_System.pdf  ← Client document
```

**Test command (no API needed):**
```bash
python3 live/paper_trader.py --test
```

---

## THE SYSTEM — 7 AGENTS

All agents are **option SELL** strategies on NIFTY weekly options:

| Agent | File | Status | Strategy |
|-------|------|--------|---------|
| THOR | signal_base.py | ❌ NEEDS BUILD | v17a — zone + EMA bias |
| HULK | signal_base.py | ❌ NEEDS BUILD | cam_l3 — Camarilla L3 touch |
| IRON MAN | signal_base.py | ❌ NEEDS BUILD | cam_h3 — Camarilla H3 touch |
| CAPTAIN | signal_base.py | ❌ NEEDS BUILD | iv2_pdl/r1/r2 — level touch |
| SPIDER-MAN | signal_crt.py | ✅ COMPLETE | CRT 15M+5M LTF |
| BLACK WIDOW | signal_mrc.py | ✅ COMPLETE | MRC HA 5M + Fibonacci |
| HAWKEYE | paper_trader.py | ✅ COMPLETE | S4 re-entry after target |

---

## WHAT NEEDS TO BE IMPLEMENTED

### `live/signal_base.py` — The Base Scanner (MAIN TASK)

Implements THOR + HULK + IRON MAN + CAPTAIN in one class.

#### Signal Logic (from backtest script `56_combined_full_backtest.py`)

**Step 1: Pre-market level computation** (already in `indicators.py`)
```python
# From prev day H/L/C:
pp  = (H + L + C) / 3
bc  = (H + L) / 2
tc  = 2*pp - bc
r1  = 2*pp - L
r2  = pp + (H - L)
r3  = r1 + (H - L)
r4  = r2 + (H - L)
s1  = 2*pp - H
s2  = pp - (H - L)
s3  = s1 - (H - L)
s4  = s2 - (H - L)
cam_h3 = prev_close + range * 1.1 / 4
cam_l3 = prev_close - range * 1.1 / 4
ema20  = EMA(20) of previous day close (40-day seed, shift 1)
```

**Step 2: Zone classification** — based on TODAY's OPEN price:
```python
def classify_zone(today_open, pvt, pdh, pdl):
    if   today_open > pvt['r4']: return 'above_r4'
    elif today_open > pvt['r3']: return 'r3_to_r4'
    elif today_open > pvt['r2']: return 'r2_to_r3'
    elif today_open > pvt['r1']: return 'r1_to_r2'
    elif today_open > pdh:       return 'pdh_to_r1'
    elif today_open > pvt['tc']: return 'tc_to_pdh'
    elif today_open >= pvt['bc']:return 'within_cpr'
    elif today_open > pdl:       return 'pdl_to_bc'
    elif today_open > pvt['s1']: return 'pdl_to_s1'
    elif today_open > pvt['s2']: return 's1_to_s2'
    elif today_open > pvt['s3']: return 's2_to_s3'
    elif today_open > pvt['s4']: return 's3_to_s4'
    else:                        return 'below_s4'
```

**Step 3: EMA bias**
```python
bias = 'bull' if today_open > ema20 else 'bear'
```

**Step 4: THOR signal lookup** — V17A_PARAMS table:
```python
# KEY: (zone, bias, opt_type)
# VALUE: (strike_type, entry_time, target_pct, sl_pct)
V17A_PARAMS = {
    ("above_r4",   "bull", "PE"): ("ITM1", "09:31:02", 0.50, 2.00),
    ("below_s4",   "bear", "CE"): ("ITM1", "09:16:02", 0.20, 0.50),
    ("pdh_to_r1",  "bear", "PE"): ("OTM1", "09:20:02", 0.50, 0.50),
    ("pdl_to_bc",  "bull", "PE"): ("OTM1", "09:31:02", 0.20, 1.50),
    ("pdl_to_s1",  "bear", "CE"): ("ITM1", "09:20:02", 0.20, 2.00),
    ("r1_to_r2",   "bear", "PE"): ("ATM",  "09:20:02", 0.50, 2.00),
    ("r1_to_r2",   "bull", "PE"): ("OTM1", "09:16:02", 0.50, 1.00),
    ("r2_to_r3",   "bull", "PE"): ("ATM",  "09:20:02", 0.20, 1.50),
    ("r2_to_r3",   "bear", "PE"): ("ATM",  "09:20:02", 0.20, 1.50),
    ("r3_to_r4",   "bull", "PE"): ("ITM1", "09:25:02", 0.20, 0.50),
    ("s1_to_s2",   "bear", "CE"): ("ATM",  "09:16:02", 0.50, 2.00),
    ("s3_to_s4",   "bear", "CE"): ("ITM1", "09:20:02", 0.40, 0.50),
    ("tc_to_pdh",  "bear", "PE"): ("OTM1", "09:31:02", 0.50, 0.50),
    ("tc_to_pdh",  "bull", "PE"): ("ITM1", "09:25:02", 0.20, 0.50),
    ("within_cpr", "bear", "CE"): ("ATM",  "09:16:02", 0.20, 2.00),
    ("within_cpr", "bull", "PE"): ("ATM",  "09:20:02", 0.30, 2.00),
}

# Strike type resolution:
def get_strike(atm, opt_type, stype, strike_interval=50):
    if opt_type == 'CE':
        return {'OTM1': atm + strike_interval,
                'ATM':  atm,
                'ITM1': atm - strike_interval}[stype]
    else:  # PE
        return {'OTM1': atm - strike_interval,
                'ATM':  atm,
                'ITM1': atm + strike_interval}[stype]
```

**THOR filter**: Previous day body must be > 0.10%
```python
prev_body = abs(prev_close - prev_open) / prev_open * 100
if prev_body <= 0.10: skip  # doji day = no signal
```

**QUALITY CUTS applied in backtest (must apply in live too):**
- Remove: `score == 6` → never fires (structural negative)
- Remove: `cam_h3 + tc_to_pdh` zone combination → direction mismatch (WR 42%)
- Remove: PE sells when `fut_basis_pts` between 50 and 100
- `inside_cpr` reduces lot count by 1 (min 1)

**Step 5: HULK signal** — Camarilla L3 touch:
```python
# HULK: spot touches or crosses CAM L3 from above (bearish)
# Direction: DOWN through L3 → sell PE (mean reversion up expected)
CAM_L3_PARAMS = dict(opt='PE', stype='ITM1', tgt=0.20, sl=0.50)

# Detection: scan spot ticks from 09:16 to 15:15
# Touch = price crosses L3 going DOWN (prev > L3, current <= L3 + tolerance)
# tolerance = level * 0.05 / 100
# Entry: next tick after touch + 2 seconds
```

**Step 6: IRON MAN signal** — Camarilla H3 touch:
```python
# IRON MAN: spot touches or crosses CAM H3 from below (bearish rejection)
# Direction: UP through H3 → sell CE (rejection expected)
CAM_H3_PARAMS = dict(opt='CE', stype='OTM1', tgt=0.50, sl=1.00)

# Skip window: do NOT enter if touch occurs between 09:20 and 09:30
# (false signals common in opening gap fill period)
# Detection same as HULK but direction='up'
```

**Step 7: CAPTAIN signal** — IV2 level touch:
```python
# iv2_pdl: spot touches PDL going DOWN → CE sell
# iv2_r1:  spot touches R1 going UP → PE sell (rejection)
# iv2_r2:  spot touches R2 going UP → PE sell (rejection)

IV2_PARAMS = {
    'PDL': dict(direction='down', opt='CE', stype='ATM', tgt=0.20, sl=0.50),
    'R1':  dict(direction='up',   opt='PE', stype='ATM', tgt=0.20, sl=0.50),
    'R2':  dict(direction='up',   opt='PE', stype='ATM', tgt=0.20, sl=0.50),
}
# Entry: next candle after touch + 2 seconds
```

#### Score7 computation (for lot sizing of THOR/HULK/IRON MAN/CAPTAIN):

```python
# All 7 features — each returns 0 or 1:
features = {
    'vix_ok':            int(vix_today < vix_ma20),
    'cpr_trend_aligned': int((direction=='CE' and prev_close < prev_bc) or
                             (direction=='PE' and prev_close > prev_tc)),
    'consec_aligned':    int(2 consecutive days close above/below CPR midpoint),
    'cpr_gap_aligned':   int((direction=='CE' and open_below_bc) or
                             (direction=='PE' and open_above_tc)),
    'dte_sweet':         int(3 <= days_to_expiry <= 5),
    'cpr_narrow':        int(0.10 <= cpr_width_pct <= 0.20),
    'cpr_dir_aligned':   int((direction=='PE' and cpr_midpoint_ascending_3days) or
                             (direction=='CE' and cpr_midpoint_descending_3days)),
}
score = sum(features.values())

# inside_cpr: today's CPR inside yesterday's CPR (reduces lots by 1)
inside_cpr = (today_tc < yesterday_tc) and (today_bc > yesterday_bc)

# Lot sizing:
# score 0-1 → 1 lot
# score 2-3 → 2 lots
# score 4+  → 3 lots
# if inside_cpr: lots = max(1, lots - 1)
```

---

## BASE SCANNER CLASS INTERFACE

Implement `signal_base.py` as a `BaseScanner` class with this interface:

```python
class BaseScanner:
    def __init__(self, levels: dict, ema20: float, prev_close: float,
                 prev_open: float, fut_basis_pts: float,
                 vix_today: float, vix_ma20: float, dte: int,
                 ohlc_history: pd.DataFrame):
        """
        levels: dict from compute_cpr() + compute_camarilla()
                keys: tc, bc, pivot, r1, r2, r3, r4, s1, s2, s3, s4,
                      cam_h3, cam_l3, pdh, pdl
        ema20:          yesterday's EMA20 value
        prev_close:     previous day close
        prev_open:      previous day open (for body filter)
        fut_basis_pts:  futures - spot at 09:15
        vix_today:      India VIX at open
        vix_ma20:       20-day MA of India VIX
        dte:            days to nearest weekly expiry
        ohlc_history:   pd.DataFrame with [date, open, high, low, close, tc, bc, cpr_mid]
                        for last 5 days (for consec_aligned and cpr_dir_aligned)
        """

    def on_open(self, today_open: float) -> dict | None:
        """
        Call once at 09:15:02 with today's first tick price.
        Determines zone, bias, and if THOR/CAPTAIN signal exists (fixed-time entry).
        Returns signal dict if THOR fires at 09:16-09:31, else None.
        """

    def on_spot_tick(self, time_str: str, price: float,
                     spot_ticks: pd.DataFrame) -> dict | None:
        """
        Call on every spot tick after 09:16.
        Monitors for HULK (CAM L3 touch) and IRON MAN (CAM H3 touch).
        Returns signal dict when a touch is detected, else None.
        """

    def get_lots(self, direction: str, today_open: float) -> int:
        """
        Compute lot count for a given direction using score7.
        direction: 'CE' or 'PE'
        """
```

**Signal dict format** (must match existing `trade_manager.py` input):
```python
{
    "strategy":   "v17a",         # or "cam_l3", "cam_h3", "iv2_pdl", "iv2_r1", "iv2_r2"
    "signal":     "THOR",         # Marvel name for logging
    "opt":        "PE",           # or "CE"
    "entry_time": "09:31:02",
    "strike":     23200,          # resolved ATM/OTM1/ITM1
    "lots":       2,
    "tgt_pct":    0.50,           # strategy-specific target
    "sl_pct":     0.50,           # strategy-specific hard SL
}
```

---

## HOW TO HOOK BASE SCANNER INTO paper_trader.py

In `paper_trader.py`, `on_ib_confirmed()` already inits CRT and MRC scanners.
After implementing `signal_base.py`, add:

```python
# In on_ib_confirmed():
state.base_scanner = BaseScanner(
    levels={**state.cpr, **state.cam, 'pdh': state.pdh, 'pdl': state.pdl},
    ema20=state.ema20,
    prev_close=state._prev_close,
    prev_open=state._prev_open,
    fut_basis_pts=state.fut_basis_pts,
    vix_today=state._vix_today,
    vix_ma20=state._vix_ma20,
    dte=state._dte,
    ohlc_history=state._ohlc_history,
)
# Fire on_open at 09:15:02
sig = state.base_scanner.on_open(today_first_tick)
if sig:
    state.base_signal_fired = True
    # schedule _enter_trade at sig['entry_time']

# In on_spot_tick():
if state.base_scanner and not state.base_signal_fired:
    sig = state.base_scanner.on_spot_tick(time_str, price, ticks_df)
    if sig:
        state.base_signal_fired = True
        _enter_trade(sig, state, client, time_str)
```

---

## TRADE MANAGEMENT (ALREADY IMPLEMENTED)

`trade_manager.py` handles exit exactly as backtest. Key: each strategy has its OWN
target and SL percentages. When creating `TradeManager`, pass the signal's tgt_pct/sl_pct.

Current `TradeManager` hardcodes TGT_PCT=0.30. **Update to use signal-specific values:**
```python
# trade_manager.py TradeManager.__init__() — change:
self.tgt = r2(entry_price * (1 - signal.get('tgt_pct', TGT_PCT)))
self.hsl = r2(entry_price * (1 + signal.get('sl_pct', 1.00)))
```

---

## ANGEL ONE API — KEY FACTS

```python
from SmartApi import SmartConnect
import pyotp

# Login
api = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_SECRET).now()
data = api.generateSession(CLIENT_ID, MPIN, totp)
auth_token = data['data']['jwtToken']
feed_token = api.getfeedToken()

# Historical OHLC (daily)
params = {
    "exchange": "NSE",
    "symboltoken": "26000",      # NIFTY 50 index
    "interval": "ONE_DAY",
    "fromdate": "2026-03-01 09:15",
    "todate": "2026-05-03 15:30",
}
resp = api.getCandleData(params)
# resp['data'] = list of [timestamp, open, high, low, close, volume]

# LTP
resp = api.ltpData("NSE", "Nifty 50", "26000")
ltp = resp['data']['ltp']

# WebSocket (real-time)
from SmartApi.SmartWebSocketV2 import SmartWebSocketV2
sws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_ID, feed_token)
sws.on_data = callback_fn   # receives tick dict
sws.on_open = lambda ws: sws.subscribe("id", 1, [{"exchangeType":1,"tokens":["26000"]}])
sws.connect()  # run in thread

# Tick price note: last_traded_price is in PAISE (divide by 100) for some instruments
# For NIFTY index: check if value > 100000, if so divide by 100

# Option token lookup: download scrip master
import requests
master = requests.get(
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
).json()
# Filter: exch_seg=="NFO", name=="NIFTY", instrumenttype=="OPTIDX"
# Symbol format: "NIFTY29MAY2026CE23200"
```

---

## DATA FILES FOR REFERENCE

| File | Description |
|------|-------------|
| `data/20260503/127_all_trades.csv` | 5yr backtest trades (949 rows) — compare live results |
| `data/20260430/124_base_clean.csv` | Base trades with strategy/zone/score columns |
| `data/20260430/124_s4_clean.csv` | S4 (HAWKEYE) trades |
| `data/consolidated/103_crt_mrc_atm30_*.csv` | CRT+MRC blank day trades |

127_all_trades.csv columns:
`date, component, signal, opt, entry_time, ep, xp, exit_reason, lots, pnl, win, year, cum_pnl`

---

## VALIDATION CHECKLIST

After implementing `signal_base.py`, run these checks:

```python
# 1. Test THOR zone classification
assert classify_zone(23250, pvt, pdh=23400, pdl=22900) == 'tc_to_pdh'

# 2. Test V17A lookup
assert get_v17a_signal('tc_to_pdh', 'bear') == ('PE', ('OTM1', '09:31:02', 0.50, 0.50))

# 3. Test score7
assert score_to_lots(4) == 3
assert score_to_lots(4, inside_cpr=True) == 2

# 4. Test paper trader dry run
# python3 live/paper_trader.py --test
# Should print DAY PLAN with all levels

# 5. Check no overlap: base signal days vs CRT/MRC days
# In backtest: assert len(base_dates & blank_dates) == 0
```

---

## RULES (READ BEFORE CODING)

1. **No forward bias**: All pre-market levels from previous day data only. EMA20 uses `shift(1)` — yesterday's EMA.
2. **Entry = next candle open + 2s**: Never enter on the signal candle itself.
3. **One trade per day**: Once base signal fires, CRT/MRC scanners stop. Once any trade entered, no new signals.
4. **Score 6 = never trade**: Filter out score==6 combinations (net negative over 5 years).
5. **cam_h3 + tc_to_pdh = skip**: Direction mismatch. If cam_h3 fires and zone is tc_to_pdh, do not trade.
6. **PE sell + basis 50-100 = skip**: fut_basis_pts between 50 and 100 → skip all PE sells from base agents.
7. **All P&L in Rs.**: LOT_SIZE=65 shares. pnl = (ep - xp) * 65 * lots. Round to 2 decimal places.
8. **PAPER_TRADE=True in config.py**: Default. Never place real orders until confirmed.

---

## RUNNING THE SYSTEM

```bash
# Install
pip install -r live/requirements.txt

# Configure (fill credentials)
vi live/config.py

# Dry run (no API)
python3 live/paper_trader.py --test

# Paper trade (run during market hours 09:00-15:30)
python3 live/paper_trader.py

# View today's results
python3 -c "
import pandas as pd
df = pd.read_csv('live/paper_trades.csv')
print(df.tail(10).to_string())
print('Total P&L:', df['pnl'].sum().round(0))
"
```

---

## EXPECTED LIVE PERFORMANCE (from 5yr backtest)

- **Avg trades/month**: ~16 base + ~5 blank = ~21 total
- **Avg monthly P&L**: Rs.29,247
- **Win rate**: 74.5%
- **Max single-day loss**: Rs.~17,000 (1 hard SL, 3-lot)
- **Negative months**: 3 out of 58 (5.2%)
- **Target hit rate**: 63.5% of all trades

---

*FIFTO v1.0 · Implementation prompt generated 2026-05-03*
*Backtest: 949 trades · 2021–2026 · Bias-free · All validations passed*
