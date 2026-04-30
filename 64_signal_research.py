"""
64_signal_research.py
Test everything from deep research:

  S1. DTE Sweet Spot      — WR/avg by DTE bucket (1,2,3,4,5,6+)
  S2. CPR Width Buckets   — ultra-narrow / narrow / medium / wide
  S3. ±300pt Weekly Range — skip if spot already >250pts from Friday open
  S4. H4/L4 Breakout Buy  — buy CE/PE when prev close crosses Cam H4/L4
  S5. 0DTE 2:45pm Straddle— sell ATM CE+PE at 14:45, exit 15:20 (expiry Thursday)
  S6. Virgin CPR Deeper   — WR when prev-day was virgin vs non-virgin CPR
"""
import sys, os, warnings, time, multiprocessing as mp
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart

from my_util import (list_trading_dates, load_spot_data, load_tick_data,
                     fetch_option_chain, list_expiry_dates)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_ROOT  = os.environ['INTER_SERVER_DATA_PATH']
LOT        = 65
SCALE      = 65 / 75
STRIKE_INT = 50
EOD_STR    = '15:20:00'

def r2(v): return round(float(v), 2)

def show(label, pnls, n=None):
    pnls = pd.Series(pnls, dtype=float)
    n_   = n or len(pnls)
    if n_ == 0: print(f"  {label:<35}  0t"); return
    wr   = r2((pnls > 0).mean() * 100)
    tot  = r2(pnls.sum())
    avg  = r2(pnls.mean())
    print(f"  {label:<35} {n_:>4}t | WR {wr:>5.1f}% | Avg {avg:>7,.0f} | Total Rs {tot:>10,.0f}")

# ─────────────────────────────────────────────
# Load base data
# ─────────────────────────────────────────────
df_all = pd.read_csv('data/56_combined_trades.csv', parse_dates=['date'])
v17a   = df_all[df_all['strategy'] == 'v17a'].copy().sort_values('date').reset_index(drop=True)
v17a['pnl65'] = v17a['pnl'] * SCALE

all_dates = list_trading_dates()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def days_to_expiry(date_str):
    """Calendar days from date to next Thursday (NIFTY weekly expiry)."""
    dt = datetime.strptime(date_str, '%Y%m%d')
    days = (3 - dt.weekday()) % 7   # Thu=3
    return days if days > 0 else 7   # if today=Thu→7 (next Thu), 0 handled separately

def compute_cpr(h, l, c):
    pp = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * pp - bc
    return r2(pp), r2(tc), r2(bc)

def compute_cam(h, l, c):
    rng = h - l
    return {
        'h3': r2(c + rng * 1.1 / 4), 'l3': r2(c - rng * 1.1 / 4),
        'h4': r2(c + rng * 1.1 / 2), 'l4': r2(c - rng * 1.1 / 2),
    }

def sim_sell(opt_tks, ep, tgt_pct, sl_pct, eod_str):
    """Simulate option sell. Returns (exit_price, exit_time, reason)."""
    tgt  = r2(ep * (1 - tgt_pct))
    hsl  = r2(ep * (1 + sl_pct))
    sl   = hsl
    md   = 0.0
    eod  = pd.Timestamp('1970-01-01 ' + eod_str)
    for _, row in opt_tks.iterrows():
        t  = pd.Timestamp('1970-01-01 ' + row['time'])
        cp = float(row['price'])
        if t >= eod:
            return r2(cp), row['time'], 'eod'
        d = (ep - cp) / ep
        if d > md:
            md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if cp <= tgt: return r2(cp), row['time'], 'target'
        if cp >= sl:  return r2(cp), row['time'], 'hard_sl'
    return r2(float(opt_tks['price'].iloc[-1])), opt_tks['time'].iloc[-1], 'eod'

def sim_buy(opt_tks, ep, tgt_mult, sl_mult, eod_str):
    """Simulate option buy. Returns (exit_price, exit_time, reason)."""
    tgt = r2(ep * tgt_mult)
    hsl = r2(ep * sl_mult)
    eod = pd.Timestamp('1970-01-01 ' + eod_str)
    for _, row in opt_tks.iterrows():
        t  = pd.Timestamp('1970-01-01 ' + row['time'])
        cp = float(row['price'])
        if t >= eod: return r2(cp), row['time'], 'eod'
        if cp >= tgt: return r2(cp), row['time'], 'target'
        if cp <= hsl: return r2(cp), row['time'], 'hard_sl'
    return r2(float(opt_tks['price'].iloc[-1])), opt_tks['time'].iloc[-1], 'eod'

def get_atm(spot_price):
    return int(round(spot_price / STRIKE_INT)) * STRIKE_INT

# ─────────────────────────────────────────────
# Load daily OHLC once (used by multiple sections)
# ─────────────────────────────────────────────
print("Loading daily OHLC...")
t0 = time.time()
daily_ohlc = {}
for d in all_dates:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty: continue
    p = tks['price']
    daily_ohlc[d] = (float(p.max()), float(p.min()),
                     float(p.iloc[-1]), float(p.iloc[0]))
print(f"  {len(daily_ohlc)} days in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────
# S1. DTE SWEET SPOT ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  S1. DTE SWEET SPOT ANALYSIS")
print("=" * 65)

v17a['dte'] = v17a['date'].apply(lambda d: days_to_expiry(d.strftime('%Y%m%d')))

print(f"  {'DTE':<8} {'Trades':>7} {'WR':>9} {'Avg P&L':>10} {'Total P&L':>12}")
print(f"  {'-'*50}")
for dte in sorted(v17a['dte'].unique()):
    sub = v17a[v17a['dte'] == dte]
    wr  = sub['pnl65'].gt(0).mean() * 100
    avg = sub['pnl65'].mean()
    tot = sub['pnl65'].sum()
    flag = ' <-- BEST' if avg == v17a.groupby('dte')['pnl65'].mean().max() else ''
    print(f"  DTE={dte:<4} {len(sub):>7} {wr:>8.1f}% {avg:>10,.0f} {tot:>12,.0f}{flag}")

# Grouped buckets
print(f"\n  Grouped buckets:")
v17a['dte_bucket'] = pd.cut(v17a['dte'],
    bins=[0, 1, 2, 3, 4, 5, 99],
    labels=['DTE=1','DTE=2','DTE=3','DTE=4','DTE=5','DTE=6+'])
for bkt, g in v17a.groupby('dte_bucket', observed=True):
    wr  = g['pnl65'].gt(0).mean() * 100
    avg = g['pnl65'].mean()
    show(str(bkt), g['pnl65'])

# ─────────────────────────────────────────────
# S2. CPR WIDTH BUCKET ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  S2. CPR WIDTH BUCKET ANALYSIS")
print("=" * 65)

# Build CPR width for each trade date
cpr_width_map = {}
for i, d in enumerate(all_dates):
    if i < 1 or d not in daily_ohlc: continue
    prev = all_dates[i-1]
    if prev not in daily_ohlc: continue
    ph, pl, pc, _ = daily_ohlc[prev]
    _, tc, bc = compute_cpr(ph, pl, pc)
    spot_o = daily_ohlc[d][3]
    cpr_width_map[d] = abs(tc - bc) / spot_o * 100

v17a['cpr_w'] = v17a['date'].apply(lambda d: cpr_width_map.get(d.strftime('%Y%m%d'), np.nan))

buckets = [
    ('Ultra-narrow (<0.10%)',  v17a['cpr_w'] < 0.10),
    ('Narrow (0.10–0.20%)',   (v17a['cpr_w'] >= 0.10) & (v17a['cpr_w'] < 0.20)),
    ('Medium (0.20–0.35%)',   (v17a['cpr_w'] >= 0.20) & (v17a['cpr_w'] < 0.35)),
    ('Wide (0.35–0.50%)',     (v17a['cpr_w'] >= 0.35) & (v17a['cpr_w'] < 0.50)),
    ('Very wide (>0.50%)',     v17a['cpr_w'] >= 0.50),
]
print(f"  {'Bucket':<28} {'Trades':>7} {'WR':>9} {'Avg P&L':>10} {'Total':>12}")
print(f"  {'-'*65}")
for label, mask in buckets:
    sub = v17a[mask]
    if len(sub) == 0: continue
    wr  = sub['pnl65'].gt(0).mean() * 100
    avg = sub['pnl65'].mean()
    tot = sub['pnl65'].sum()
    print(f"  {label:<28} {len(sub):>7} {wr:>8.1f}% {avg:>10,.0f} {tot:>12,.0f}")

# ─────────────────────────────────────────────
# S3. ±300pt WEEKLY RANGE FILTER (Durgia 2025)
# Skip trade if spot already >250pt from Friday open
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  S3. ±300pt WEEKLY RANGE FILTER (Durgia 2025 paper)")
print("=" * 65)
print("  Logic: skip if today_open is >250pts from prev Friday open")

# Build Friday open map
friday_open = {}
for d in all_dates:
    if d not in daily_ohlc: continue
    dt = datetime.strptime(d, '%Y%m%d')
    if dt.weekday() == 4:  # Friday
        friday_open[d] = daily_ohlc[d][3]  # open price

def get_prev_friday_open(date_str):
    dt = datetime.strptime(date_str, '%Y%m%d')
    # go back to last Friday
    days_back = (dt.weekday() - 4) % 7
    if days_back == 0:
        days_back = 7  # if today is Friday, get last Friday
    target = dt - timedelta(days=days_back)
    # search near that date
    for i in range(5):
        key = (target - timedelta(days=i)).strftime('%Y%m%d')
        if key in friday_open:
            return friday_open[key]
    return None

v17a['fri_open'] = v17a['date'].apply(lambda d: get_prev_friday_open(d.strftime('%Y%m%d')))
v17a['spot_open'] = v17a['date'].apply(lambda d: daily_ohlc.get(d.strftime('%Y%m%d'), (0,0,0,0))[3])
v17a['dist_from_fri'] = (v17a['spot_open'] - v17a['fri_open']).abs()

DIST_THRESH = 250
in_range  = v17a[v17a['dist_from_fri'] <= DIST_THRESH]
out_range = v17a[v17a['dist_from_fri'] > DIST_THRESH]

print(f"\n  Trades within ±{DIST_THRESH}pt of Friday open:")
show(f"Within ±{DIST_THRESH}pt (take trade)",  in_range['pnl65'])
show(f"Beyond ±{DIST_THRESH}pt (skip trade)", out_range['pnl65'])

# Try different thresholds
print(f"\n  Threshold sweep:")
print(f"  {'Threshold':>12} {'Kept':>6} {'WR':>9} {'Avg P&L':>10}")
for thresh in [150, 200, 250, 300, 350]:
    sub = v17a[v17a['dist_from_fri'] <= thresh]
    if len(sub) == 0: continue
    wr  = sub['pnl65'].gt(0).mean() * 100
    avg = sub['pnl65'].mean()
    print(f"  ≤{thresh:>9}pt  {len(sub):>6} {wr:>8.1f}% {avg:>10,.0f}")

# ─────────────────────────────────────────────
# S4. H4/L4 CAMARILLA BREAKOUT BUY
# When prev close > H4 → next day BUY CE
# When prev close < L4 → next day BUY PE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  S4. H4/L4 CAMARILLA BREAKOUT BUY")
print("=" * 65)
print("  Logic: prev close > H4 → buy ATM CE | prev close < L4 → buy ATM PE")

YEARS  = 5
latest = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

# Entry params: buy CE/PE (buying, not selling)
BUY_TGT  = 2.0   # target = 2x entry (double)
BUY_SL   = 0.50  # SL = 50% of entry (lose half)
BUY_ENTRY = '09:16:02'

def run_h4l4_day(date):
    """
    Intraday H4/L4 breakout: during the day, if spot price crosses H4 → buy CE,
    if it crosses L4 → buy PE. Entry at the tick after crossing + 2s.
    """
    import sys as _sys
    _sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

    from my_util import load_spot_data, load_tick_data, list_expiry_dates

    idx = all_dates.index(date)
    if idx < 1: return []
    prev = all_dates[idx - 1]
    if prev not in daily_ohlc or date not in daily_ohlc: return []

    ph, pl, pc, _ = daily_ohlc[prev]
    cam = compute_cam(ph, pl, pc)
    h4  = cam['h4']
    l4  = cam['l4']

    # Load spot ticks to detect intraday H4/L4 break
    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty: return []

    # Scan from 09:16 to 13:00 for first H4 or L4 break
    scan = spot_tks[(spot_tks['time'] >= '09:16:00') &
                    (spot_tks['time'] <= '13:00:00')].reset_index(drop=True)
    if len(scan) < 2: return []

    touch_time = None
    opt_type   = None
    tol = h4 * 0.0005  # 0.05% tolerance

    for i in range(1, len(scan)):
        prev_p = float(scan.iloc[i-1]['price'])
        cur_p  = float(scan.iloc[i]['price'])
        t      = scan.iloc[i]['time']
        if opt_type is None:
            if prev_p < h4 and cur_p >= h4 - tol:
                opt_type   = 'CE'
                touch_time = t
                break
            if prev_p > l4 and cur_p <= l4 + tol:
                opt_type   = 'PE'
                touch_time = t
                break

    if opt_type is None: return []

    # Entry = touch_time + 2 seconds
    from datetime import datetime, timedelta
    tt = datetime.strptime(touch_time, '%H:%M:%S') + timedelta(seconds=2)
    entry_time = tt.strftime('%H:%M:%S')
    if entry_time >= EOD_STR: return []

    try:
        expiries = list_expiry_dates(date)
        if not expiries: return []
        expiry = expiries[0]  # YYMMDD format
    except Exception:
        return []

    entry_tks = spot_tks[spot_tks['time'] >= entry_time]
    if entry_tks.empty: return []
    atm = get_atm(float(entry_tks.iloc[0]['price']))

    inst    = f'NIFTY{expiry}{atm}{opt_type}'
    opt_tks = load_tick_data(date, inst, entry_time)
    if opt_tks is None or opt_tks.empty: return []

    after = opt_tks[opt_tks['time'] >= entry_time].reset_index(drop=True)
    if after.empty: return []
    ep = float(after.iloc[0]['price'])
    if ep < 5: return []

    xp, xt, reason = sim_buy(after, ep, BUY_TGT, BUY_SL, EOD_STR)
    pnl = r2((xp - ep) * LOT)

    return [{'date': date, 'opt': opt_type, 'ep': ep, 'xp': xp,
             'exit_reason': reason, 'pnl': pnl,
             'h4': h4, 'l4': l4, 'touch_time': touch_time}]

print("  Running H4/L4 backtest (parallel)...")
t2 = time.time()
with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
    results_h4l4 = pool.map(run_h4l4_day, dates_5yr)
h4l4_trades = [r for day in results_h4l4 for r in day]
df_h4l4 = pd.DataFrame(h4l4_trades) if h4l4_trades else pd.DataFrame()

if not df_h4l4.empty:
    print(f"  Done in {time.time()-t2:.1f}s — {len(df_h4l4)} signals")
    show('H4/L4 Breakout Buy (all)', df_h4l4['pnl'])
    ce_trades = df_h4l4[df_h4l4['opt'] == 'CE']
    pe_trades = df_h4l4[df_h4l4['opt'] == 'PE']
    show('  H4 Breakout → Buy CE',  ce_trades['pnl'])
    show('  L4 Breakdown → Buy PE', pe_trades['pnl'])
    df_h4l4['year'] = pd.to_datetime(df_h4l4['date']).dt.year
    print(f"\n  Year-wise:")
    for yr, g in df_h4l4.groupby('year'):
        show(f'    {yr}', g['pnl'])
    exit_dist = df_h4l4['exit_reason'].value_counts().to_dict()
    print(f"\n  Exit reasons: {exit_dist}")
    df_h4l4.to_csv('data/20260428/64_h4l4_breakout.csv', index=False)
    print("  Saved: data/20260428/64_h4l4_breakout.csv")
else:
    print("  No trades generated")

# ─────────────────────────────────────────────
# S5. 0DTE 2:45pm STRADDLE SELL
# On expiry Thursday: sell ATM CE + PE at 14:45, exit 15:20
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  S5. 0DTE 2:45pm STRADDLE SELL (Expiry Thursday)")
print("=" * 65)
print("  Logic: sell ATM CE + PE at 14:45 on expiry day, exit 15:20")

STRADDLE_ENTRY = '14:45:02'
STRADDLE_EOD   = '15:20:00'

# Find all expiry Thursdays in 5yr window
expiry_thursdays = []
for d in dates_5yr:
    dt = datetime.strptime(d, '%Y%m%d')
    if dt.weekday() == 3:  # Thursday
        try:
            expiries = list_expiry_dates(d)
            # expiries are YYMMDD, date is YYYYMMDD — compare last 6 chars
            if expiries and expiries[0] == d[2:]:
                expiry_thursdays.append(d)
        except Exception:
            pass

print(f"  Found {len(expiry_thursdays)} expiry Thursdays")

def run_straddle_day(date):
    import sys as _sys
    _sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
    from my_util import load_spot_data, load_tick_data, list_expiry_dates

    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty: return []

    entry_tks = spot_tks[spot_tks['time'] >= STRADDLE_ENTRY]
    if entry_tks.empty: return []
    spot_price = float(entry_tks.iloc[0]['price'])
    atm = get_atm(spot_price)

    try:
        expiries = list_expiry_dates(date)
        if not expiries: return []
        expiry = expiries[0]  # YYMMDD format e.g. '260428'
        # date is YYYYMMDD e.g. '20260428', compare last 6 chars
        if expiry != date[2:]: return []  # only trade on actual expiry day
    except Exception:
        return []

    inst_ce = f'NIFTY{expiry}{atm}CE'
    inst_pe = f'NIFTY{expiry}{atm}PE'
    ce_tks  = load_tick_data(date, inst_ce, STRADDLE_ENTRY)
    pe_tks  = load_tick_data(date, inst_pe, STRADDLE_ENTRY)
    if ce_tks is None or ce_tks.empty: return []
    if pe_tks is None or pe_tks.empty: return []

    ce_after = ce_tks[ce_tks['time'] >= STRADDLE_ENTRY].reset_index(drop=True)
    pe_after = pe_tks[pe_tks['time'] >= STRADDLE_ENTRY].reset_index(drop=True)
    if ce_after.empty or pe_after.empty: return []

    ce_ep = float(ce_after.iloc[0]['price'])
    pe_ep = float(pe_after.iloc[0]['price'])
    if ce_ep < 2 or pe_ep < 2: return []

    # Sell both — exit at EOD only (very short window, no S/T)
    ce_exit_tks = ce_after[ce_after['time'] >= STRADDLE_EOD]
    pe_exit_tks = pe_after[pe_after['time'] >= STRADDLE_EOD]

    ce_xp = float(ce_exit_tks.iloc[0]['price']) if not ce_exit_tks.empty else float(ce_after.iloc[-1]['price'])
    pe_xp = float(pe_exit_tks.iloc[0]['price']) if not pe_exit_tks.empty else float(pe_after.iloc[-1]['price'])

    total_ep  = r2(ce_ep + pe_ep)
    total_xp  = r2(ce_xp + pe_xp)
    pnl_total = r2((total_ep - total_xp) * LOT)

    return [{'date': date, 'atm': atm, 'spot': r2(spot_price),
             'ce_ep': ce_ep, 'pe_ep': pe_ep, 'total_ep': total_ep,
             'ce_xp': ce_xp, 'pe_xp': pe_xp, 'total_xp': total_xp,
             'pnl': pnl_total}]

print("  Running 0DTE straddle backtest (parallel)...")
t3 = time.time()
with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
    results_straddle = pool.map(run_straddle_day, expiry_thursdays)
straddle_trades = [r for day in results_straddle for r in day]
df_straddle = pd.DataFrame(straddle_trades) if straddle_trades else pd.DataFrame()

if not df_straddle.empty:
    print(f"  Done in {time.time()-t3:.1f}s — {len(df_straddle)} trades")
    show('0DTE Straddle 14:45 sell', df_straddle['pnl'])
    df_straddle['year'] = pd.to_datetime(df_straddle['date']).dt.year
    print(f"\n  Year-wise:")
    for yr, g in df_straddle.groupby('year'):
        show(f'    {yr}', g['pnl'])
    avg_ep  = r2(df_straddle['total_ep'].mean())
    avg_xp  = r2(df_straddle['total_xp'].mean())
    avg_decay = r2(avg_ep - avg_xp)
    print(f"\n  Avg entry premium:  Rs {avg_ep:.2f}")
    print(f"  Avg exit premium:   Rs {avg_xp:.2f}")
    print(f"  Avg decay captured: Rs {avg_decay:.2f} ({r2(avg_decay/avg_ep*100)}%)")
    df_straddle.to_csv('data/20260428/64_0dte_straddle.csv', index=False)
    print("  Saved: data/20260428/64_0dte_straddle.csv")
else:
    print("  No trades generated")

# ─────────────────────────────────────────────
# S6. VIRGIN CPR DEEPER ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  S6. VIRGIN CPR DEEPER ANALYSIS")
print("=" * 65)
print("  Virgin CPR = today's CPR does NOT overlap yesterday's CPR (CPR gap)")
print("  Bull Virgin: today BC > yesterday TC (CPR gapped UP = strong bull trend)")
print("  Bear Virgin: today TC < yesterday BC (CPR gapped DOWN = strong bear trend)")

virgin_map   = {}   # d → True/False (any gap)
vir_bull_map = {}   # d → True/False (bull gap)
vir_bear_map = {}   # d → True/False (bear gap)

for i, d in enumerate(all_dates):
    if i < 2 or d not in daily_ohlc: continue
    prev  = all_dates[i-1]
    prev2 = all_dates[i-2]
    if prev not in daily_ohlc or prev2 not in daily_ohlc: continue

    # Today's CPR = from prev day
    ph, pl, pc, _ = daily_ohlc[prev]
    _, tc_today, bc_today = compute_cpr(ph, pl, pc)
    cpr_lo_today = min(tc_today, bc_today)
    cpr_hi_today = max(tc_today, bc_today)

    # Yesterday's CPR = from prev2 day
    p2h, p2l, p2c, _ = daily_ohlc[prev2]
    _, tc_prev, bc_prev = compute_cpr(p2h, p2l, p2c)
    cpr_lo_prev = min(tc_prev, bc_prev)
    cpr_hi_prev = max(tc_prev, bc_prev)

    # Gap: no overlap between today's CPR and yesterday's CPR
    bull_gap = cpr_lo_today > cpr_hi_prev   # today's CPR fully above yesterday's
    bear_gap = cpr_hi_today < cpr_lo_prev   # today's CPR fully below yesterday's

    virgin_map[d]   = bull_gap or bear_gap
    vir_bull_map[d] = bull_gap
    vir_bear_map[d] = bear_gap

v17a['virgin']      = v17a['date'].apply(lambda d: virgin_map.get(d.strftime('%Y%m%d'), False))
v17a['vir_bull']    = v17a['date'].apply(lambda d: vir_bull_map.get(d.strftime('%Y%m%d'), False))
v17a['vir_bear']    = v17a['date'].apply(lambda d: vir_bear_map.get(d.strftime('%Y%m%d'), False))

virgin_yes = v17a[v17a['virgin'] == True]
virgin_no  = v17a[v17a['virgin'] == False]

print(f"\n  {'Category':<40} {'Count':>5}")
print(f"  {'-'*50}")
print(f"  Any CPR gap (bull or bear):           {v17a['virgin'].sum():>5}")
print(f"  Bull gap only:                        {v17a['vir_bull'].sum():>5}")
print(f"  Bear gap only:                        {v17a['vir_bear'].sum():>5}")
print()
show('CPR gap day (bull+bear)',         virgin_yes['pnl65'])
show('No CPR gap (normal overlap)',     virgin_no['pnl65'])

# Direction-aligned: bull gap + PE trade, bear gap + CE trade
vir_aligned = v17a[(v17a['vir_bull'] & (v17a['opt']=='PE')) |
                   (v17a['vir_bear'] & (v17a['opt']=='CE'))]
show('CPR gap ALIGNED with trade dir', vir_aligned['pnl65'])

# Combine virgin + vix_below_ma (compound filter)
# Load VIX for the filter
vix_data = {}
for d in all_dates:
    path = os.path.join(DATA_ROOT, d, 'INDIAVIX.csv')
    if os.path.exists(path):
        try:
            tks = pd.read_csv(path, header=None, names=['date','time','price','vol','oi'])
            vix_data[d] = float(tks['price'].iloc[-1])
        except Exception: pass
vix_s  = pd.Series(vix_data)
vix_s.index = pd.to_datetime(vix_s.index, format='%Y%m%d')
vix_ma = vix_s.rolling(20, min_periods=5).mean()
v17a['vix_ok'] = v17a['date'].apply(
    lambda d: bool(vix_s.get(d, np.nan) < vix_ma.get(d, np.nan)))

print(f"\n  Compound filter (virgin + VIX below MA):")
compound = v17a[v17a['virgin'] & v17a['vix_ok']]
show('CPR gap + VIX below MA', compound['pnl65'])

# Extra: CPR gap aligned + VIX below MA
compound2 = v17a[(v17a['vir_bull'] & (v17a['opt']=='PE') & v17a['vix_ok']) |
                 (v17a['vir_bear'] & (v17a['opt']=='CE') & v17a['vix_ok'])]
show('CPR gap aligned + VIX below MA', compound2['pnl65'])

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
ts_base = int(pd.Timestamp('2021-01-01').timestamp())

# Chart 1: DTE WR bar
dte_stats = v17a.groupby('dte').agg(wr=('pnl65', lambda x: (x>0).mean()*100),
                                     avg=('pnl65','mean'), n=('pnl65','count')).reset_index()
dte_bar = [{'time': ts_base + int(r.dte) * 86400 * 60,
            'value': round(float(r.wr), 1),
            'color': '#26a69a' if r.wr >= 71.9 else '#ef5350',
            'label': f'DTE={int(r.dte)} ({int(r.n)}t)'}
           for _, r in dte_stats.iterrows()]
send_custom_chart('dte_wr', {
    'lines': [{'id':'dte','label':'WR by DTE','seriesType':'bar','data':dte_bar,
               'xLabels':[b['label'] for b in dte_bar]}],
    'candlestick':[],'volume':[],'isTvFormat':False},
    title='DTE Sweet Spot — Win Rate | Base=71.9%')
print("\n📊 DTE WR chart sent")

# Chart 2: CPR width WR
width_data = []
for label, mask in buckets:
    sub = v17a[mask]
    if len(sub) == 0: continue
    wr = sub['pnl65'].gt(0).mean() * 100
    width_data.append((label, wr, len(sub)))
w_bar = [{'time': ts_base + i * 86400 * 90,
          'value': round(wr, 1),
          'color': '#26a69a' if wr >= 71.9 else '#f59e0b',
          'label': f'{label[:18]}({n}t)'}
         for i, (label, wr, n) in enumerate(width_data)]
send_custom_chart('cpr_width_wr', {
    'lines': [{'id':'width','label':'WR by CPR width','seriesType':'bar','data':w_bar,
               'xLabels':[b['label'] for b in w_bar]}],
    'candlestick':[],'volume':[],'isTvFormat':False},
    title='CPR Width Buckets — Win Rate')
print("📊 CPR width WR chart sent")

# Chart 3: H4/L4 equity (if exists)
if not df_h4l4.empty:
    df_h4l4_s = df_h4l4.sort_values('date').reset_index(drop=True)
    eq_h4l4   = df_h4l4_s['pnl'].cumsum()
    h4l4_line = [{'time': int(pd.Timestamp(row.date).timestamp()), 'value': round(float(v),2)}
                 for (_, row), v in zip(df_h4l4_s.iterrows(), eq_h4l4)]
    send_custom_chart('h4l4_equity', {
        'lines': [{'id':'h4l4','label':'H4/L4 Breakout Buy','color':'#a78bfa','data':h4l4_line}],
        'candlestick':[],'volume':[],'isTvFormat':False},
        title=f'H4/L4 Breakout Buy Equity | {len(df_h4l4)}t | Rs {r2(df_h4l4.pnl.sum()):,.0f}')
    print("📊 H4/L4 equity chart sent")

# Chart 4: Straddle equity (if exists)
if not df_straddle.empty:
    df_st_s = df_straddle.sort_values('date').reset_index(drop=True)
    eq_st   = df_st_s['pnl'].cumsum()
    st_line = [{'time': int(pd.Timestamp(row.date).timestamp()), 'value': round(float(v),2)}
               for (_, row), v in zip(df_st_s.iterrows(), eq_st)]
    send_custom_chart('straddle_equity', {
        'lines': [{'id':'straddle','label':'0DTE Straddle 14:45','color':'#f59e0b','data':st_line}],
        'candlestick':[],'volume':[],'isTvFormat':False},
        title=f'0DTE Straddle 14:45 | {len(df_straddle)}t | Rs {r2(df_straddle.pnl.sum()):,.0f}')
    print("📊 0DTE straddle chart sent")

print("\nAll done!")
