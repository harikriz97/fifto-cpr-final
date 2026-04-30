"""
70_intraday_v2_r1r2.py
Intraday v2 — R1/R2/PDL break entries on no-signal days at LOT=65

Findings from optimize_intraday_v2.py (LOT=75):
  R1  → PE ATM,  target 20%, SL  50%  → 29t, 76-81% WR
  R2  → PE ITM1, target 50%, SL 100%  → 5t,  80%    WR
  PDL → CE ATM,  target 30%, SL 200%  → 26t, 62%    WR
Skip: TC, S1, S2, PDH (negative/weak)
Scan window: 09:30-11:20 (best from optimization)

Sections:
  A. R1/R2/PDL backtest at LOT=65 — year-wise breakdown
  B. DTE analysis
  C. Combined with 69_final_trades for total equity
"""
import sys, os, glob, warnings, time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_tick_data, load_spot_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np
import plotly.graph_objects as go

FOLDER    = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell'
OUT_DIR   = f'{FOLDER}/data/20260430'
LOT_SIZE  = 65
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_EXIT   = '15:20:00'
SCAN_FROM  = '09:30'
SCAN_TO    = '11:20'
YEARS      = 5
BODY_MIN   = 0.10   # min prev-day body size % to trade

PARAMS = {
    'R1':  ('PE', 'ATM',  0.20, 0.50),
    'R2':  ('PE', 'ITM1', 0.50, 1.00),
    'PDL': ('CE', 'ATM',  0.30, 2.00),
}

os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp = r2((h + l + c) / 3)
    bc = r2((h + l) / 2)
    tc = r2(2 * pp - bc)
    r1 = r2(2 * pp - l);  r2_ = r2(pp + (h - l))
    s1 = r2(2 * pp - h);  s2_ = r2(pp - (h - l))
    return dict(pp=pp, bc=bc, tc=tc, r1=r1, r2=r2_, s1=s1, s2=s2_)

def classify_zone(op, pvt, pdh, pdl):
    if   op > pvt['r2']:  return 'r2_to_r3_plus'
    elif op > pvt['r1']:  return 'r1_to_r2'
    elif op > pdh:        return 'pdh_to_r1'
    elif op > pvt['tc']:  return 'tc_to_pdh'
    elif op >= pvt['bc']: return 'within_cpr'
    elif op > pdl:        return 'pdl_to_bc'
    elif op > pvt['s1']:  return 'pdl_to_s1'
    elif op > pvt['s2']:  return 's1_to_s2'
    else:                 return 'below_s2'

def get_v17a_signal(zone, ema_bias):
    if zone in {'r2_to_r3_plus', 'r1_to_r2'}:           return 'PE'
    if zone == 'pdh_to_r1'   and ema_bias == 'bear':    return 'PE'
    if zone == 'tc_to_pdh':                              return 'PE'
    if zone == 'within_cpr'  and ema_bias == 'bull':     return 'PE'
    if zone == 'within_cpr'  and ema_bias == 'bear':     return 'CE'
    if zone == 'pdl_to_bc'   and ema_bias == 'bull':     return 'PE'
    if zone in {'pdl_to_s1', 's1_to_s2', 'below_s2'} and ema_bias == 'bear': return 'CE'
    return None

def get_strike(atm, opt_type, stype):
    if opt_type == 'CE':
        return {'ATM': atm, 'ITM1': atm - STRIKE_INT, 'OTM1': atm + STRIKE_INT}[stype]
    return {'ATM': atm, 'ITM1': atm + STRIKE_INT, 'OTM1': atm - STRIKE_INT}[stype]

def sim_pct(ts, ps, ep, eod_ns, tgt_pct, sl_pct):
    """% target + hard SL simulation."""
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct))
    sl = hsl; md = 0.0
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= eod_ns:
            return r2((ep - p) * LOT_SIZE), 'eod', p
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt:
            return r2((ep - p) * LOT_SIZE), 'target', p
        if p >= sl:
            return r2((ep - p) * LOT_SIZE), 'lockin_sl' if sl < hsl else 'hard_sl', p
    return r2((ep - ps[-1]) * LOT_SIZE), 'eod', ps[-1]

def detect_break(ohlc5, pvt, pdh, pdl):
    """Detect first R1/R2/PDL break in scan window."""
    up_levels = [('R2', pvt['r2']), ('R1', pvt['r1'])]
    dn_levels = [('PDL', pdl)]

    try:
        scan = ohlc5.between_time(SCAN_FROM, SCAN_TO)
    except Exception:
        return None
    if len(scan) < 2: return None

    candles = scan.reset_index()
    ts_col  = candles.columns[0]
    for idx in range(1, len(candles)):
        row  = candles.iloc[idx]
        prev = candles.iloc[idx - 1]
        c_close = row['close']; p_close = prev['close']
        c_time  = row[ts_col]
        entry_dt = c_time + pd.Timedelta(minutes=5, seconds=2)
        for name, level in up_levels:
            if p_close <= level < c_close:
                return dict(entry_dt=entry_dt, level_name=name, level=level, opt='PE')
        for name, level in dn_levels:
            if p_close >= level > c_close:
                return dict(entry_dt=entry_dt, level_name=name, level=level, opt='CE')
    return None


# ── Pass 1: daily OHLC + EMA ──────────────────────────────────────────────────
print(f"Pass 1: loading daily OHLC + EMA({EMA_PERIOD})...")
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4] + '-' + all_dates[-1][4:6] + '-' + all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)
t0 = time.time()
daily_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) == 0: continue
    daily_ohlc[d] = (
        round(day['price'].max(), 2),
        round(day['price'].min(), 2),
        round(day.iloc[-1]['price'], 2),
        round(day.iloc[0]['price'], 2),
    )

close_s = pd.Series({d: v[2] for d, v in daily_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD, adjust=False).mean()
print(f"  {len(daily_ohlc)} days in {time.time()-t0:.0f}s")


# ── Pass 2: simulate R1/R2/PDL breaks on no-signal days ──────────────────────
print("Pass 2: scanning no-signal days (R1/R2/PDL)...")
records = []
t1 = time.time(); processed = 0

for date in dates_5yr:
    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx - 1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph, pl, pc, _ = daily_ohlc[prev]
    _, _, _, today_op = daily_ohlc[date]

    pvt   = compute_pivots(ph, pl, pc)
    e20   = ema_s.get(prev, np.nan)
    if np.isnan(e20): continue

    # Skip thin prev-day candles
    prev_body = abs(pc - daily_ohlc[prev][3]) / daily_ohlc[prev][3] * 100
    if prev_body <= BODY_MIN: continue

    bias   = 'bull' if today_op > e20 else 'bear'
    zone   = classify_zone(today_op, pvt, ph, pl)
    signal = get_v17a_signal(zone, bias)

    # Only no-signal days
    if signal is not None: continue

    dstr    = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    expiries = list_expiry_dates(date)
    if not expiries: continue
    expiry  = expiries[0]
    exp_dt  = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte     = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: continue

    # Build 5-min OHLC
    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty: continue
    spot_tks['dt'] = pd.to_datetime(dstr + ' ' + spot_tks['time'])
    sp = spot_tks[['dt', 'price']].set_index('dt').rename(columns={'price': 'p'})
    ohlc5 = sp['p'].resample('5min', closed='left', label='left').agg(
        open='first', high='max', low='min', close='last').dropna()
    if len(ohlc5) < 2: continue

    pdh = r2(ph); pdl_val = r2(pl)
    atm = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    brk = detect_break(ohlc5, pvt, pdh, pdl_val)
    if brk is None: continue

    level_name = brk['level_name']
    if level_name not in PARAMS: continue
    opt_type, stype, tgt_pct, sl_pct = PARAMS[level_name]

    strike = get_strike(atm, opt_type, stype)
    instr  = f'NIFTY{expiry}{strike}{opt_type}'
    ot = load_tick_data(date, instr, '09:15:00', '15:30:00')
    if ot is None or ot.empty: continue
    ot['dt'] = pd.to_datetime(dstr + ' ' + ot['time'])
    entry_mask = ot['dt'] >= brk['entry_dt']
    if not entry_mask.any(): continue
    ot_entry = ot[entry_mask]
    ep = float(ot_entry['price'].iloc[0])
    if ep <= 0: continue

    opt_ts = ot_entry['dt'].values.astype('datetime64[ns]').astype('int64')
    opt_ps = ot_entry['price'].values.astype(float)

    pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt_pct, sl_pct)

    records.append(dict(
        date=dstr, strategy='intraday_v2', level=level_name, opt=opt_type,
        strike=strike, strike_type=stype, dte=dte,
        entry_time=brk['entry_dt'].strftime('%H:%M:%S'),
        entry_price=r2(ep), exit_price=r2(xp), exit_reason=reason,
        pnl=pnl, win=pnl > 0, year=date[:4],
    ))

    processed += 1
    if processed % 50 == 0:
        print(f"  {processed} no-signal days processed, {len(records)} trades found...")

print(f"Pass 2 done in {time.time()-t1:.0f}s  |  {processed} no-signal days  |  {len(records)} trades")

if not records:
    print("No trades found — check data path.")
    import sys; sys.exit(1)

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ── A. OVERALL + YEAR-WISE ────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  A. INTRADAY V2 — R1/R2/PDL — OVERALL RESULTS (LOT=65)')
print(f'{"="*65}')

total_wr  = df['win'].mean() * 100
total_pnl = df['pnl'].sum()
avg_pnl   = df['pnl'].mean()
print(f'  Trades: {len(df)}  |  WR: {total_wr:.1f}%  |  Avg: ₹{avg_pnl:,.0f}  |  Total: ₹{total_pnl:,.0f}')

print(f'\n  {"Level":<8} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*45}')
for lvl in ['R1', 'R2', 'PDL']:
    g = df[df['level'] == lvl]
    if len(g) == 0: continue
    print(f'  {lvl:<8} {len(g):>5} {g["win"].mean()*100:>7.1f}% {g["pnl"].mean():>10,.0f} {g["pnl"].sum():>12,.0f}')

print(f'\n  {"Year":<8} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*45}')
for yr in sorted(df['year'].unique()):
    g = df[df['year'] == yr]
    print(f'  {yr:<8} {len(g):>5} {g["win"].mean()*100:>7.1f}% {g["pnl"].mean():>10,.0f} {g["pnl"].sum():>12,.0f}')

# ── B. DTE ANALYSIS ───────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  B. DTE ANALYSIS')
print(f'{"="*65}')
print(f'  {"DTE":<6} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*45}')
for dte in sorted(df['dte'].unique()):
    g = df[df['dte'] == dte]
    print(f'  {dte:<6} {len(g):>5} {g["win"].mean()*100:>7.1f}% {g["pnl"].mean():>10,.0f} {g["pnl"].sum():>12,.0f}')

# ── C. COMBINED WITH 69 TRADES ────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  C. COMBINED WITH 69_final_trades (v17a + cam_l3 + cam_h3 + intraday_v2)')
print(f'{"="*65}')

final_path = f'{FOLDER}/data/20260430/69_final_trades.csv'
if os.path.exists(final_path):
    t69 = pd.read_csv(final_path)
    t69.columns = [c.lower().replace(' ', '_') for c in t69.columns]
    if 'date' in t69.columns:
        t69['date'] = pd.to_datetime(t69['date'].astype(str).str.replace('-', '').str[:8])

    # Align pnl column names
    if 'pnl_conv' in t69.columns:
        t69_pnl = t69[['date', 'pnl_conv']].rename(columns={'pnl_conv': 'pnl'})
    elif 'pnl_65' in t69.columns:
        t69_pnl = t69[['date', 'pnl_65']].rename(columns={'pnl_65': 'pnl'})
    else:
        t69_pnl = t69[['date', 'pnl']].copy()

    v2_pnl = df[['date', 'pnl']].copy()
    all_pnl = pd.concat([t69_pnl, v2_pnl], ignore_index=True)

    t69_total = t69_pnl['pnl'].sum()
    v2_total  = v2_pnl['pnl'].sum()
    combined  = all_pnl['pnl'].sum()

    print(f'  69 trades  (conviction): ₹{t69_total:>10,.0f}')
    print(f'  Intraday v2 R1/R2/PDL:  ₹{v2_total:>10,.0f}  (+{len(df)}t)')
    print(f'  Combined total:          ₹{combined:>10,.0f}  (+₹{combined - t69_total:,.0f})')
else:
    print(f'  69_final_trades.csv not found at {final_path}')
    all_pnl = df[['date', 'pnl']].copy()
    t69_total = 0

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = f'{OUT_DIR}/70_intraday_v2_trades.csv'
df.to_csv(save_path, index=False)
print(f'\n  Saved → {save_path}  ({len(df)} rows)')

# ── CHARTS ────────────────────────────────────────────────────────────────────
from plot_util import plot_equity, super_plotter, send_custom_chart

# Chart 1: Intraday v2 equity curve
eq_v2 = df.set_index('date')['pnl'].sort_index().cumsum()
dd_v2 = (eq_v2 - eq_v2.cummax())
plot_equity(eq_v2, dd_v2, '70_intraday_v2_equity',
            title='Intraday v2 R1/R2/PDL — Equity (LOT=65)')

# Chart 2: Combined equity
if os.path.exists(final_path) and t69_total != 0:
    all_pnl_sorted = all_pnl.sort_values('date')
    eq_all = all_pnl_sorted.groupby('date')['pnl'].sum().cumsum()
    dd_all = (eq_all - eq_all.cummax())
    plot_equity(eq_all, dd_all, '70_combined_equity',
                title='Combined (v17a + cam_l3 + cam_h3 + Intraday v2) — Conviction Sizing')

# Chart 3: Per-level — as line series in tv_json format
lvl_data = df.groupby('level').agg(
    trades=('pnl', 'count'),
    wr=('win', lambda x: round(x.mean()*100, 1)),
    total_pnl=('pnl', 'sum')
).reset_index()

# Build cumulative equity per level as lines
lines = []
colors = {'R1': '#26a69a', 'R2': '#1e88e5', 'PDL': '#ef5350'}
for lvl in ['R1', 'R2', 'PDL']:
    g = df[df['level'] == lvl].sort_values('date')
    if len(g) == 0: continue
    eq = g.set_index('date')['pnl'].cumsum()
    pts = [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
           for d, v in eq.items()]
    lines.append({"id": lvl, "data": pts, "color": colors.get(lvl, '#888888'),
                  "lineWidth": 2, "title": f"{lvl} (WR {lvl_data[lvl_data['level']==lvl]['wr'].values[0]}%)"})

tv_json = {"candlestick": [], "volume": [], "lines": lines, "markers": [], "isTvFormat": True}
send_custom_chart('70_level_breakdown', tv_json,
                  title='Intraday v2 — R1/R2/PDL Cumulative PnL')

print('\nDone.')
