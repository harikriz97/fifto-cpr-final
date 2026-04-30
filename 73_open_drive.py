"""
73_open_drive.py
Open Drive strategy — first 5-min candle closes above PDH or below PDL

Signal: if 09:15-09:20 candle (first 5-min) closes above PDH → sell PE ATM
        if 09:15-09:20 candle (first 5-min) closes below PDL → sell CE ATM

Only on days with NO existing v17a/cam signal AND no intraday v2 trade.

Sections:
  A. Full backtest results — year-wise, level breakdown
  B. Parameter sensitivity (target/SL combos)
  C. DTE analysis
  D. Combined total with scripts 72 + 70
"""
import sys, os, glob, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_tick_data, load_spot_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np

FOLDER     = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell'
OUT_DIR    = f'{FOLDER}/data/20260430'
LOT_SIZE   = 65
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_EXIT   = '15:20:00'
YEARS      = 5
BODY_MIN   = 0.10

# Best params to test — will pick winner
PARAM_COMBOS = [
    ('ATM',  0.20, 0.50),
    ('ATM',  0.25, 0.50),
    ('ATM',  0.30, 1.00),
    ('OTM1', 0.20, 0.50),
    ('OTM1', 0.25, 0.50),
]

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
    if   op > pvt['r2']:  return 'r2_plus'
    elif op > pvt['r1']:  return 'r1_to_r2'
    elif op > pdh:        return 'pdh_to_r1'
    elif op > pvt['tc']:  return 'tc_to_pdh'
    elif op >= pvt['bc']: return 'within_cpr'
    elif op > pdl:        return 'pdl_to_bc'
    elif op > pvt['s1']:  return 'pdl_to_s1'
    elif op > pvt['s2']:  return 's1_to_s2'
    else:                 return 'below_s2'

def get_v17a_signal(zone, ema_bias):
    if zone in {'r2_plus', 'r1_to_r2'}:                          return 'PE'
    if zone == 'pdh_to_r1'  and ema_bias == 'bear':              return 'PE'
    if zone == 'tc_to_pdh':                                       return 'PE'
    if zone == 'within_cpr' and ema_bias == 'bull':               return 'PE'
    if zone == 'within_cpr' and ema_bias == 'bear':               return 'CE'
    if zone == 'pdl_to_bc'  and ema_bias == 'bull':               return 'PE'
    if zone in {'pdl_to_s1','s1_to_s2','below_s2'} and ema_bias=='bear': return 'CE'
    return None

def get_strike(atm, opt_type, stype):
    if opt_type == 'CE':
        return {'ATM': atm, 'ITM1': atm-STRIKE_INT, 'OTM1': atm+STRIKE_INT}[stype]
    return {'ATM': atm, 'ITM1': atm+STRIKE_INT, 'OTM1': atm-STRIKE_INT}[stype]

def sim_pct(ts, ps, ep, eod_ns, tgt_pct, sl_pct):
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


# ── Already-covered dates ─────────────────────────────────────────────────────
covered_dates = set()
try:
    t56 = pd.read_csv('data/56_combined_trades.csv')
    t56.columns = [c.lower().replace(' ','_') for c in t56.columns]
    t56['date'] = t56['date'].astype(str).str.replace('-','').str[:8]
    covered_dates |= set(t56['date'].unique())
except: pass
try:
    t70 = pd.read_csv(f'{OUT_DIR}/70_intraday_v2_trades.csv')
    t70['date2'] = pd.to_datetime(t70['date']).dt.strftime('%Y%m%d')
    covered_dates |= set(t70['date2'].unique())
except: pass
print(f'Already covered dates: {len(covered_dates)}')

# ── Pass 1: daily OHLC + EMA ──────────────────────────────────────────────────
print(f'Pass 1: loading daily OHLC + EMA({EMA_PERIOD})...')
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)

t0 = time.time()
daily_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    # first 5-min close = price at or just before 09:20
    first5 = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '09:20:00')]
    open5_close = float(first5.iloc[-1]['price']) if len(first5) > 0 else float(day.iloc[0]['price'])
    daily_ohlc[d] = (
        round(day['price'].max(), 2),
        round(day['price'].min(), 2),
        round(day.iloc[-1]['price'], 2),
        round(day.iloc[0]['price'], 2),
        open5_close,  # first 5-min candle close
    )

close_s = pd.Series({d: v[2] for d, v in daily_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD, adjust=False).mean()
print(f'  {len(daily_ohlc)} days in {time.time()-t0:.0f}s')

# ── Pass 2: scan Open Drive days ──────────────────────────────────────────────
print('Pass 2: scanning Open Drive days...')
records = []
t1 = time.time(); processed = 0

for date in dates_5yr:
    # Skip already-covered dates
    if date in covered_dates: continue

    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx - 1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph, pl, pc, _, _ = daily_ohlc[prev]
    _, _, _, today_op, open5_close = daily_ohlc[date]

    pvt   = compute_pivots(ph, pl, pc)
    e20   = ema_s.get(prev, np.nan)
    if np.isnan(e20): continue

    prev_body = abs(pc - daily_ohlc[prev][3]) / daily_ohlc[prev][3] * 100
    if prev_body <= BODY_MIN: continue

    bias   = 'bull' if today_op > e20 else 'bear'
    zone   = classify_zone(today_op, pvt, ph, pl)
    signal = get_v17a_signal(zone, bias)
    if signal is not None: continue  # v17a has a signal — skip

    # Check Open Drive
    pdh = r2(ph); pdl = r2(pl)
    if open5_close > pdh:
        od_opt = 'PE'   # price broke above PDH → bearish fade → sell PE
    elif open5_close < pdl:
        od_opt = 'CE'   # price broke below PDL → bullish fade → sell CE
    else:
        continue        # no OD signal

    dstr    = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    expiries = list_expiry_dates(date)
    if not expiries: continue
    expiry  = expiries[0]
    exp_dt  = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte     = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: continue

    atm     = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns  = pd.Timestamp(dstr + ' ' + EOD_EXIT).value
    entry_dt = pd.Timestamp(dstr + ' 09:20:02')

    for stype, tgt_pct, sl_pct in PARAM_COMBOS:
        strike = get_strike(atm, od_opt, stype)
        instr  = f'NIFTY{expiry}{strike}{od_opt}'
        ot = load_tick_data(date, instr, '09:15:00', '15:30:00')
        if ot is None or ot.empty: continue
        ot['dt'] = pd.to_datetime(dstr + ' ' + ot['time'])
        em = ot['dt'] >= entry_dt
        if not em.any(): continue
        ot_e = ot[em]
        ep = float(ot_e['price'].iloc[0])
        if ep <= 0: continue

        opt_ts = ot_e['dt'].values.astype('datetime64[ns]').astype('int64')
        opt_ps = ot_e['price'].values.astype(float)
        pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt_pct, sl_pct)

        records.append(dict(
            date=dstr, strategy='open_drive', opt=od_opt,
            strike=strike, strike_type=stype, dte=dte,
            entry_price=r2(ep), exit_price=r2(xp), exit_reason=reason,
            pnl=pnl, win=pnl > 0, year=date[:4],
            tgt_pct=tgt_pct, sl_pct=sl_pct, param=f'{stype}_t{int(tgt_pct*100)}_sl{int(sl_pct*100)}',
        ))

    processed += 1
    if processed % 100 == 0:
        print(f'  {processed} OD candidate days processed, {len(records)} records...')

print(f'Pass 2 done in {time.time()-t1:.0f}s  |  {processed} OD days  |  {len(records)} records')

if not records:
    print('No OD trades found.')
    import sys; sys.exit(1)

df = pd.DataFrame(records)

# ── A. PARAMETER SWEEP ────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  A. PARAMETER SWEEP')
print(f'{"="*65}')
print(f'  {"Params":<25} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*60}')
best_total = -999999; best_param = None
for param in df['param'].unique():
    g = df[df['param']==param].drop_duplicates('date')  # 1 trade per day per param
    wr  = g['win'].mean()*100
    avg = g['pnl'].mean()
    tot = g['pnl'].sum()
    print(f'  {param:<25} {len(g):>5} {wr:>7.1f}% {avg:>10,.0f} {tot:>12,.0f}')
    if tot > best_total:
        best_total = tot; best_param = param

print(f'\n  Best param: {best_param}')

# Use best param for remaining analysis
best_stype, best_tgt, best_sl = None, None, None
for stype, tgt, sl in PARAM_COMBOS:
    p = f'{stype}_t{int(tgt*100)}_sl{int(sl*100)}'
    if p == best_param:
        best_stype, best_tgt, best_sl = stype, tgt, sl

df_best = df[df['param']==best_param].drop_duplicates('date').copy()
df_best = df_best.sort_values('date').reset_index(drop=True)

# ── B. YEAR-WISE ──────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'  B. YEAR-WISE (best param: {best_param})')
print(f'{"="*65}')
print(f'  {"Year":<6} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*45}')
for yr in sorted(df_best['year'].unique()):
    g = df_best[df_best['year']==yr]
    print(f'  {yr:<6} {len(g):>5} {g["win"].mean()*100:>7.1f}% {g["pnl"].mean():>10,.0f} {g["pnl"].sum():>12,.0f}')

# ── C. DTE ANALYSIS ───────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  C. DTE ANALYSIS')
print(f'{"="*65}')
print(f'  {"DTE":<6} {"t":>5} {"WR":>8} {"Avg":>10} {"Total":>12}')
print(f'  {"-"*45}')
for dte in sorted(df_best['dte'].unique()):
    g = df_best[df_best['dte']==dte]
    print(f'  {dte:<6} {len(g):>5} {g["win"].mean()*100:>7.1f}% {g["pnl"].mean():>10,.0f} {g["pnl"].sum():>12,.0f}')

# ── D. COMBINED TOTAL ─────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('  D. COMBINED (72 conviction + 70 intraday v2 + 73 open drive)')
print(f'{"="*65}')
t72_path = f'{OUT_DIR}/72_final_trades.csv'
t70_path = f'{OUT_DIR}/70_intraday_v2_trades.csv'

totals = {}
if os.path.exists(t72_path):
    t72 = pd.read_csv(t72_path)
    t72.columns = [c.lower().replace(' ','_') for c in t72.columns]
    pnl_col = 'pnl_conv7n' if 'pnl_conv7n' in t72.columns else 'pnl_final'
    totals['72 conviction (480t)'] = t72[pnl_col].sum()
if os.path.exists(t70_path):
    t70 = pd.read_csv(t70_path)
    totals['70 intraday v2 (70t)'] = t70['pnl'].sum()

od_total = df_best['pnl'].sum()
totals[f'73 open drive ({len(df_best)}t)'] = od_total
grand = sum(totals.values())

for k, v in totals.items():
    print(f'  {k:<30} ₹{v:>10,.0f}')
print(f'  {"─"*42}')
print(f'  {"GRAND TOTAL":<30} ₹{grand:>10,.0f}')

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = f'{OUT_DIR}/73_open_drive_trades.csv'
df_best.to_csv(save_path, index=False)
print(f'\n  Saved → {save_path}  ({len(df_best)} rows)')

# ── CHARTS ────────────────────────────────────────────────────────────────────
from plot_util import plot_equity, send_custom_chart

# Chart 1: OD equity curve
eq_od = df_best.set_index(pd.to_datetime(df_best['date']))['pnl'].cumsum()
dd_od = eq_od - eq_od.cummax()
plot_equity(eq_od, dd_od, '73_od_equity', title='Open Drive — Equity (LOT=65)')

# Chart 2: Grand combined equity
all_pnl_dfs = []
if os.path.exists(t72_path):
    t72_eq = pd.read_csv(t72_path)
    t72_eq.columns = [c.lower().replace(' ','_') for c in t72_eq.columns]
    pnl_col = 'pnl_conv7n' if 'pnl_conv7n' in t72_eq.columns else 'pnl_final'
    t72_eq['date_ts'] = pd.to_datetime(t72_eq['date'].astype(str).str[:8])
    all_pnl_dfs.append(t72_eq[['date_ts', pnl_col]].rename(columns={pnl_col:'pnl'}))
if os.path.exists(t70_path):
    t70_eq = pd.read_csv(t70_path)
    t70_eq['date_ts'] = pd.to_datetime(t70_eq['date'])
    all_pnl_dfs.append(t70_eq[['date_ts','pnl']])

od_eq = df_best[['date','pnl']].copy()
od_eq['date_ts'] = pd.to_datetime(od_eq['date'])
all_pnl_dfs.append(od_eq[['date_ts','pnl']])

if all_pnl_dfs:
    combined = pd.concat(all_pnl_dfs, ignore_index=True)
    eq_all = combined.sort_values('date_ts').set_index('date_ts')['pnl'].cumsum()
    dd_all = eq_all - eq_all.cummax()
    plot_equity(eq_all, dd_all, '73_grand_equity',
                title='Grand Total: Conviction + Intraday v2 + Open Drive')

print('\nDone.')
