"""
114_quality_expansion.py — Quality trade expansion
===================================================
Two tracks:

TRACK A: Blank days — add quality entries
  Filter blank days for D4 CPR (25-55pts) + wide IB + Ichimoku aligned
  Entry: IB failure on those filtered days
  Expectation: blank day IB quality improves significantly

TRACK B: Base days — 2nd trade after target exit
  After base trade exits via TARGET, look for pullback re-entry
  Pullback = option premium bounces back to 65-75% of original entry
  Re-enter same direction, same target/SL rules
  Only if re-entry before 13:30 (time left for second move)

Uses 113_full_day_analysis.xlsx all_days sheet as day classifier
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
from plot_util import send_custom_chart

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
IB_END     = '09:45:00'
ENTRY_START= '09:46:00'
ENTRY_END  = '13:00:00'
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
BRK_BUF    = 0.0005
PULLBACK_LO = 0.60   # re-entry if option bounces back to 60-75% of original EP
PULLBACK_HI = 0.75
REENTRY_CUTOFF = '13:30:00'
CPR_D4_LO  = 25.0    # CPR width range for D4 class
CPR_D4_HI  = 55.0

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot / STRIKE_INT) * STRIKE_INT)
def t2m(t):
    h,m,s = map(int,t.split(':'))
    return h*60+m+s/60


def simulate_sell(date_str, instrument, entry_time, label=''):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - TGT_PCT))
    hsl = r2(ep * (1 + 1.00)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep-p)*LOT_SIZE*SCALE), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE*SCALE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE*SCALE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE*SCALE), 'eod', r2(ep), r2(ps[-1]), ts[-1] if len(ts) else EOD_EXIT


def detect_ib_failure(tks, ib_h, ib_l, spot_open):
    buf = spot_open * BRK_BUF
    h_broken = False; l_broken = False
    for _, row in tks[(tks['time']>=ENTRY_START)&(tks['time']<=ENTRY_END)].iterrows():
        p=row['price']; t=row['time']
        if p > ib_h+buf: h_broken=True
        if p < ib_l-buf: l_broken=True
        if h_broken and p<=ib_h:
            h,m,s=map(int,t.split(':'))
            return 'CE', f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
        if l_broken and p>=ib_l:
            h,m,s=map(int,t.split(':'))
            return 'PE', f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
    return None, None


# ── Load day classification from 113 output ───────────────────────────────────
print("Loading day classification from 113...")
df_all = pd.read_excel(f'{OUT_DIR}/113_full_day_analysis.xlsx', sheet_name='all_days',
                       dtype={'date': str})
df_all['date'] = df_all['date'].astype(str)
day_map = {row['date']: row for _, row in df_all.iterrows()}
print(f"  {len(df_all)} days loaded")

# ── Load base trades ──────────────────────────────────────────────────────────
base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base['date_str'] = pd.to_datetime(base['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
base = base.rename(columns={'pnl_conv': 'pnl'})
base_target = base[base['exit_reason']=='target'].copy()
print(f"  Base trades: {len(base)} | Target exits: {len(base_target)}")

all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]

blank_raw = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = set(blank_raw[blank_raw['is_blank']==True]['date'].astype(str))
blank_rem = pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')
blank_set = crt_blank | set(blank_rem['date'].astype(str))


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK A WORKER — Blank day IB failure with D4-CPR quality filter
# ═══════════════════════════════════════════════════════════════════════════════
def _worker_a(args):
    date_str, day_meta, blank_set = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set: return []
    if day_meta is None: return []

    cpr_width = day_meta.get('cpr_width')
    ib_range  = day_meta.get('ib_range')
    ichi_sig  = day_meta.get('ichi_sig')
    ib_class  = day_meta.get('ib_class')
    open_pos  = day_meta.get('open_pos')
    cpr_class = day_meta.get('cpr_class')

    # Quality filter: CPR D4 range + wide IB + Ichimoku directional + open outside CPR
    d4_cpr = (cpr_width is not None and not pd.isna(cpr_width) and
              CPR_D4_LO <= float(cpr_width) <= CPR_D4_HI)
    wide_ib = (ib_class == 'wide')
    ichi_ok = (ichi_sig in ('CE', 'PE'))
    open_outside = (open_pos in ('above_tc', 'below_bc'))

    quality_score = int(d4_cpr) + int(wide_ib) + int(ichi_ok) + int(open_outside)

    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        ib_tks = day[day['time']<=IB_END]
        if ib_tks.empty: return []
        ib_h = ib_tks['price'].max(); ib_l = ib_tks['price'].min()
        ib_rng = ib_h - ib_l; spot_open = ib_tks.iloc[0]['price']
        if ib_rng <= 0: return []

        signal, entry_t = detect_ib_failure(day, ib_h, ib_l, spot_open)
        if signal is None: return []

        # Ichimoku direction agreement
        ichi_agrees = (ichi_sig == signal)

        spot_at = day[day['time']>=entry_t[:8]]
        if spot_at.empty: return []
        spot_ref = spot_at.iloc[0]['price']
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        atm   = get_atm(spot_ref)
        instr = f'NIFTY{expiries[0]}{atm}{signal}'

        res = simulate_sell(date_str, instr, entry_t)
        if res is None: return []
        pnl, reason, ep, xp, xt = res

        return [{
            'track': 'A_blank',
            'date': date_str, 'signal': signal, 'entry_time': entry_t,
            'cpr_width': cpr_width, 'd4_cpr': d4_cpr,
            'wide_ib': wide_ib, 'ichi_ok': ichi_ok, 'ichi_agrees': ichi_agrees,
            'open_outside': open_outside, 'quality_score': quality_score,
            'strike': atm, 'ep': ep, 'xp': xp,
            'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
        }]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK B WORKER — 2nd trade pullback after base TARGET exit
# ═══════════════════════════════════════════════════════════════════════════════
def _worker_b(args):
    trade_row, = args,
    trade_row = args
    date_str  = trade_row['date_str']
    opt       = trade_row['opt']
    exit_time = trade_row['exit_time'] if 'exit_time' in trade_row.index else None
    ep1       = trade_row['entry_price']
    os.environ.pop('MESSAGE_CALLBACK_URL', None)

    # We need to find exit_time from option ticks
    try:
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        spot_all = load_spot_data(date_str, 'NIFTY')
        if spot_all is None: return []
        spot_day = spot_all[(spot_all['time']>='09:15:00')&(spot_all['time']<='15:30:00')]

        spot_at = spot_day[spot_day['time']>=trade_row['entry_time']]
        if spot_at.empty: return []
        spot_ref = spot_at.iloc[0]['price']
        atm1  = get_atm(spot_ref)
        instr = f'NIFTY{expiries[0]}{atm1}{opt}'

        opt_tks = load_tick_data(date_str, instr, trade_row['entry_time'])
        if opt_tks is None or opt_tks.empty: return []
        os_ = opt_tks[opt_tks['time']>=trade_row['entry_time']].reset_index(drop=True)
        if os_.empty: return []

        ps = os_['price'].values; ts = os_['time'].values

        # Find actual exit time (target hit)
        exit_t = None; exit_p = None
        tgt = ep1 * (1 - TGT_PCT)
        hsl = ep1 * (1 + 1.00); sl = hsl; md = 0.0
        for t_, p in zip(ts, ps):
            if t_ >= EOD_EXIT: exit_t = t_; exit_p = p; break
            d = (ep1-p)/ep1
            if d > md: md = d
            if   md >= 0.60: sl = min(sl, ep1*(1-md*0.95))
            elif md >= 0.40: sl = min(sl, ep1*0.80)
            elif md >= 0.25: sl = min(sl, ep1)
            if p <= tgt: exit_t = t_; exit_p = p; break
            if p >= sl:  exit_t = t_; exit_p = p; break
        if exit_t is None: return []
        if t2m(exit_t) > t2m(REENTRY_CUTOFF): return []

        # Now scan from exit_t for pullback: premium bounces to 60-75% of ep1
        pullback_lo = ep1 * PULLBACK_LO
        pullback_hi = ep1 * PULLBACK_HI
        reentry_t = None
        scan = opt_tks[(opt_tks['time']>=exit_t)&(opt_tks['time']<=REENTRY_CUTOFF)]
        for _, row in scan.iterrows():
            p = row['price']; t = row['time']
            if pullback_lo <= p <= pullback_hi:
                h,m,s = map(int,t.split(':'))
                reentry_t = f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
                break

        if reentry_t is None: return []
        if t2m(reentry_t) > t2m(REENTRY_CUTOFF): return []

        # Get fresh ATM at re-entry time
        spot_re = spot_day[spot_day['time']>=reentry_t]
        if spot_re.empty: return []
        atm2  = get_atm(spot_re.iloc[0]['price'])
        instr2= f'NIFTY{expiries[0]}{atm2}{opt}'

        res = simulate_sell(date_str, instr2, reentry_t)
        if res is None: return []
        pnl2, reason2, ep2, xp2, xt2 = res

        return [{
            'track': 'B_2nd',
            'date': date_str, 'opt': opt,
            'first_exit_time': exit_t, 'first_ep': ep1,
            'reentry_time': reentry_t, 'ep': ep2, 'xp': xp2,
            'pnl': pnl2, 'win': pnl2 > 0, 'exit_reason': reason2,
        }]
    except Exception:
        return []


def main():
    # ── TRACK A ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TRACK A: Blank day IB failure with quality filter")
    print("="*70)
    args_a = [(d, day_map.get(d), blank_set) for d in dates_5yr]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_a = pool.map(_worker_a, args_a)
    elapsed_a = (datetime.now()-t0).total_seconds()
    trades_a = pd.DataFrame([t for day in raw_a for t in day])
    print(f"  {len(trades_a)} trades in {elapsed_a:.1f}s")

    def stats(sub, label):
        if sub.empty: return
        wr  = sub['win'].mean()*100
        tp  = sub['pnl'].sum()
        ap  = sub['pnl'].mean()
        print(f"  {label:<35} | {len(sub):>4}t | WR {wr:>5.1f}% | "
              f"Rs.{tp:>9,.0f} | Avg Rs.{ap:>5,.0f}")

    if not trades_a.empty:
        stats(trades_a,                                    'All IB blank (no filter)')
        stats(trades_a[trades_a['quality_score']>=1],      'Quality >= 1')
        stats(trades_a[trades_a['quality_score']>=2],      'Quality >= 2')
        stats(trades_a[trades_a['quality_score']>=3],      'Quality >= 3')
        stats(trades_a[trades_a['quality_score']>=4],      'Quality = 4 (all filters)')
        print(f"  ---")
        stats(trades_a[trades_a['d4_cpr']],                'D4 CPR (25-55pts)')
        stats(trades_a[trades_a['wide_ib']],               'Wide IB')
        stats(trades_a[trades_a['ichi_agrees']],           'Ichimoku agrees')
        stats(trades_a[trades_a['d4_cpr']&trades_a['wide_ib']], 'D4 CPR + Wide IB')
        stats(trades_a[trades_a['d4_cpr']&trades_a['ichi_agrees']], 'D4 CPR + Ichimoku')
        stats(trades_a[trades_a['d4_cpr']&trades_a['wide_ib']&trades_a['ichi_agrees']],
              'D4 CPR + Wide IB + Ichi')

        trades_a['year'] = trades_a['date'].str[:4]
        best_a = trades_a[trades_a['d4_cpr'] & trades_a['wide_ib'] & trades_a['ichi_agrees']]
        if len(best_a):
            print(f"\n  Year breakdown (D4+WideIB+Ichi):")
            for yr, g in best_a.groupby('year'):
                print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                      f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── TRACK B ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TRACK B: 2nd trade pullback after base TARGET exit")
    print("="*70)
    print(f"  Pullback zone: {int(PULLBACK_LO*100)}-{int(PULLBACK_HI*100)}% of entry price")
    print(f"  Re-entry cutoff: {REENTRY_CUTOFF}")

    args_b = [row for _, row in base_target.iterrows()]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_b = pool.map(_worker_b, args_b)
    elapsed_b = (datetime.now()-t0).total_seconds()
    trades_b = pd.DataFrame([t for day in raw_b for t in day])
    print(f"  {len(trades_b)} 2nd trades found (from {len(base_target)} target exits) in {elapsed_b:.1f}s")

    if not trades_b.empty:
        stats(trades_b, 'All 2nd trades')
        stats(trades_b[trades_b['opt']=='CE'], 'CE 2nd trades')
        stats(trades_b[trades_b['opt']=='PE'], 'PE 2nd trades')
        trades_b['year'] = trades_b['date'].str[:4]
        print(f"\n  Year breakdown (2nd trades):")
        for yr, g in trades_b.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")
        exits = trades_b['exit_reason'].value_counts()
        print(f"\n  Exit breakdown (2nd trades):")
        for r, c in exits.items():
            print(f"    {r}: {c} ({round(c/len(trades_b)*100,1)}%)")

    # ── Combined picture ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("COMBINED PICTURE")
    print("="*70)
    base_total = base['pnl'].sum()
    print(f"  Base:                       Rs.{base_total:>12,.0f}  ({len(base)}t | 71.4% WR)")

    best_blank = trades_a[trades_a['d4_cpr']&trades_a['wide_ib']&trades_a['ichi_agrees']] if not trades_a.empty else pd.DataFrame()
    blank_total = best_blank['pnl'].sum() if not best_blank.empty else 0
    print(f"  Blank (D4+WideIB+Ichi):     Rs.{blank_total:>12,.0f}  ({len(best_blank)}t)")

    b2_total = trades_b['pnl'].sum() if not trades_b.empty else 0
    print(f"  2nd trade (pullback):        Rs.{b2_total:>12,.0f}  ({len(trades_b)}t)")

    grand = base_total + blank_total + b2_total
    print(f"  {'─'*45}")
    print(f"  GRAND TOTAL:                Rs.{grand:>12,.0f}")
    print(f"  Improvement vs base:        Rs.{grand-base_total:>12,.0f}  "
          f"({(grand/base_total-1)*100:+.1f}%)")

    # Save
    if not trades_a.empty: trades_a.to_csv(f'{OUT_DIR}/114_blank_quality_trades.csv', index=False)
    if not trades_b.empty: trades_b.to_csv(f'{OUT_DIR}/114_second_trades.csv', index=False)

    # ── Equity chart ──────────────────────────────────────────────────────────
    base_daily = base.copy()
    base_daily['date_dt'] = pd.to_datetime(base_daily['date'].astype(str), format='mixed')
    base_daily = base_daily.groupby('date_dt')['pnl'].sum().reset_index()
    base_daily.columns = ['date','base_pnl']

    def daily_pnl(df_, col):
        if df_.empty: return pd.DataFrame(columns=['date', col])
        df_ = df_.copy()
        df_['date_dt'] = pd.to_datetime(df_['date'].astype(str), format='%Y%m%d')
        g = df_.groupby('date_dt')['pnl'].sum().reset_index()
        g.columns = ['date', col]
        return g

    d_blank = daily_pnl(best_blank, 'blank_pnl')
    d_b2    = daily_pnl(trades_b,   'b2_pnl')

    all_dt = pd.DataFrame({'date': sorted(
        set(base_daily['date']) | set(d_blank['date']) | set(d_b2['date']))})
    m = all_dt.merge(base_daily,on='date',how='left')\
               .merge(d_blank,on='date',how='left')\
               .merge(d_b2,on='date',how='left').fillna(0)
    m['base_eq']  = m['base_pnl'].cumsum()
    m['comb_pnl'] = m['base_pnl'] + m['blank_pnl'] + m['b2_pnl']
    m['comb_eq']  = m['comb_pnl'].cumsum()
    dd = m['comb_eq'] - m['comb_eq'].cummax()

    def eq_pts(s, dates):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v),2)}
                for d,v in zip(dates,s) if pd.notna(v)]

    tv_json = {
        "isTvFormat": False, "candlestick": [], "volume": [],
        "lines": [
            {"id":"combined","label":f"Combined Rs.{int(m['comb_eq'].iloc[-1]):,.0f}",
             "color":"#26a69a","seriesType":"line","data":eq_pts(m['comb_eq'],m['date'])},
            {"id":"base","label":f"Base Rs.{int(base_total):,.0f}",
             "color":"#0ea5e9","seriesType":"line","data":eq_pts(m['base_eq'],m['date'])},
            {"id":"dd","label":f"DD Rs.{int(dd.min()):,.0f}","color":"#ef5350",
             "seriesType":"baseline","baseValue":0,"isNewPane":True,"data":eq_pts(dd,m['date'])},
        ]
    }
    send_custom_chart("114_quality_expansion", tv_json,
        title=f"Quality Expansion | Base Rs.{int(base_total):,.0f} → "
              f"Combined Rs.{int(m['comb_eq'].iloc[-1]):,.0f} "
              f"(Blank+{int(blank_total):,.0f} | 2nd+{int(b2_total):,.0f})")
    print("\nDone.")

if __name__ == '__main__':
    main()
