"""
117_s3_fixed.py — Forward-bias-corrected S3 TrendIB + final combined picture
=============================================================================
S3 TrendIB was forward-biased in 116: ib_exp_up/down was computed from the
FULL post-9:45 day, but entry is at 9:46:02.  Only ticks between 9:45:01 and
9:46:01 (the single minute before entry) are visible at signal time.

Fix: restrict IB expansion check to  day[(time > 09:45:00) & (time < 09:46:02)]

F1 filter was also found to be forward-biased — discarded.
S4 2nd trade was already confirmed clean — carried forward unchanged.

Combined (corrected):  Base + S3_fixed + S4
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
S3_ENTRY   = '09:46:02'   # S3 entry time — IB expansion must be confirmed before this
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
PULLBACK_LO= 0.60; PULLBACK_HI = 0.75
REENTRY_CUT= '13:30:00'

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def t2m(t):
    h,m,s = map(int,t.split(':'))
    return h*60+m+s/60


def simulate_sell(date_str, instrument, entry_time):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep*(1-TGT_PCT)); hsl = r2(ep*2.0); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep-p)*LOT_SIZE*SCALE), 'eod', r2(ep), r2(p)
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE*SCALE), 'target', r2(ep), r2(p)
        if p >= sl:  return r2((ep-p)*LOT_SIZE*SCALE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p)
    return r2((ep-ps[-1])*LOT_SIZE*SCALE), 'eod', r2(ep), r2(ps[-1])


def _worker_s3(args):
    """S3 TrendIB — CORRECTED: IB expansion checked only pre-entry (9:45:01–9:46:01)."""
    date_str, meta, blank_set = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set: return []
    tc = meta.get('tc'); bc = meta.get('bc')
    if tc is None or bc is None: return []

    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        ib = day[day['time'] <= IB_END]
        if ib.empty: return []
        ib_h = ib['price'].max(); ib_l = ib['price'].min()
        spot_open = ib.iloc[0]['price']
        if ib_h <= ib_l: return []

        # CORRECTED: only ticks strictly between IB_END and S3_ENTRY
        pre_entry = day[(day['time'] > IB_END) & (day['time'] < S3_ENTRY)]
        if pre_entry.empty:
            return []  # no tick in the 1-minute window → skip
        ib_exp_up   = pre_entry['price'].max() > ib_h
        ib_exp_down = pre_entry['price'].min() < ib_l

        # S3 signal: open outside CPR + IB expands in open-bias direction
        bullish_open = spot_open > tc
        bearish_open = spot_open < bc
        sig = None
        if bullish_open and ib_exp_up and not ib_exp_down:
            sig = 'PE'
        elif bearish_open and ib_exp_down and not ib_exp_up:
            sig = 'CE'
        if sig is None: return []

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        spot_at = day[day['time'] >= S3_ENTRY]
        if spot_at.empty: return []
        atm   = get_atm(spot_at.iloc[0]['price'])
        instr = f'NIFTY{expiries[0]}{atm}{sig}'
        res   = simulate_sell(date_str, instr, S3_ENTRY)
        if res is None: return []
        pnl, reason, ep, xp = res
        return [{'strategy':'S3_TrendIB_fixed', 'date': date_str, 'signal': sig,
                 'entry_time': S3_ENTRY, 'strike': atm, 'ep': ep, 'xp': xp,
                 'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason}]
    except Exception:
        return []


def _worker_s4(trade_row):
    """S4: 2nd trade pullback after base TARGET exit — already validated clean."""
    date_str = trade_row['date_str']
    opt = trade_row['opt']; ep1 = trade_row['entry_price']
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        spot_all = load_spot_data(date_str, 'NIFTY')
        if spot_all is None: return []
        spot_day = spot_all[(spot_all['time']>='09:15:00')&(spot_all['time']<='15:30:00')]
        spot_at  = spot_day[spot_day['time']>=trade_row['entry_time']]
        if spot_at.empty: return []
        atm1  = get_atm(spot_at.iloc[0]['price'])
        instr = f'NIFTY{expiries[0]}{atm1}{opt}'
        opt_tks = load_tick_data(date_str, instr, trade_row['entry_time'])
        if opt_tks is None or opt_tks.empty: return []
        os_ = opt_tks[opt_tks['time']>=trade_row['entry_time']].reset_index(drop=True)
        if os_.empty: return []
        ps = os_['price'].values; ts = os_['time'].values

        tgt = ep1*(1-TGT_PCT); hsl = ep1*2.0; sl = hsl; md = 0.0
        exit_t = None
        for t_, p in zip(ts, ps):
            if t_ >= EOD_EXIT: exit_t = t_; break
            d = (ep1-p)/ep1
            if d > md: md = d
            if   md >= 0.60: sl = min(sl, ep1*(1-md*0.95))
            elif md >= 0.40: sl = min(sl, ep1*0.80)
            elif md >= 0.25: sl = min(sl, ep1)
            if p <= tgt: exit_t = t_; break
            if p >= sl:  exit_t = t_; break
        if exit_t is None or t2m(exit_t) > t2m(REENTRY_CUT): return []

        pb_lo = ep1*PULLBACK_LO; pb_hi = ep1*PULLBACK_HI
        scan_pb = opt_tks[(opt_tks['time']>=exit_t)&(opt_tks['time']<=REENTRY_CUT)]
        reentry_t = None
        for _, row in scan_pb.iterrows():
            if pb_lo <= row['price'] <= pb_hi:
                h,m,s = map(int, row['time'].split(':'))
                reentry_t = f'{h:02d}:{m:02d}:{min(s+2,59):02d}'; break
        if reentry_t is None: return []

        spot_re = spot_day[spot_day['time']>=reentry_t]
        if spot_re.empty: return []
        atm2  = get_atm(spot_re.iloc[0]['price'])
        instr2= f'NIFTY{expiries[0]}{atm2}{opt}'
        res = simulate_sell(date_str, instr2, reentry_t)
        if res is None: return []
        pnl, reason, ep2, xp2 = res
        return [{'strategy':'S4_2nd','date':date_str,'opt':opt,
                 'reentry_time':reentry_t,'ep':ep2,'xp':xp2,
                 'pnl':pnl,'win':pnl>0,'exit_reason':reason}]
    except Exception:
        return []


def stats(sub, label):
    if sub.empty: return
    wr = sub['win'].mean()*100; tp = sub['pnl'].sum(); ap = sub['pnl'].mean()
    print(f"  {label:<40} | {len(sub):>4}t | WR {wr:>5.1f}% | Rs.{tp:>10,.0f} | Avg Rs.{ap:>6,.0f}")


def main():
    # ── Load base trades ───────────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str),format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv':'pnl'})

    # ── Load day behavior for metadata (CPR levels) ────────────────────────────
    beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                        sheet_name='all_days_observations', dtype={'date':str})
    beh['date'] = beh['date'].astype(str)
    meta_map = {row['date']: {k: (None if pd.isna(v) else v)
                              for k, v in row.items()
                              if k in ('tc','bc','pvt','r1','s1')}
                for _, row in beh.iterrows()}

    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]
    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    blank_set = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str)) | \
                set(pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')['date'].astype(str))

    # ── S3 corrected ──────────────────────────────────────────────────────────
    print("S3 TrendIB (bias-corrected): IB expansion checked only 9:45:01–9:46:01...")
    args_s3 = [(d, meta_map.get(d,{}), blank_set) for d in dates_5yr]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_s3 = pool.map(_worker_s3, args_s3)
    el = (datetime.now()-t0).total_seconds()
    df_s3 = pd.DataFrame([t for day in raw_s3 for t in day])
    print(f"  {len(df_s3)} trades in {el:.1f}s")

    if not df_s3.empty:
        stats(df_s3, 'S3_TrendIB_fixed (all)')
        for sig, g in df_s3.groupby('signal'):
            stats(g, f"  S3_TrendIB_fixed {sig}")
        df_s3['year'] = df_s3['date'].str[:4]
        print(f"\n  Year breakdown:")
        for yr, g in df_s3.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")
    else:
        print("  No S3 trades after bias correction.")

    # ── S4: 2nd trade pullback (carry forward from 116 — already valid) ────────
    print("\nS4: 2nd trade pullback (bias-clean, re-running for consistency)...")
    base_target = base[base['exit_reason']=='target']
    args_s4 = [row for _, row in base_target.iterrows()]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_s4 = pool.map(_worker_s4, args_s4)
    el = (datetime.now()-t0).total_seconds()
    df_s4 = pd.DataFrame([t for day in raw_s4 for t in day])
    print(f"  {len(df_s4)} 2nd trades in {el:.1f}s")
    if not df_s4.empty:
        stats(df_s4, 'S4 2nd trade (all)')
        for sig, g in df_s4.groupby('opt'):
            stats(g, f"  S4 {sig}")
        df_s4['year'] = df_s4['date'].str[:4]
        print(f"\n  Year breakdown:")
        for yr, g in df_s4.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── Bias comparison: S3 biased vs corrected ───────────────────────────────
    s3_biased_total = 279685   # from 116 run
    s3_biased_count = 210
    s3_fixed_total  = df_s3['pnl'].sum() if not df_s3.empty else 0
    s3_fixed_count  = len(df_s3)

    print(f"\n{'='*70}")
    print(f"  S3 BIAS CORRECTION IMPACT")
    print(f"{'='*70}")
    print(f"  S3 (biased, 116):   {s3_biased_count}t | Rs.{s3_biased_total:,.0f}")
    print(f"  S3 (corrected):     {s3_fixed_count}t | Rs.{s3_fixed_total:,.0f}")
    print(f"  Overstatement:      Rs.{s3_biased_total - s3_fixed_total:,.0f}  "
          f"({s3_biased_count - s3_fixed_count} trades removed)")

    # ── CORRECTED COMBINED PICTURE ─────────────────────────────────────────────
    base_total = base['pnl'].sum()
    s4_total   = df_s4['pnl'].sum() if not df_s4.empty else 0
    grand      = base_total + s3_fixed_total + s4_total

    print(f"\n{'='*70}")
    print(f"  CORRECTED COMBINED PICTURE (no forward bias)")
    print(f"{'='*70}")
    print(f"  Base (validated):          Rs.{base_total:>12,.0f}  {len(base)}t | "
          f"WR {base['win'].mean()*100:.1f}%")
    print(f"  + S3 TrendIB (corrected):  Rs.{s3_fixed_total:>+12,.0f}  {s3_fixed_count}t")
    print(f"  + S4 2nd trade (clean):    Rs.{s4_total:>+12,.0f}  {len(df_s4)}t")
    print(f"  {'─'*47}")
    print(f"  GRAND TOTAL:               Rs.{grand:>12,.0f}")
    print(f"  vs Base:                   Rs.{grand - base_total:>+12,.0f}  "
          f"({(grand/base_total - 1)*100:+.1f}%)")

    # ── 116 claimed vs corrected ───────────────────────────────────────────────
    claimed_116 = 1882189
    print(f"\n  116 claimed (biased):      Rs.{claimed_116:>12,.0f}")
    print(f"  117 corrected:             Rs.{grand:>12,.0f}")
    print(f"  Bias overstatement:        Rs.{claimed_116 - grand:>+12,.0f}")

    # ── Equity curve ──────────────────────────────────────────────────────────
    def to_daily(df_, date_col, pnl_col='pnl'):
        if df_.empty: return pd.Series(dtype=float)
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_[date_col].astype(str), format='%Y%m%d')
        return df_.groupby('dt')[pnl_col].sum()

    base_d = base.copy()
    base_d['dt'] = pd.to_datetime(base_d['date'].astype(str), format='mixed')
    base_daily = base_d.groupby('dt')['pnl'].sum()

    s3_d = to_daily(df_s3, 'date') if not df_s3.empty else pd.Series(dtype=float)
    s4_d = to_daily(df_s4, 'date') if not df_s4.empty else pd.Series(dtype=float)

    all_idx = pd.date_range(
        start=base_daily.index.min(),
        end=max(base_daily.index.max(),
                s3_d.index.max() if not s3_d.empty else base_daily.index.max(),
                s4_d.index.max() if not s4_d.empty else base_daily.index.max()),
        freq='B'
    )
    m = pd.DataFrame(index=all_idx)
    m['base']     = base_daily.reindex(all_idx, fill_value=0)
    m['s3_fixed'] = s3_d.reindex(all_idx, fill_value=0)
    m['s4']       = s4_d.reindex(all_idx, fill_value=0)
    m['combined'] = m['base'] + m['s3_fixed'] + m['s4']

    eq_base = m['base'].cumsum()
    eq_comb = m['combined'].cumsum()
    dd_base = eq_base - eq_base.cummax()
    dd_comb = eq_comb - eq_comb.cummax()

    print(f"\n  Equity stats:")
    print(f"    Base:     Rs.{int(eq_base.iloc[-1]):,.0f} | DD Rs.{int(dd_base.min()):,.0f}")
    print(f"    Combined: Rs.{int(eq_comb.iloc[-1]):,.0f} | DD Rs.{int(dd_comb.min()):,.0f}")

    def eq_pts(s):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                for d, v in s.items() if pd.notna(v)]

    tv_json = {
        "isTvFormat": False, "candlestick": [], "volume": [],
        "lines": [
            {"id":"combined","label":f"Combined (corrected) Rs.{int(eq_comb.iloc[-1]):,.0f}",
             "color":"#26a69a","seriesType":"line","data":eq_pts(eq_comb)},
            {"id":"base","label":f"Base Rs.{int(eq_base.iloc[-1]):,.0f}",
             "color":"#9e9e9e","seriesType":"line","data":eq_pts(eq_base)},
            {"id":"dd_comb","label":f"Combined DD Rs.{int(dd_comb.min()):,.0f}",
             "color":"#ef5350","seriesType":"baseline","baseValue":0,"isNewPane":True,
             "data":eq_pts(dd_comb)},
            {"id":"dd_base","label":f"Base DD Rs.{int(dd_base.min()):,.0f}",
             "color":"#ff9800","seriesType":"baseline","baseValue":0,"isNewPane":True,
             "data":eq_pts(dd_base)},
        ]
    }
    send_custom_chart("117_corrected", tv_json,
        title=f"117 Corrected | Base {int(eq_base.iloc[-1]):,} → "
              f"Combined {int(eq_comb.iloc[-1]):,} | DD {int(dd_comb.min()):,}")

    # Save results
    if not df_s3.empty:
        df_s3.to_csv(f'{OUT_DIR}/117_s3_fixed.csv', index=False)
    if not df_s4.empty:
        df_s4.to_csv(f'{OUT_DIR}/117_s4_clean.csv', index=False)
    print(f"\n  Saved: 117_s3_fixed.csv | 117_s4_clean.csv")
    print("\nDone.")


if __name__ == '__main__':
    main()
