"""
116_final_all.py — Comprehensive strategy expansion + final combined equity
===========================================================================
Tests every identified opportunity:

  BASE IMPROVEMENT
    F1: Skip PE sells when open>TC but IB expands DOWN (detectable at 9:45)
        → removes normal_down disaster trades

  NEW BLANK DAY STRATEGIES (run in parallel workers)
    S1: O=H / O=L  — open = IB extreme → strong directional bias
        open = IB high (price opened and immediately sold) → sell CE
        open = IB low  (price opened and immediately rallied) → sell PE
        Entry: 9:46:02 (after IB complete)

    S2: VWAP mean reversion
        VWAP from 9:15 (tick-count weighted)
        Price extends >0.8% from VWAP then returns to within 0.2% → sell against extended direction
        Entry: price re-touches VWAP (10:00–13:00)

    S3: Trend IB confirmation
        Open outside CPR + IB expands in SAME direction as open bias
        → trend day confirmed at 9:45 → sell option in trend direction
        Entry: 9:46:02

    S4: 2nd trade pullback (from script 114 — already validated)
        After base TARGET exit, option bounces to 60-75% of entry → re-sell
        Entry cutoff: 13:30

  COMBINED FINAL
    F1 improvement + best blank day strategy + S4 2nd trade
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
from plot_util import send_custom_chart, plot_equity

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
IB_END     = '09:45:00'
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
BRK_BUF    = 0.0005
OHL_TOL    = 0.0012   # 0.12% tolerance for open=IB_high/low
VWAP_EXT   = 0.008    # 0.8% extension threshold
VWAP_NEAR  = 0.002    # 0.2% "near VWAP" for re-entry
VWAP_START = '10:00:00'
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


def detect_s1_ohl(day, ib_h, ib_l, spot_open):
    """S1: open=IB_high (bearish) or open=IB_low (bullish)."""
    if abs(spot_open - ib_h) / spot_open <= OHL_TOL:
        return 'CE', '09:46:02'
    if abs(spot_open - ib_l) / spot_open <= OHL_TOL:
        return 'PE', '09:46:02'
    return None, None


def detect_s2_vwap(day, entry_start=VWAP_START, entry_end='13:00:00'):
    """S2: VWAP mean reversion — first touch of VWAP after extension."""
    scan = day[(day['time'] >= '09:15:00') & (day['time'] <= '15:30:00')].copy()
    scan = scan.reset_index(drop=True)
    prices = scan['price'].values
    times  = scan['time'].values
    vwap   = np.cumsum(prices) / (np.arange(len(prices)) + 1)  # tick-weighted mean

    ext_up = False; ext_dn = False
    for i in range(len(times)):
        t = times[i]; p = prices[i]; v = vwap[i]
        dev = (p - v) / v
        if t < entry_start: continue
        if t > entry_end:   break
        if dev >  VWAP_EXT: ext_up = True
        if dev < -VWAP_EXT: ext_dn = True
        if ext_up and abs(dev) <= VWAP_NEAR:
            h,m,s = map(int,t.split(':'))
            return 'CE', f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
        if ext_dn and abs(dev) <= VWAP_NEAR:
            h,m,s = map(int,t.split(':'))
            return 'PE', f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
    return None, None


def detect_s3_trend_ib(spot_open, ib_h, ib_l, tc, bc, ib_exp_up, ib_exp_down):
    """S3: Open outside CPR + IB expands in bias direction → trend confirmed."""
    if tc is None or bc is None: return None, None
    bullish_open = spot_open > tc
    bearish_open = spot_open < bc
    if bullish_open and ib_exp_up and not ib_exp_down:
        return 'PE', '09:46:02'
    if bearish_open and ib_exp_down and not ib_exp_up:
        return 'CE', '09:46:02'
    return None, None


def _worker_blank(args):
    date_str, meta, blank_set = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set: return []
    tc  = meta.get('tc'); bc = meta.get('bc')

    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        ib    = day[day['time'] <= IB_END]
        if ib.empty: return []
        ib_h  = ib['price'].max(); ib_l = ib['price'].min()
        spot_open = ib.iloc[0]['price']
        if ib_h <= ib_l: return []

        ib_exp_up   = day[day['time'] > IB_END]['price'].max() > ib_h
        ib_exp_down = day[day['time'] > IB_END]['price'].min() < ib_l

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        results = []
        for strat, (sig, et) in [
            ('S1_OHL',    detect_s1_ohl(day, ib_h, ib_l, spot_open)),
            ('S2_VWAP',   detect_s2_vwap(day)),
            ('S3_TrendIB',detect_s3_trend_ib(spot_open, ib_h, ib_l, tc, bc, ib_exp_up, ib_exp_down)),
        ]:
            if sig is None: continue
            spot_at = day[day['time'] >= et[:8]]
            if spot_at.empty: continue
            atm   = get_atm(spot_at.iloc[0]['price'])
            instr = f'NIFTY{expiries[0]}{atm}{sig}'
            res   = simulate_sell(date_str, instr, et)
            if res is None: continue
            pnl, reason, ep, xp = res
            results.append({
                'strategy': strat, 'date': date_str, 'signal': sig,
                'entry_time': et, 'strike': atm, 'ep': ep, 'xp': xp,
                'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
            })
        return results
    except Exception:
        return []


def _worker_s4(trade_row):
    """S4: 2nd trade pullback after base TARGET exit."""
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


def main():
    # ── Load base trades ───────────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str),format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv':'pnl'})

    # Load day behavior for F1 filter
    beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                        sheet_name='all_days_observations', dtype={'date':str})
    beh['date'] = beh['date'].astype(str)
    beh_map = {row['date']: row for _, row in beh.iterrows()}

    # ── F1: Base filter — skip PE when open>TC + IB expands DOWN ──────────────
    print("F1: Base filter analysis...")
    base_f1 = base.copy()
    filtered_out = []
    for _, row in base_f1.iterrows():
        d = row['date_str']
        b = beh_map.get(d)
        if b is not None:
            if (row['opt'] == 'PE' and
                str(b.get('open_pos','')) == 'above_tc' and
                str(b.get('ib_expand','')) == 'down'):
                filtered_out.append(d)
    print(f"  Trades filtered by F1: {len(filtered_out)}")
    base_kept = base_f1[~base_f1['date_str'].isin(filtered_out)]
    base_removed = base_f1[base_f1['date_str'].isin(filtered_out)]

    print(f"  Removed: {len(base_removed)}t | WR {base_removed['win'].mean()*100:.1f}% | "
          f"Avg Rs.{base_removed['pnl'].mean():.0f} | Total Rs.{base_removed['pnl'].sum():,.0f}")
    print(f"  Kept:    {len(base_kept)}t | WR {base_kept['win'].mean()*100:.1f}% | "
          f"Avg Rs.{base_kept['pnl'].mean():.0f} | Total Rs.{base_kept['pnl'].sum():,.0f}")

    # ── Build metadata ─────────────────────────────────────────────────────────
    meta_map = {row['date']: {k: (None if pd.isna(v) else v)
                              for k, v in row.items()
                              if k in ('tc','bc','pvt','r1','s1','ichi_sig','cpr_class','ib_class')}
                for _, row in beh.iterrows()}

    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]
    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    blank_set = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str)) | \
                set(pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')['date'].astype(str))

    # ── Blank day strategies S1/S2/S3 ─────────────────────────────────────────
    print("\nS1/S2/S3: Blank day strategies (parallel)...")
    args_blank = [(d, meta_map.get(d,{}), blank_set) for d in dates_5yr]
    t0 = datetime.now()
    with Pool(processes=min(16,cpu_count() or 4)) as pool:
        raw_blank = pool.map(_worker_blank, args_blank)
    el = (datetime.now()-t0).total_seconds()
    df_blank = pd.DataFrame([t for day in raw_blank for t in day])
    print(f"  {len(df_blank)} trades in {el:.1f}s")

    def stats(sub, label):
        if sub.empty: return
        wr = sub['win'].mean()*100; tp = sub['pnl'].sum(); ap = sub['pnl'].mean()
        print(f"  {label:<32} | {len(sub):>4}t | WR {wr:>5.1f}% | Rs.{tp:>9,.0f} | Avg Rs.{ap:>6,.0f}")

    if not df_blank.empty:
        print(f"\n  {'Strategy':<32} | {'T':>4} | {'WR':>7} | {'Total P&L':>11} | {'Avg':>9}")
        print(f"  {'-'*72}")
        for strat, g in df_blank.groupby('strategy'):
            stats(g, strat)
            for sig, gg in g.groupby('signal'):
                stats(gg, f"  {strat} {sig}")
        print(f"  {'-'*72}")
        stats(df_blank, 'All blank strategies combined')

        # Year breakdown per strategy
        df_blank['year'] = df_blank['date'].str[:4]
        print(f"\n  Year breakdown by strategy:")
        for strat, g in df_blank.groupby('strategy'):
            print(f"  {strat}:")
            for yr, gg in g.groupby('year'):
                print(f"    {yr}: {len(gg)}t | WR {gg['win'].mean()*100:.1f}% | "
                      f"Rs.{gg['pnl'].sum():,.0f} | Avg Rs.{gg['pnl'].mean():.0f}")

    # ── S4: 2nd trade pullback ─────────────────────────────────────────────────
    print("\nS4: 2nd trade pullback...")
    base_target = base[base['exit_reason']=='target']
    args_s4 = [row for _, row in base_target.iterrows()]
    t0 = datetime.now()
    with Pool(processes=min(16,cpu_count() or 4)) as pool:
        raw_s4 = pool.map(_worker_s4, args_s4)
    el = (datetime.now()-t0).total_seconds()
    df_s4 = pd.DataFrame([t for day in raw_s4 for t in day])
    print(f"  {len(df_s4)} 2nd trades in {el:.1f}s")
    if not df_s4.empty:
        stats(df_s4, 'S4 2nd trade (all)')
        df_s4['year'] = df_s4['date'].str[:4]
        for yr, g in df_s4.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── Best blank strategy selection ─────────────────────────────────────────
    best_blank = pd.DataFrame()
    best_blank_name = ''
    if not df_blank.empty:
        # Pick strategy with best total P&L and WR > 60%
        strat_scores = df_blank.groupby('strategy').apply(lambda g: pd.Series({
            'total': g['pnl'].sum(), 'wr': g['win'].mean(), 'n': len(g)
        })).reset_index()
        strat_scores = strat_scores[strat_scores['wr'] >= 0.60]
        if not strat_scores.empty:
            best_row = strat_scores.loc[strat_scores['total'].idxmax()]
            best_blank_name = best_row['strategy']
            best_blank = df_blank[df_blank['strategy'] == best_blank_name]
            print(f"\n  Best blank strategy selected: {best_blank_name}")

    # ── COMBINED FINAL ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMBINED PICTURE")
    print(f"{'='*70}")

    base_orig  = base['pnl'].sum()
    base_f1v   = base_kept['pnl'].sum()
    f1_gain    = base_f1v - base_orig
    blank_gain = best_blank['pnl'].sum() if not best_blank.empty else 0
    s4_gain    = df_s4['pnl'].sum() if not df_s4.empty else 0

    print(f"  Base (original):           Rs.{base_orig:>12,.0f}  {len(base)}t")
    print(f"  F1 filter improvement:     Rs.{f1_gain:>+12,.0f}  (removed {len(base_removed)}t losing)")
    print(f"  Base after F1:             Rs.{base_f1v:>12,.0f}  {len(base_kept)}t | "
          f"WR {base_kept['win'].mean()*100:.1f}%")
    print(f"  + Blank ({best_blank_name}):  Rs.{blank_gain:>+12,.0f}  "
          f"{len(best_blank)}t")
    print(f"  + S4 2nd trade:            Rs.{s4_gain:>+12,.0f}  "
          f"{len(df_s4)}t")
    grand = base_f1v + blank_gain + s4_gain
    print(f"  {'─'*45}")
    print(f"  GRAND TOTAL:               Rs.{grand:>12,.0f}")
    print(f"  vs Base original:          Rs.{grand-base_orig:>+12,.0f}  "
          f"({(grand/base_orig-1)*100:+.1f}%)")

    # ── Equity curve ──────────────────────────────────────────────────────────
    def to_daily(df_, date_col, pnl_col='pnl'):
        if df_.empty: return pd.Series(dtype=float)
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_[date_col].astype(str), format='%Y%m%d')
        return df_.groupby('dt')[pnl_col].sum()

    base_d   = base.copy(); base_d['dt'] = pd.to_datetime(base_d['date'].astype(str), format='mixed')
    base_daily_orig = base_d.groupby('dt')['pnl'].sum()

    base_kept_d = base_kept.copy()
    base_kept_d['dt'] = pd.to_datetime(base_kept_d['date'].astype(str), format='mixed')
    base_daily_f1 = base_kept_d.groupby('dt')['pnl'].sum()

    blank_d = to_daily(best_blank, 'date') if not best_blank.empty else pd.Series(dtype=float)
    s4_d    = to_daily(df_s4,      'date') if not df_s4.empty else pd.Series(dtype=float)

    all_idx = pd.date_range(
        start=min(base_daily_orig.index.min(), base_daily_f1.index.min()),
        end=max(base_daily_orig.index.max(),
                blank_d.index.max() if not blank_d.empty else base_daily_orig.index.max(),
                s4_d.index.max()    if not s4_d.empty    else base_daily_orig.index.max()),
        freq='B'
    )
    m = pd.DataFrame(index=all_idx)
    m['base_orig'] = base_daily_orig.reindex(all_idx, fill_value=0)
    m['base_f1']   = base_daily_f1.reindex(all_idx, fill_value=0)
    m['blank']     = blank_d.reindex(all_idx, fill_value=0)
    m['s4']        = s4_d.reindex(all_idx, fill_value=0)
    m['combined']  = m['base_f1'] + m['blank'] + m['s4']

    eq_orig = m['base_orig'].cumsum()
    eq_f1   = m['base_f1'].cumsum()
    eq_comb = m['combined'].cumsum()
    dd_comb = (eq_comb - eq_comb.cummax())
    dd_orig = (eq_orig - eq_orig.cummax())

    comb_total = round(eq_comb.iloc[-1], 0)
    comb_dd    = round(dd_comb.min(), 0)
    orig_dd    = round(dd_orig.min(), 0)
    print(f"\n  Equity stats:")
    print(f"    Base original:  Rs.{int(eq_orig.iloc[-1]):,.0f} | DD Rs.{int(orig_dd):,.0f}")
    print(f"    Combined final: Rs.{int(comb_total):,.0f} | DD Rs.{int(comb_dd):,.0f}")

    def eq_pts(s):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                for d, v in s.items() if pd.notna(v)]

    tv_json = {
        "isTvFormat": False, "candlestick": [], "volume": [],
        "lines": [
            {"id":"combined","label":f"Combined Rs.{int(comb_total):,.0f}",
             "color":"#26a69a","seriesType":"line","data":eq_pts(eq_comb)},
            {"id":"base_f1","label":f"Base+F1 Rs.{int(eq_f1.iloc[-1]):,.0f}",
             "color":"#0ea5e9","seriesType":"line","data":eq_pts(eq_f1)},
            {"id":"base_orig","label":f"Base original Rs.{int(eq_orig.iloc[-1]):,.0f}",
             "color":"#9e9e9e","seriesType":"line","data":eq_pts(eq_orig)},
            {"id":"dd","label":f"Combined DD Rs.{int(comb_dd):,.0f}",
             "color":"#ef5350","seriesType":"baseline","baseValue":0,"isNewPane":True,
             "data":eq_pts(dd_comb)},
        ]
    }
    send_custom_chart("116_final_all", tv_json,
        title=f"Final Combined | Base {int(eq_orig.iloc[-1]):,} → "
              f"Combined {int(comb_total):,} | DD {int(comb_dd):,}")

    # Save all results
    if not df_blank.empty:
        df_blank.to_csv(f'{OUT_DIR}/116_blank_strategies.csv', index=False)
    if not df_s4.empty:
        df_s4.to_csv(f'{OUT_DIR}/116_second_trades.csv', index=False)
    base_removed.to_csv(f'{OUT_DIR}/116_filtered_out.csv', index=False)

    print(f"\n  Saved: 116_blank_strategies.csv | 116_second_trades.csv | 116_filtered_out.csv")
    print("\nDone.")


if __name__ == '__main__':
    main()
