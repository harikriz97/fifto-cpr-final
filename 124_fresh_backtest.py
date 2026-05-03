"""
124_fresh_backtest.py — Clean fresh combined backtest (all bugs fixed)
=======================================================================
Audit findings fixed:
  BUG 1: 15 iv2 trades overlapping v17a/cam on same day (mixed date formats
          caused 75_live_simulation.csv to show 550 unique but only 535 real)
          FIX: load 72_final_trades (480, 1/day) + 55 clean iv2 non-overlapping

  BUG 2: S4 had 2 trades on 20250401 + 20260109 because both iv2 & cam hit
          target on those days, both went into S4 pool
          FIX: rebuild S4 from clean base (1 trade/day), dedup after

  BUG 3: LOT_SIZE=75, SCALE=65/75 roundabout hack across all new scripts
          FIX: LOT_SIZE=65 directly everywhere in this script

Clean sources:
  A: 72_final_trades.csv  — 480 v17a/cam trades, 1/day, conviction lots (1-3)
  B: iv2 clean            — 55 non-overlapping iv2 trades, 1 lot
  C: S4 2nd trade         — re-run from clean base, 1 trade/day cap
  D: Blank day P1-P7      — from 123 output (already 1/day, no base overlap)
"""
import sys, os, warnings
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

OUT_DIR     = 'data/20260430'
LOT_SIZE    = 65           # FIXED: no more 75*scale hack
STRIKE_INT  = 50
IB_END      = '09:45:00'
ENTRY_LATE  = '09:46:02'
EOD_EXIT    = '15:20:00'
TGT_PCT     = 0.30
PULLBACK_LO = 0.60
PULLBACK_HI = 0.75
REENTRY_CUT = '13:30:00'
OHL_TOL     = 0.0015
MONTH_MAP   = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
               7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}


def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def t2m(t):
    h, m, s = map(int, str(t).split(':'))
    return h * 60 + m + s / 60


def simulate_sell(date_str, instr, entry_time):
    """Tick-level sell simulation. LOT_SIZE=65 fixed. Returns (pnl, reason, ep, xp) or None."""
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - TGT_PCT))
    hsl = r2(ep * 2.0)      # hard SL: option price doubles (100% loss on premium)
    sl  = hsl
    md  = 0.0
    ps  = tks['price'].values
    ts  = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE), 'eod', r2(ep), r2(p)
        d = (ep - p) / ep      # positive = price fell (favorable for seller)
        if d > md: md = d
        # 3-tier trailing SL (ratchets DOWN as price falls → locks profit)
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE), 'target',     r2(ep), r2(p)
        if p >= sl:  return r2((ep - p) * LOT_SIZE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p)
    return r2((ep - ps[-1]) * LOT_SIZE), 'eod', r2(ep), r2(ps[-1])


# ─────────────────────────────────────────────────────────────────────────────
# S4: 2nd trade pullback worker (clean — 1 per base trade)
# ─────────────────────────────────────────────────────────────────────────────
def _worker_s4(args):
    date_str, opt, ep1_raw, entry_time = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    ep1 = float(ep1_raw)
    try:
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return None
        spot_all = load_spot_data(date_str, 'NIFTY')
        if spot_all is None: return None
        spot_day = spot_all[(spot_all['time'] >= '09:15:00') & (spot_all['time'] <= '15:30:00')]

        # Re-simulate the base trade to find its exit time
        spot_at = spot_day[spot_day['time'] >= entry_time]
        if spot_at.empty: return None
        atm1  = get_atm(spot_at.iloc[0]['price'])
        instr = f'NIFTY{expiries[0]}{atm1}{opt}'
        opt_tks = load_tick_data(date_str, instr, entry_time)
        if opt_tks is None or opt_tks.empty: return None
        os_ = opt_tks[opt_tks['time'] >= entry_time].reset_index(drop=True)
        if os_.empty: return None

        # Find first exit
        tgt = ep1 * (1 - TGT_PCT); hsl = ep1 * 2.0; sl = hsl; md = 0.0
        exit_t = None
        for _, row in os_.iterrows():
            t_, p = row['time'], row['price']
            if t_ >= EOD_EXIT: exit_t = t_; break
            d = (ep1 - p) / ep1
            if d > md: md = d
            if   md >= 0.60: sl = min(sl, ep1 * (1 - md * 0.95))
            elif md >= 0.40: sl = min(sl, ep1 * 0.80)
            elif md >= 0.25: sl = min(sl, ep1)
            if p <= tgt: exit_t = t_; break
            if p >= sl:  exit_t = t_; break
        if exit_t is None or t2m(exit_t) > t2m(REENTRY_CUT):
            return None

        # Scan for pullback (60-75% of ep1) after exit
        pb_lo = ep1 * PULLBACK_LO; pb_hi = ep1 * PULLBACK_HI
        scan = opt_tks[(opt_tks['time'] >= exit_t) & (opt_tks['time'] <= REENTRY_CUT)]
        reentry_t = None
        for _, row in scan.iterrows():
            if pb_lo <= row['price'] <= pb_hi:
                h, m, s = map(int, row['time'].split(':'))
                reentry_t = f'{h:02d}:{m:02d}:{min(s + 2, 59):02d}'
                break
        if reentry_t is None: return None

        # Simulate 2nd trade
        spot_re = spot_day[spot_day['time'] >= reentry_t]
        if spot_re.empty: return None
        atm2   = get_atm(spot_re.iloc[0]['price'])
        instr2 = f'NIFTY{expiries[0]}{atm2}{opt}'
        res = simulate_sell(date_str, instr2, reentry_t)
        if res is None: return None
        pnl, reason, ep2, xp2 = res
        return {'strategy': 'S4_2nd', 'date': date_str, 'opt': opt,
                'reentry_time': reentry_t, 'ep': ep2, 'xp': xp2,
                'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason}
    except Exception:
        return None


def stats(df_, label):
    if df_ is None or len(df_) == 0: return
    df_ = pd.DataFrame(df_) if not isinstance(df_, pd.DataFrame) else df_
    wr  = df_['win'].mean() * 100
    tp  = df_['pnl'].sum()
    ap  = df_['pnl'].mean()
    losses = df_[~df_['win']]['pnl'].sum()
    wins   = df_[df_['win']]['pnl'].sum()
    pf = wins / abs(losses) if losses != 0 else 99.0
    print(f"  {label:<40} | {len(df_):>4}t | WR {wr:>5.1f}% | "
          f"Rs.{tp:>12,.0f} | Avg Rs.{ap:>7,.0f} | PF {pf:.2f}")


def main():
    print("=" * 75)
    print("  124 — FRESH CLEAN BACKTEST (all bugs fixed)")
    print("=" * 75)

    # ── A: Load 72_final_trades (v17a + cam, 480 trades, 1/day) ──────────────
    raw72 = pd.read_csv(f'{OUT_DIR}/72_final_trades.csv')
    raw72['date'] = pd.to_datetime(raw72['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
    assert raw72['date'].nunique() == len(raw72), "72_final_trades has duplicate dates!"

    base_v17a = pd.DataFrame({
        'date':        raw72['date'],
        'strategy':    raw72['strategy'],
        'zone':        raw72['zone'],
        'opt':         raw72['opt'],
        'entry_time':  raw72['entry_time'],
        'ep':          raw72['ep'].round(2),
        'xp':          raw72['xp'].round(2),
        'exit_reason': raw72['exit_reason'],
        'pnl_1lot':    (raw72['pnl_65']).round(2),          # 1 lot × 65
        'lots':        raw72['lots7n'].astype(int),
        'score':       raw72['score7'].astype(int),
        'pnl':         raw72['pnl_final'].round(2),          # conviction (lots × 65)
        'win':         raw72['win'],
        'year':        raw72['date'].str[:4],
    })

    # ── B: iv2 clean (55 non-overlapping days) ────────────────────────────────
    raw75 = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    raw75['date'] = pd.to_datetime(raw75['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
    iv2_all = raw75[raw75['strategy'].str.startswith('iv2')].copy()
    iv2_clean = iv2_all[~iv2_all['date'].isin(set(base_v17a['date']))].copy()

    base_iv2 = pd.DataFrame({
        'date':        iv2_clean['date'],
        'strategy':    iv2_clean['strategy'],
        'zone':        iv2_clean['zone'],
        'opt':         iv2_clean['opt'],
        'entry_time':  iv2_clean['entry_time'],
        'ep':          iv2_clean['entry_price'].round(2),
        'xp':          iv2_clean['exit_price'].round(2),
        'exit_reason': iv2_clean['exit_reason'],
        'pnl_1lot':    iv2_clean['pnl_65'].round(2),
        'lots':        1,
        'score':       -1,
        'pnl':         iv2_clean['pnl_65'].round(2),         # 1 lot only
        'win':         iv2_clean['win'],
        'year':        iv2_clean['date'].str[:4],
    })

    base = pd.concat([base_v17a, base_iv2], ignore_index=True)
    base = base.sort_values('date').reset_index(drop=True)

    print(f"\n  Base assembly:")
    print(f"    v17a + cam trades: {len(base_v17a):>4} (480, 1/day, conviction lots)")
    print(f"    iv2 clean trades:  {len(base_iv2):>4} (non-overlapping days only)")
    print(f"    TOTAL base:        {len(base):>4} | Unique dates: {base['date'].nunique()}")
    print(f"    [Removed: {len(iv2_all) - len(base_iv2)} iv2 that overlapped with cam/v17a on same day]")
    print()
    stats(base, 'A: Base (v17a+cam+iv2 clean)')
    for strat, g in base.groupby('strategy'):
        stats(g, f"    {strat}")

    # ── Verify no duplicate base dates ────────────────────────────────────────
    dups = base.groupby('date').size()
    dup_dates = dups[dups > 1]
    if len(dup_dates) > 0:
        print(f"\n  WARNING: {len(dup_dates)} duplicate dates in base!")
        print(dup_dates)
    else:
        print(f"\n  Sequential check: PASS — 1 trade per day in base")

    # ── C: S4 2nd trade (re-run from clean base) ─────────────────────────────
    print(f"\n  Running S4 2nd trade from clean base (target exits only)...")
    base_target = base[base['exit_reason'] == 'target'].copy()
    args_s4 = [(r['date'], r['opt'], r['ep'], r['entry_time'])
               for _, r in base_target.iterrows()]

    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_s4 = pool.map(_worker_s4, args_s4)
    el = (datetime.now() - t0).total_seconds()

    s4_all = pd.DataFrame([r for r in raw_s4 if r is not None])
    print(f"    {len(s4_all)} S4 trades in {el:.1f}s")

    # Dedup: keep only 1 S4 per day (earliest reentry_time)
    if not s4_all.empty:
        s4_all = s4_all.sort_values('reentry_time')
        s4_all = s4_all.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
        s4_all['year'] = s4_all['date'].str[:4]
        print(f"    After dedup: {len(s4_all)} S4 trades | {s4_all['date'].nunique()} unique days")
        stats(s4_all, 'C: S4 2nd trade (clean, 1/day)')
        for sig, g in s4_all.groupby('opt'):
            stats(g, f"    S4 {sig}")
    else:
        print("    No S4 trades.")

    # ── D: Blank day P1-P7 (load from 123 output — already 1/day, no overlap) ─
    blank = pd.read_csv(f'{OUT_DIR}/123_blank_all_strategies.csv')
    blank['date'] = blank['date'].astype(str)

    # Quality filter: keep strategies with WR >= 60% AND avg P&L > 0
    keep_strats = []
    for strat, g in blank.groupby('strategy'):
        if g['win'].mean() >= 0.60 and g['pnl'].mean() > 0:
            keep_strats.append(strat)
    blank_kept = blank[blank['strategy'].isin(keep_strats)].copy()

    print(f"\n  D: Blank day P1-P7 (quality-filtered):")
    print(f"    All strategies: {len(blank)} trades | {blank['date'].nunique()} days")
    print(f"    Kept strategies (WR>=60%, avg>0): {keep_strats}")
    print(f"    Kept trades: {len(blank_kept)} | {blank_kept['date'].nunique()} days")
    stats(blank, 'D_all: All blank strategies')
    stats(blank_kept, 'D_kept: Quality-filtered blank')
    for strat, g in blank_kept.groupby('strategy'):
        stats(g, f"    {strat}")

    # ── Coverage analysis ─────────────────────────────────────────────────────
    all_dates = list_trading_dates()
    latest = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]
    total_days = len(dates_5yr)

    base_d_set  = set(base['date'])
    s4_d_set    = set(s4_all['date']) if not s4_all.empty else set()
    blank_d_set = set(blank_kept['date'])

    # All three are additive (base and blank don't overlap, S4 is same-day extra)
    assert len(base_d_set & blank_d_set) == 0, "Blank days overlap with base!"
    total_covered = len(base_d_set | blank_d_set)

    print(f"\n{'='*75}")
    print(f"  COVERAGE ANALYSIS")
    print(f"{'='*75}")
    print(f"  Base (v17a+cam+iv2 clean):  {len(base_d_set):>4}/{total_days}  ({len(base_d_set)/total_days*100:.1f}%)")
    print(f"  + S4 2nd trade:             {len(s4_d_set):>4} days (same-day extra)")
    print(f"  + Blank day P1-P7 (kept):   {len(blank_d_set):>4}/{total_days}  ({len(blank_d_set)/total_days*100:.1f}%)")
    print(f"  TOTAL DAYS WITH A TRADE:    {total_covered:>4}/{total_days}  ({total_covered/total_days*100:.1f}%)")
    print(f"  Target (75%):               {int(total_days*0.75):>4} days")

    # ── Combined P&L ──────────────────────────────────────────────────────────
    base_total  = base['pnl'].sum()
    s4_total    = s4_all['pnl'].sum() if not s4_all.empty else 0
    blank_total = blank_kept['pnl'].sum()
    grand       = base_total + s4_total + blank_total

    print(f"\n{'='*75}")
    print(f"  FINAL COMBINED SYSTEM")
    print(f"{'='*75}")
    print(f"  A: Base (clean):          Rs.{base_total:>12,.0f}  {len(base)}t  | WR {base['win'].mean()*100:.1f}%")
    print(f"  C: S4 2nd trade (clean):  Rs.{s4_total:>+12,.0f}  {len(s4_all)}t  | WR {s4_all['win'].mean()*100:.1f}%" if not s4_all.empty else "  C: S4: N/A")
    print(f"  D: Blank P1-P7 (kept):   Rs.{blank_total:>+12,.0f}  {len(blank_kept)}t  | WR {blank_kept['win'].mean()*100:.1f}%")
    print(f"  {'─'*55}")
    print(f"  GRAND TOTAL:              Rs.{grand:>12,.0f}  {len(base)+len(s4_all)+len(blank_kept)}t")
    print(f"  Coverage:                 {total_covered/total_days*100:.1f}%  ({total_covered}/{total_days} days)")
    print(f"  vs old base (buggy):      Rs.{grand - 1072048:>+12,.0f}  ({(grand/1072048-1)*100:+.1f}%)")

    # ── Year breakdown ────────────────────────────────────────────────────────
    print(f"\n  Year breakdown:")
    all_df = pd.concat([
        base[['date','year','pnl','win','strategy']],
        (s4_all[['date','year','pnl','win','strategy']] if not s4_all.empty else pd.DataFrame()),
        blank_kept[['date','year','pnl','win','strategy']].rename(columns={'strategy':'strategy'}),
    ], ignore_index=True)
    all_df['year'] = all_df['date'].str[:4]
    for yr, g in all_df.groupby('year'):
        print(f"    {yr}: {len(g):>4}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():>10,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    print(f"\n  Exit reason breakdown (base):")
    for reason, g in base.groupby('exit_reason'):
        print(f"    {reason:<12}: {len(g):>4}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f}")

    # ── Lot size contribution ─────────────────────────────────────────────────
    print(f"\n  Base lot-size contribution:")
    for lots_val, g in base.groupby('lots'):
        print(f"    Lots {lots_val}: {len(g):>4}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():>10,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    print(f"\n  Base score contribution:")
    for sc, g in base[base['score'] >= 0].groupby('score'):
        lots_val = g['lots'].iloc[0]
        print(f"    Score {sc} ({lots_val}L): {len(g):>4}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():>10,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── Equity curve ─────────────────────────────────────────────────────────
    def to_daily_pnl(df_, date_col='date'):
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_[date_col].astype(str), format='%Y%m%d')
        return df_.groupby('dt')['pnl'].sum()

    base_d  = to_daily_pnl(base)
    s4_d    = to_daily_pnl(s4_all)   if not s4_all.empty   else pd.Series(dtype=float)
    blank_d = to_daily_pnl(blank_kept) if not blank_kept.empty else pd.Series(dtype=float)

    idx = pd.date_range(start=base_d.index.min(), end=base_d.index.max(), freq='B')

    def cum(s):
        return s.reindex(idx, fill_value=0).cumsum()

    eq_base  = cum(base_d)
    eq_full  = cum(base_d
                   .add(s4_d.reindex(base_d.index, fill_value=0))
                   .add(blank_d.reindex(base_d.index, fill_value=0)))

    dd_base  = eq_base - eq_base.cummax()
    dd_full  = eq_full - eq_full.cummax()

    print(f"\n  Equity summary:")
    print(f"    Base only:  Rs.{int(eq_base.iloc[-1]):>12,.0f} | Max DD Rs.{int(dd_base.min()):>8,.0f}")
    print(f"    Full combo: Rs.{int(eq_full.iloc[-1]):>12,.0f} | Max DD Rs.{int(dd_full.min()):>8,.0f}")

    # Build equity chart
    def ep_list(s):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                for d, v in s.items() if pd.notna(v)]

    tv = {
        "isTvFormat": False,
        "candlestick": [],
        "volume": [],
        "lines": [
            {"id": "base",  "label": "Base (clean)",          "data": ep_list(eq_base),
             "color": "#2196F3", "lineWidth": 2},
            {"id": "full",  "label": "Full combo (Base+S4+Blank)", "data": ep_list(eq_full),
             "color": "#4CAF50", "lineWidth": 2},
        ],
        "markers": []
    }
    send_custom_chart("124_equity", tv,
                      title="124 — Fresh Clean Backtest: Equity Curve")

    # ── Save cleaned CSVs ─────────────────────────────────────────────────────
    base.to_csv(f'{OUT_DIR}/124_base_clean.csv', index=False)
    if not s4_all.empty:
        s4_all.to_csv(f'{OUT_DIR}/124_s4_clean.csv', index=False)
    blank_kept.to_csv(f'{OUT_DIR}/124_blank_kept.csv', index=False)
    all_df.to_csv(f'{OUT_DIR}/124_all_trades.csv', index=False)

    print(f"\n  Saved: 124_base_clean.csv | 124_s4_clean.csv | 124_blank_kept.csv | 124_all_trades.csv")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
