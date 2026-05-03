"""
123_full_day_coverage.py — Maximize coverage to 75%+ trading days
=================================================================
Current: 550 base + 231 blank = 781/1155 days (67.6%)
Target : 75% = 866 days minimum

Tests every blank day signal combination:
  X1: OHL CE filtered  — O=H + IB expanded DOWN (bearish confirmed)
  X2: Futures momentum — blank days, strong premium/discount → directional
  X3: Gap reversal     — large gap + inside CPR → fade
  X4: IB breakout      — price above/below IB at 9:46 → trend sell
  X5: Combined best    — priority chain to maximise coverage + quality

For each: WR, avg P&L, year breakdown, day coverage %.
Final: pick best combo, show combined system P&L + coverage.
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

OUT_DIR    = 'data/20260430'
DATA_PATH  = os.environ.get('INTER_SERVER_DATA_PATH', '/mnt/data/day-wise')
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
IB_END     = '09:45:00'
ENTRY_LATE = '09:46:02'
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
OHL_TOL    = 0.0015
MONTH_MAP  = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
              7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def t2m(t):
    h,m,s = map(int, str(t).split(':'))
    return h*60 + m + s/60


def simulate_sell(date_str, instr, entry_time):
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep*(1-TGT_PCT)); hsl = r2(ep*2.0); sl = hsl; md = 0.0
    for i in range(len(tks)):
        t = tks.iloc[i]['time']; p = tks.iloc[i]['price']
        if t >= EOD_EXIT:
            return r2((ep-p)*LOT_SIZE*SCALE), 'eod', r2(ep), r2(p)
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE*SCALE), 'target', r2(ep), r2(p)
        if p >= sl:  return r2((ep-p)*LOT_SIZE*SCALE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p)
    return r2((ep-tks.iloc[-1]['price'])*LOT_SIZE*SCALE),'eod',r2(ep),r2(tks.iloc[-1]['price'])


def load_futures_basis(date_str, spot_open):
    """Returns futures premium/discount at open (pts)."""
    try:
        dt = pd.Timestamp(date_str)
        yy = dt.strftime('%y'); mm = MONTH_MAP[dt.month]
        fut_path = f'{DATA_PATH}/{date_str}/NIFTY{yy}{mm}FUT.csv'
        if not os.path.exists(fut_path):
            # Try next month's contract
            nm = dt.month % 12 + 1; ny = dt.year + (1 if dt.month == 12 else 0)
            mm2 = MONTH_MAP[nm]; yy2 = str(ny)[2:]
            fut_path = f'{DATA_PATH}/{date_str}/NIFTY{yy2}{mm2}FUT.csv'
        if not os.path.exists(fut_path): return None
        fc = pd.read_csv(fut_path, header=None, names=['date','time','price','vol','oi'])
        fc = fc[fc['time'] >= '09:15:00']
        if fc.empty: return None
        return r2(fc.iloc[0]['price'] - spot_open)
    except Exception:
        return None


def _worker_blank_full(args):
    """Run all blank day strategies with priority chain."""
    date_str, meta, blank_set, already_traded = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set or date_str in already_traded:
        return []
    tc = meta.get('tc'); bc = meta.get('bc')
    gap_pct = meta.get('gap_pct', 0) or 0

    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        ib = day[day['time'] <= IB_END]
        if len(ib) < 5: return []
        ib_h = ib['price'].max(); ib_l = ib['price'].min()
        ib_close = ib.iloc[-1]['price']
        spot_open = ib.iloc[0]['price']
        if ib_h <= ib_l: return []

        # Pre-entry IB expansion (bias-free)
        pre = day[(day['time'] > IB_END) & (day['time'] < ENTRY_LATE)]
        ib_exp_up   = (not pre.empty) and (pre['price'].max() > ib_h)
        ib_exp_down = (not pre.empty) and (pre['price'].min() < ib_l)

        # Futures basis
        fut_basis = load_futures_basis(date_str, spot_open)

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        results = []
        fired = False  # one trade per day priority

        def do_trade(strat, sig, et=ENTRY_LATE):
            spot_at = day[day['time'] >= et]
            if spot_at.empty: return False
            atm = get_atm(spot_at.iloc[0]['price'])
            instr = f'NIFTY{expiries[0]}{atm}{sig}'
            res = simulate_sell(date_str, instr, et)
            if res is None: return False
            pnl, reason, ep, xp = res
            results.append({
                'strategy': strat, 'date': date_str, 'signal': sig,
                'entry_time': et, 'strike': atm, 'ep': ep, 'xp': xp,
                'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
                'fut_basis': fut_basis, 'gap_pct': gap_pct,
                'ib_exp_up': ib_exp_up, 'ib_exp_down': ib_exp_down,
            })
            return True

        # ── P1: S3 TrendIB corrected ──────────────────────────────────
        if not fired and tc is not None and bc is not None:
            bullish = spot_open > tc; bearish = spot_open < bc
            if bullish and ib_exp_up and not ib_exp_down:
                fired = do_trade('P1_S3_TrendIB', 'PE')
            elif bearish and ib_exp_down and not ib_exp_up:
                fired = do_trade('P1_S3_TrendIB', 'CE')

        # ── P2: OHL PE (O=IB_low, bullish) ───────────────────────────
        if not fired:
            is_ol = abs(spot_open - ib_l) / spot_open <= OHL_TOL
            if is_ol:
                fired = do_trade('P2_OHL_PE', 'PE')

        # ── P3: OHL CE filtered (O=IB_high + IB expanded DOWN) ────────
        if not fired:
            is_oh = abs(spot_open - ib_h) / spot_open <= OHL_TOL
            if is_oh and ib_exp_down:  # IB expanded down = bearish confirmed
                fired = do_trade('P3_OHL_CE_confirmed', 'CE')

        # ── P4: Futures momentum (premium > +20 → PE, discount < -20 → CE) ──
        if not fired and fut_basis is not None:
            if fut_basis >= 20:
                # Bullish sentiment → sell PE
                if tc is None or spot_open > bc:  # not strongly bearish CPR context
                    fired = do_trade('P4_FutPremium_PE', 'PE')
            elif fut_basis <= -20:
                # Bearish sentiment → sell CE
                if bc is None or spot_open < tc:  # not strongly bullish CPR context
                    fired = do_trade('P4_FutDiscount_CE', 'CE')

        # ── P5: IB breakout direction (price above/below IB at entry) ─
        if not fired:
            price_at_entry = pre.iloc[-1]['price'] if not pre.empty else spot_open
            if price_at_entry > ib_h * 1.001:  # price broke IB high → trend up → sell PE
                fired = do_trade('P5_IB_breakout_PE', 'PE')
            elif price_at_entry < ib_l * 0.999:  # price broke IB low → trend down → sell CE
                fired = do_trade('P5_IB_breakout_CE', 'CE')

        # ── P6: Gap fade (gap + inside CPR → fade direction) ──────────
        if not fired and tc is not None and bc is not None:
            inside_cpr = bc <= spot_open <= tc
            try:
                gap = float(gap_pct) if gap_pct else 0
            except Exception:
                gap = 0
            if inside_cpr:
                if gap >= 0.5:
                    # Gap up but opened inside CPR → fade → sell CE
                    fired = do_trade('P6_GapFade_CE', 'CE')
                elif gap <= -0.5:
                    # Gap down but opened inside CPR → fade → sell PE
                    fired = do_trade('P6_GapFade_PE', 'PE')

        # ── P7: IB direction bias (IB moved strongly one way → sell opposite option)
        if not fired:
            ib_move_pct = (ib_close - spot_open) / spot_open * 100
            if ib_move_pct >= 0.3:  # IB moved up 0.3%+ → uptrend → sell PE
                fired = do_trade('P7_IB_up_PE', 'PE')
            elif ib_move_pct <= -0.3:  # IB moved down 0.3%+ → downtrend → sell CE
                fired = do_trade('P7_IB_down_CE', 'CE')

        return results
    except Exception:
        return []


def stats(df_, label):
    if df_.empty: return
    wr = df_['win'].mean()*100; tp = df_['pnl'].sum(); ap = df_['pnl'].mean()
    pf_val = df_[df_['win']]['pnl'].sum() / abs(df_[~df_['win']]['pnl'].sum()) \
             if (~df_['win']).any() and df_[~df_['win']]['pnl'].sum() != 0 else 99
    print(f"  {label:<35} | {len(df_):>4}t | WR {wr:>5.1f}% | "
          f"Rs.{tp:>10,.0f} | Avg Rs.{ap:>7,.0f} | PF {pf_val:.2f}")


def main():
    # ── Load base + existing blank ─────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_key'] = pd.to_datetime(base['date'].astype(str),
                                       format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv':'pnl'})
    already_base = set(base['date_key'])

    beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                        sheet_name='all_days_observations', dtype={'date':str})
    beh['date_key'] = beh['date'].astype(str).str.strip()
    beh_map = {r['date_key']: {k:(None if pd.isna(v) else v) for k,v in r.items()}
               for _, r in beh.iterrows()}

    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]

    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    blank_set = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str)) | \
                set(pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')['date'].astype(str))

    total_days = len(dates_5yr)
    blank_days = len([d for d in dates_5yr if d in blank_set])
    print(f"Total days: {total_days} | Blank days: {blank_days} | Base traded: {len(already_base)}")
    print(f"Current coverage: {(len(already_base)/total_days*100):.1f}%")

    # ── Run full blank day strategies ──────────────────────────────────────
    print(f"\nRunning all blank day strategies (P1–P7, priority chain)...")
    args = [(d, beh_map.get(d,{}), blank_set, already_base) for d in dates_5yr]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw = pool.map(_worker_blank_full, args)
    el = (datetime.now()-t0).total_seconds()
    df_new = pd.DataFrame([t for day in raw for t in day])
    print(f"  {len(df_new)} blank day trades in {el:.1f}s")

    if df_new.empty:
        print("No blank day trades generated."); return

    df_new['year'] = df_new['date'].str[:4]

    # ── Per-strategy breakdown ─────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  BLANK DAY STRATEGY BREAKDOWN")
    print(f"{'='*75}")
    print(f"  {'Strategy':<35} | {'T':>4} | {'WR':>7} | {'Total':>12} | {'Avg':>9} | {'PF':>5}")
    print(f"  {'-'*75}")
    for strat, g in df_new.groupby('strategy'):
        stats(g, strat)

    print(f"  {'-'*75}")
    stats(df_new, 'ALL BLANK STRATEGIES')

    # ── CE vs PE breakdown ─────────────────────────────────────────────────
    print(f"\n  By signal:")
    for sig, g in df_new.groupby('signal'):
        stats(g, f"  {sig}")

    # ── Year breakdown ─────────────────────────────────────────────────────
    print(f"\n  Year breakdown (all blank):")
    for yr, g in df_new.groupby('year'):
        print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── Futures basis impact on blank trades ───────────────────────────────
    fb = df_new.dropna(subset=['fut_basis'])
    if not fb.empty:
        print(f"\n  Blank trades by futures basis:")
        fb['fb_bin'] = pd.cut(fb['fut_basis'],
            bins=[-200,-20,-5,5,20,50,200],
            labels=['<-20','-20to-5','-5to+5','+5to+20','+20to+50','>+50'])
        for b, g in fb.groupby('fb_bin', observed=True):
            print(f"    {b}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Avg Rs.{g['pnl'].mean():.0f}")

    # ── Coverage analysis ──────────────────────────────────────────────────
    new_days = set(df_new['date'].astype(str))
    total_covered = len(already_base | new_days)
    coverage_pct  = total_covered / total_days * 100
    print(f"\n{'='*75}")
    print(f"  COVERAGE ANALYSIS")
    print(f"{'='*75}")
    print(f"  Base days:              {len(already_base):>4} / {total_days}  ({len(already_base)/total_days*100:.1f}%)")
    print(f"  New blank day trades:   {len(new_days):>4} unique days")
    print(f"  TOTAL covered:          {total_covered:>4} / {total_days}  ({coverage_pct:.1f}%)")
    print(f"  Target (75%):           {int(total_days*0.75):>4} days")
    print(f"  Gap to 75%:             {max(0, int(total_days*0.75) - total_covered):>4} days")

    # ── Priority coverage breakdown ────────────────────────────────────────
    print(f"\n  Coverage by priority:")
    cum_days = set(already_base)
    for prio in sorted(df_new['strategy'].unique()):
        g = df_new[df_new['strategy']==prio]
        new_d = set(g['date'].astype(str)) - cum_days
        cum_days |= new_d
        print(f"    {prio:<30}: +{len(new_d):>3} new days | "
              f"cum {len(cum_days):>4}/{total_days} ({len(cum_days)/total_days*100:.1f}%) | "
              f"WR {g['win'].mean()*100:.1f}% | Avg Rs.{g['pnl'].mean():.0f}")

    # ── Also load S4 2nd trades ────────────────────────────────────────────
    try:
        df_s4 = pd.read_csv(f'{OUT_DIR}/117_s4_clean.csv')
        df_s4['year'] = df_s4['date'].astype(str).str[:4]
    except Exception:
        df_s4 = pd.DataFrame()

    # ── FINAL COMBINED SYSTEM ──────────────────────────────────────────────
    base_total = base['pnl'].sum()
    blank_total = df_new['pnl'].sum()
    s4_total = df_s4['pnl'].sum() if not df_s4.empty else 0
    grand = base_total + blank_total + s4_total

    print(f"\n{'='*75}")
    print(f"  FINAL COMBINED SYSTEM")
    print(f"{'='*75}")
    print(f"  A: Base strategy:       Rs.{base_total:>12,.0f}  {len(base)}t | WR {base['win'].mean()*100:.1f}%")
    print(f"  B: Blank day (P1-P7):   Rs.{blank_total:>+12,.0f}  {len(df_new)}t | WR {df_new['win'].mean()*100:.1f}%")
    print(f"  D: S4 2nd trade:        Rs.{s4_total:>+12,.0f}  {len(df_s4)}t | WR {df_s4['win'].mean()*100:.1f}%" if not df_s4.empty else "")
    print(f"  {'─'*53}")
    print(f"  GRAND TOTAL:            Rs.{grand:>12,.0f}  {len(base)+len(df_new)+len(df_s4)}t")
    print(f"  Coverage:               {coverage_pct:.1f}%  ({total_covered}/{total_days} days)")
    print(f"  vs Base:                Rs.{grand-base_total:>+12,.0f}  ({(grand/base_total-1)*100:+.1f}%)")

    # ── Quality: should we skip any strategy? ─────────────────────────────
    print(f"\n  Quality filter — strategies to keep (WR≥60% AND avg>0):")
    keep = []; skip = []
    for strat, g in df_new.groupby('strategy'):
        if g['win'].mean() >= 0.60 and g['pnl'].mean() > 0:
            keep.append(strat)
            print(f"    KEEP: {strat} — WR {g['win'].mean()*100:.1f}% | Avg Rs.{g['pnl'].mean():.0f}")
        else:
            skip.append(strat)
            print(f"    SKIP: {strat} — WR {g['win'].mean()*100:.1f}% | Avg Rs.{g['pnl'].mean():.0f}")

    df_kept = df_new[df_new['strategy'].isin(keep)]
    kept_days = set(df_kept['date'].astype(str))
    filtered_total = base_total + df_kept['pnl'].sum() + s4_total
    filtered_covered = len(already_base | kept_days)

    print(f"\n  After quality filter:")
    print(f"    P&L:      Rs.{filtered_total:>12,.0f}")
    print(f"    Coverage: {filtered_covered}/{total_days} ({filtered_covered/total_days*100:.1f}%)")

    # ── Equity curves ──────────────────────────────────────────────────────
    def to_daily(df_, date_col='date', fmt='%Y%m%d'):
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_[date_col].astype(str), format=fmt)
        return df_.groupby('dt')['pnl'].sum()

    base_d = to_daily(base, 'date', fmt='mixed')
    blank_d = to_daily(df_new, 'date')
    blank_kept_d = to_daily(df_kept, 'date') if not df_kept.empty else pd.Series(dtype=float)
    s4_d = to_daily(df_s4, 'date') if not df_s4.empty else pd.Series(dtype=float)

    all_idx = pd.date_range(start=base_d.index.min(), end=base_d.index.max(), freq='B')
    def cum(s): return s.reindex(all_idx, fill_value=0).cumsum()

    eq_base  = cum(base_d)
    eq_all   = cum(base_d.add(blank_d.reindex(base_d.index, fill_value=0))
                              .add(s4_d.reindex(base_d.index, fill_value=0)))
    eq_kept  = cum(base_d.add(blank_kept_d.reindex(base_d.index, fill_value=0))
                              .add(s4_d.reindex(base_d.index, fill_value=0))) if not blank_kept_d.empty else eq_base

    dd_base = eq_base - eq_base.cummax()
    dd_all  = eq_all  - eq_all.cummax()
    dd_kept = eq_kept - eq_kept.cummax()

    print(f"\n  Equity:")
    print(f"    Base only:    Rs.{int(eq_base.iloc[-1]):,.0f} | DD Rs.{int(dd_base.min()):,.0f}")
    print(f"    All blank:    Rs.{int(eq_all.iloc[-1]):,.0f} | DD Rs.{int(dd_all.min()):,.0f}")
    print(f"    Kept only:    Rs.{int(eq_kept.iloc[-1]):,.0f} | DD Rs.{int(dd_kept.min()):,.0f}")

    def ep(s):
        return [{"time":int(pd.Timestamp(d).timestamp()),"value":round(float(v),2)}
                for d,v in s.items() if pd.notna(v)]

    tv = {"isTvFormat":False,"candlestick":[],"volume":[],"lines":[
        {"id":"all","label":f"All P1-P7+S4 Rs.{int(eq_all.iloc[-1]):,.0f} ({coverage_pct:.0f}% days)",
         "color":"#26a69a","seriesType":"line","data":ep(eq_all)},
        {"id":"kept","label":f"Quality filtered Rs.{int(eq_kept.iloc[-1]):,.0f} ({filtered_covered/total_days*100:.0f}% days)",
         "color":"#0ea5e9","seriesType":"line","data":ep(eq_kept)},
        {"id":"base","label":f"Base Rs.{int(eq_base.iloc[-1]):,.0f} (47% days)",
         "color":"#9e9e9e","seriesType":"line","data":ep(eq_base)},
        {"id":"dd","label":f"DD all Rs.{int(dd_all.min()):,.0f}",
         "color":"#ef5350","seriesType":"baseline","baseValue":0,"isNewPane":True,"data":ep(dd_all)},
    ]}
    send_custom_chart("123_coverage", tv,
        title=f"Full Coverage System | {coverage_pct:.0f}% days | Rs.{int(eq_all.iloc[-1]):,.0f}")

    # Save
    df_new.to_csv(f'{OUT_DIR}/123_blank_all_strategies.csv', index=False)
    print(f"\n  Saved: 123_blank_all_strategies.csv")
    print("\nDone.")


if __name__ == '__main__':
    main()
