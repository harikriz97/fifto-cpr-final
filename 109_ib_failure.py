"""
109_ib_failure.py — Initial Balance (IB) False Breakout Selling
================================================================
Auction Market Theory: first 30 min (9:15-9:45) = Initial Balance
70% of days: first hour puts the day's HIGH or LOW.
When price breaks IB boundary and FAILS to hold → false breakout → sell option.

Signal:
  IB High Failure → price breaks above IB_H, comes back below → sell CE
  IB Low  Failure → price breaks below IB_L, comes back above → sell PE

Entry: tick when price crosses back inside IB + 2 seconds (no forward bias)
Entry window: 9:46 – 13:00
Target: 30%, trailing SL, ATM strike

Regime variants tested:
  A: All days (no filter)
  B: Narrow IB days only  (IB range < median IB range → ranging market)
  C: Wide IB days only    (IB range > median IB range → trending market)
  D: Narrow IB + CPR bias filter (IB fail direction matches CPR open bias)

Breakout confirmation: price must cross IB boundary by ≥ 0.05% of spot
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
YEARS      = 5
BRK_BUF    = 0.0005   # 0.05% breakout buffer

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot / STRIKE_INT) * STRIKE_INT)

def simulate_sell(date_str, instrument, entry_time):
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
            return r2((ep - p) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(p)
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE * SCALE), 'target', r2(ep), r2(p)
        if p >= sl:  return r2((ep - p) * LOT_SIZE * SCALE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p)
    return r2((ep - ps[-1]) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(ps[-1])


def detect_ib_failure(tks, ib_h, ib_l, spot_open):
    """
    Scan ticks from 9:46 onward. Return (signal, entry_time) or (None, None).
    signal = 'CE' (IB High failure) or 'PE' (IB Low failure)
    """
    buf = spot_open * BRK_BUF
    h_broken = False   # price broke above IB_H
    l_broken = False   # price broke below IB_L
    signal = None
    entry_t = None

    scan = tks[(tks['time'] >= ENTRY_START) & (tks['time'] <= ENTRY_END)]
    for _, row in scan.iterrows():
        p = row['price']
        t = row['time']

        # Mark breakout
        if p > ib_h + buf:
            h_broken = True
        if p < ib_l - buf:
            l_broken = True

        # Failure detection
        if h_broken and p <= ib_h and signal is None:
            # Price came back inside after breaking high → CE sell
            # Entry: current tick time + 2s (next available)
            h, m, s = map(int, t.split(':'))
            s = min(s + 2, 59)
            entry_t = f'{h:02d}:{m:02d}:{s:02d}'
            signal = 'CE'
            break

        if l_broken and p >= ib_l and signal is None:
            # Price came back inside after breaking low → PE sell
            h, m, s = map(int, t.split(':'))
            s = min(s + 2, 59)
            entry_t = f'{h:02d}:{m:02d}:{s:02d}'
            signal = 'PE'
            break

    return signal, entry_t


def _worker(args):
    date_str, ib_range_median, pdh, pdl = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    results = []
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return []

        # Build IB
        ib_tks = day[day['time'] <= IB_END]
        if ib_tks.empty: return []
        ib_h = ib_tks['price'].max()
        ib_l = ib_tks['price'].min()
        ib_range = ib_h - ib_l
        spot_open = ib_tks.iloc[0]['price']

        if ib_range <= 0: return []

        # CPR bias
        cpr_bias = None
        if pdh and pdl:
            pvt = round((pdh + pdl) / 2, 2)   # simplified pivot (using PDH/PDL)
            bc  = round((pdh + pdl) / 2, 2)
            tc  = bc  # will use just open vs PDH/PDL midpoint
            if spot_open > (pdh + pdl) / 2:
                cpr_bias = 'bull'
            else:
                cpr_bias = 'bear'

        # Regime
        narrow = ib_range < ib_range_median

        # Detect IB failure
        signal, entry_t = detect_ib_failure(day, ib_h, ib_l, spot_open)
        if signal is None or entry_t is None:
            return []

        # Get ATM at entry time
        spot_at = day[day['time'] >= entry_t[:8]]
        if spot_at.empty: return []
        spot_ref = spot_at.iloc[0]['price']

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        atm = get_atm(spot_ref)
        instr = f'NIFTY{expiries[0]}{atm}{signal}'

        res = simulate_sell(date_str, instr, entry_t)
        if res is None: return []
        pnl, reason, ep, xp = res

        # CPR aligned?
        cpr_aligned = (
            (signal == 'CE' and cpr_bias == 'bear') or
            (signal == 'PE' and cpr_bias == 'bull')
        )

        results.append({
            'date': date_str,
            'signal': signal,
            'entry_time': entry_t,
            'ib_h': r2(ib_h), 'ib_l': r2(ib_l),
            'ib_range': r2(ib_range),
            'narrow_ib': narrow,
            'cpr_bias': cpr_bias,
            'cpr_aligned': cpr_aligned,
            'strike': atm, 'ep': ep, 'xp': xp,
            'pnl': pnl, 'win': pnl > 0,
            'exit_reason': reason,
        })
    except Exception:
        pass
    return results


def main():
    # ── Build IB range history for median calculation ─────────────────��───────
    print("Building IB range history...")
    t0 = time.time()
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

    # Seed: extra 10 days before 5yr for PDH/PDL
    seed_idx  = max(0, all_dates.index(dates_5yr[0]) - 10)
    scan_all  = all_dates[seed_idx:]

    daily_rows = []
    ib_ranges  = []
    for d in scan_all:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None: continue
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
        if len(day) < 2: continue
        ib = day[day['time'] <= IB_END]
        if ib.empty: continue
        ib_r = ib['price'].max() - ib['price'].min()
        daily_rows.append({'date': d, 'h': day['price'].max(), 'l': day['price'].min(),
                           'ib_range': ib_r})
        if d in dates_5yr:
            ib_ranges.append(ib_r)

    df_d = pd.DataFrame(daily_rows)
    df_d['pdh'] = df_d['h'].shift(1)
    df_d['pdl'] = df_d['l'].shift(1)
    df_d = df_d.dropna()
    pdhl_map = {row['date']: (row['pdh'], row['pdl']) for _, row in df_d.iterrows()}
    ib_median = np.median(ib_ranges)
    print(f"  IB range median: {ib_median:.1f} pts | {len(dates_5yr)} days in {time.time()-t0:.0f}s")

    # ── Parallel backtest ─────────────────────────────────────────────────────
    args_list = []
    for d in dates_5yr:
        pdh, pdl = pdhl_map.get(d, (None, None))
        args_list.append((d, ib_median, pdh, pdl))

    print(f"\nRunning IB failure backtest on {len(args_list)} days...")
    t0 = datetime.now()
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_worker, args_list)
    elapsed = (datetime.now() - t0).total_seconds()

    all_trades = [t for day in raw for t in day]
    if not all_trades:
        print("No trades found.")
        return

    df = pd.DataFrame(all_trades)

    # ── Print results for all variants ────────────────────────────────────────
    def stats(sub, label):
        if sub.empty: return
        wr  = sub['win'].mean()*100
        tp  = sub['pnl'].sum()
        ap  = sub['pnl'].mean()
        wst = sub['pnl'].min()
        print(f"  {label:<32} | {len(sub):>4}t | WR {wr:>5.1f}% | "
              f"Rs.{tp:>9,.0f} | Avg Rs.{ap:>5,.0f} | Worst Rs.{wst:>7,.0f}")

    print(f"\n{'='*80}")
    print(f"  INITIAL BALANCE FAILURE — ALL DAYS 5yr ({elapsed:.0f}s)")
    print(f"{'='*80}")
    print(f"  {'Variant':<32} | {'T':>4} | {'WR':>7} | {'Total P&L':>11} | {'Avg':>8} | {'Worst':>10}")
    print(f"  {'-'*78}")
    stats(df,                               'A: All days')
    stats(df[df['signal']=='CE'],           'A-CE: All days CE only')
    stats(df[df['signal']=='PE'],           'A-PE: All days PE only')
    print(f"  {'-'*78}")
    stats(df[df['narrow_ib']==True],        'B: Narrow IB (ranging)')
    stats(df[(df['narrow_ib']==True) & (df['signal']=='CE')], 'B-CE: Narrow IB + CE')
    stats(df[(df['narrow_ib']==True) & (df['signal']=='PE')], 'B-PE: Narrow IB + PE')
    print(f"  {'-'*78}")
    stats(df[df['narrow_ib']==False],       'C: Wide IB (trending)')
    stats(df[(df['narrow_ib']==False) & (df['signal']=='CE')], 'C-CE: Wide IB + CE')
    stats(df[(df['narrow_ib']==False) & (df['signal']=='PE')], 'C-PE: Wide IB + PE')
    print(f"  {'-'*78}")
    stats(df[df['cpr_aligned']==True],      'D: CPR aligned')
    stats(df[(df['cpr_aligned']==True) & (df['narrow_ib']==True)], 'D+B: CPR + Narrow IB')

    exits = df['exit_reason'].value_counts()
    print(f"\n  Exit breakdown (all):")
    for r, c in exits.items():
        print(f"    {r:<12}: {c} ({round(c/len(df)*100,1)}%)")

    # Year breakdown
    df['year'] = df['date'].str[:4]
    print(f"\n  Year breakdown (all):")
    for yr, g in df.groupby('year'):
        print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    df.to_csv(f'{OUT_DIR}/109_ib_failure_trades.csv', index=False)

    # ── Equity chart ──────────────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
    base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
    base_daily.columns = ['date', 'base_pnl']

    def make_eq(sub_df, color, label):
        if sub_df.empty: return None
        sub_df = sub_df.copy()
        sub_df['date_dt'] = pd.to_datetime(sub_df['date'].astype(str), format='%Y%m%d')
        gd = sub_df.groupby('date_dt')['pnl'].sum().reset_index()
        gd.columns = ['date','strat_pnl']
        all_dt = pd.DataFrame({'date': sorted(set(base_daily['date'])|set(gd['date']))})
        m = all_dt.merge(base_daily,on='date',how='left').merge(gd,on='date',how='left')
        m = m.fillna(0)
        m['comb_eq'] = (m['base_pnl'] + m['strat_pnl']).cumsum()
        tp = round(sub_df['pnl'].sum(), 0)
        return {
            "id": label.replace(' ','_'),
            "label": f"{label} Rs.{tp:,.0f}",
            "color": color,
            "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v),2)}
                     for d, v in zip(m['date'], m['comb_eq']) if pd.notna(v)],
            "seriesType": "line",
        }

    lines = []
    for sub, col, lbl in [
        (df,                                             '#26a69a', 'All days'),
        (df[df['narrow_ib']==True],                      '#0ea5e9', 'Narrow IB'),
        (df[df['cpr_aligned']==True],                    '#f59e0b', 'CPR aligned'),
        (df[(df['narrow_ib']==True)&(df['cpr_aligned']==True)], '#ab47bc', 'Narrow+CPR'),
        (df[df['signal']=='CE'],                         '#ef5350', 'CE only'),
    ]:
        ln = make_eq(sub, col, lbl)
        if ln: lines.append(ln)

    # base alone
    bs = base_daily.sort_values('date')
    lines.append({
        "id": "base", "label": f"Base Rs.{int(bs['base_pnl'].sum()):,.0f}",
        "color": "#9e9e9e",
        "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v),2)}
                 for d,v in zip(bs['date'], bs['base_pnl'].cumsum())],
        "seriesType": "line",
    })

    tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}
    total = len(df); wr = round(df['win'].mean()*100, 2); tp = round(df['pnl'].sum(), 0)
    send_custom_chart("109_ib_failure", tv_json,
                      title=f"IB False Breakout | {total}t | WR {wr}% | All Rs.{tp:,.0f} | IB Median {ib_median:.0f}pts")

    print(f"\n  Saved → {OUT_DIR}/109_ib_failure_trades.csv")
    print("\nDone.")

if __name__ == '__main__':
    main()
