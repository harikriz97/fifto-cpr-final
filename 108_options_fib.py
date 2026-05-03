"""
108_options_fib.py — Fibonacci zones on OPTION PREMIUM itself (not spot)
=========================================================================
Concept:
  Option premium has its own price structure.
  In the first 45 min (9:15-10:00), option premium swings up/down.
  When CE/PE premium bounces back to 50-61.8% Fib zone of that morning swing
  → premium is at resistance of its own range → sell it there.

CE signal:
  Morning swing: CE_open (high) → CE_first45_low (swing low)
  Range = CE_open - CE_first45_low
  Zone: CE_first45_low + 0.50*Range  to  CE_first45_low + 0.618*Range
  When CE bounces INTO this zone from below → sell CE

PE signal:
  Morning swing: PE_open (high) → PE_first45_low (swing low)
  Same logic → sell PE when PE bounces into its 50-61.8% zone

Why this works:
  Option premium that dropped and bounced to 50-61.8% = 'dead cat bounce'
  Smart traders sell at this zone expecting premium to fall again
  Works for BOTH CE and PE independently

Tests:
  A: CE only sell (at CE premium Fib zone)
  B: PE only sell (at PE premium Fib zone)
  C: First signal of the day (whichever fires first, CE or PE)
  D: Both CE + PE same day if both zones hit (each independently)
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
ENTRY_WINDOW_START = '10:00:00'
ENTRY_WINDOW_END   = '13:00:00'
SEED_END   = '10:00:00'   # first 45 min for swing detection
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
YEARS      = 5

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


def fib_zone_sell(date_str, opt_type, expiry, atm):
    """
    Load option tick data, find 9:15-10:00 swing (open → low),
    detect when premium bounces into 50-61.8% zone after 10:00,
    return entry time or None.
    """
    instr = f'NIFTY{expiry}{atm}{opt_type}'
    tks = load_tick_data(date_str, instr, '09:15:00')
    if tks is None or tks.empty: return None, None

    seed = tks[(tks['time'] >= '09:15:00') & (tks['time'] < SEED_END)]
    rest = tks[(tks['time'] >= ENTRY_WINDOW_START) & (tks['time'] <= ENTRY_WINDOW_END)]

    if seed.empty or rest.empty: return None, None

    # Swing: open (first tick) → seed low
    opt_open = seed.iloc[0]['price']
    seed_low  = seed['price'].min()
    seed_high = seed['price'].max()

    # Option must have dropped at least 15% from open for a meaningful swing
    if opt_open <= 0 or (opt_open - seed_low) / opt_open < 0.12:
        return None, None

    swing_range = opt_open - seed_low
    zone_low  = r2(seed_low + 0.50 * swing_range)
    zone_high = r2(seed_low + 0.618 * swing_range)

    # Scan for first bounce INTO zone after 10:00
    prev_p = None
    for _, row in rest.iterrows():
        p = row['price']
        t = row['time']
        if prev_p is not None:
            # Price was below zone and enters from below (bounce up into zone)
            if prev_p < zone_low and zone_low <= p <= zone_high:
                entry_t = t[:5] + ':02'  # same minute + 2s (forward-bias safe)
                return entry_t, {
                    'opt_open': opt_open, 'seed_low': seed_low,
                    'zone_low': zone_low, 'zone_high': zone_high,
                    'swing_range': swing_range
                }
        prev_p = p

    return None, None


def _worker(args):
    date_str = args[0]
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    results = []
    try:
        spot = load_spot_data(date_str, 'NIFTY')
        if spot is None: return []
        spot_10 = spot[spot['time'] >= '10:00:00']
        if spot_10.empty: return []
        spot_ref = spot_10.iloc[0]['price']

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        expiry = expiries[0]
        atm = get_atm(spot_ref)

        for opt_type in ['CE', 'PE']:
            entry_t, fib_info = fib_zone_sell(date_str, opt_type, expiry, atm)
            if entry_t is None: continue

            res = simulate_sell(date_str, f'NIFTY{expiry}{atm}{opt_type}', entry_t)
            if res is None: continue
            pnl, reason, ep, xp = res

            results.append({
                'date': date_str,
                'opt_type': opt_type,
                'entry_time': entry_t,
                'strike': atm,
                'opt_open': fib_info['opt_open'],
                'zone_low': fib_info['zone_low'],
                'zone_high': fib_info['zone_high'],
                'ep': ep, 'xp': xp,
                'pnl': pnl, 'win': pnl > 0,
                'exit_reason': reason,
            })
    except Exception as e:
        pass
    return results


def main():
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

    print(f"Running options Fib backtest on {len(dates_5yr)} days...")
    t0 = datetime.now()

    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_worker, [(d,) for d in dates_5yr])

    elapsed = (datetime.now() - t0).total_seconds()
    all_trades = [t for day in raw for t in day]

    if not all_trades:
        print("No trades found.")
        return

    df = pd.DataFrame(all_trades)
    df['pnl'] = df['pnl'].round(2)

    total = len(df)
    wr    = round(df['win'].mean() * 100, 2)
    tpnl  = round(df['pnl'].sum(), 2)
    apnl  = round(df['pnl'].mean(), 2)

    ce_df = df[df['opt_type']=='CE']
    pe_df = df[df['opt_type']=='PE']

    print(f"\n{'='*65}")
    print(f"  OPTIONS FIB ZONE SELLING — ALL DAYS 5yr ({elapsed:.0f}s)")
    print(f"{'='*65}")
    print(f"  Total : {total}t | WR {wr}% | P&L Rs.{tpnl:,.0f} | Avg Rs.{apnl:,.0f}")
    print(f"  CE    : {len(ce_df)}t | WR {ce_df['win'].mean()*100:.1f}% | "
          f"Rs.{ce_df['pnl'].sum():,.0f} | Avg Rs.{ce_df['pnl'].mean():.0f}")
    print(f"  PE    : {len(pe_df)}t | WR {pe_df['win'].mean()*100:.1f}% | "
          f"Rs.{pe_df['pnl'].sum():,.0f} | Avg Rs.{pe_df['pnl'].mean():.0f}")

    exits = df['exit_reason'].value_counts()
    print(f"\n  Exit breakdown:")
    for r, c in exits.items():
        print(f"    {r:<12}: {c} ({round(c/total*100,1)}%)")

    # Days with BOTH CE and PE signals
    both = df.groupby('date').filter(lambda x: len(x['opt_type'].unique()) == 2)
    ce_only = df[~df['date'].isin(both['date'])][df['opt_type']=='CE']
    pe_only = df[~df['date'].isin(both['date'])][df['opt_type']=='PE']

    print(f"\n  Days with BOTH CE+PE signal: {len(both['date'].unique())} days | "
          f"WR {both['win'].mean()*100:.1f}% | Rs.{both['pnl'].sum():,.0f}")
    print(f"  CE only days: {len(ce_only['date'].unique())} | "
          f"WR {ce_only['win'].mean()*100:.1f}% | Rs.{ce_only['pnl'].sum():,.0f}" if len(ce_only) else "  CE only: 0")
    print(f"  PE only days: {len(pe_only['date'].unique())} | "
          f"WR {pe_only['win'].mean()*100:.1f}% | Rs.{pe_only['pnl'].sum():,.0f}" if len(pe_only) else "  PE only: 0")

    # By year
    df['year'] = df['date'].str[:4]
    print(f"\n  Year breakdown:")
    for yr, g in df.groupby('year'):
        print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # Save
    df.to_csv(f'{OUT_DIR}/108_options_fib_trades.csv', index=False)

    # Equity chart
    df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
    base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
    base_daily.columns = ['date', 'base_pnl']

    def make_combined(sub_df, label):
        gd = sub_df.groupby('date_dt')['pnl'].sum().reset_index()
        gd.columns = ['date', 'strat_pnl']
        all_dt = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(gd['date']))})
        m = all_dt.merge(base_daily, on='date', how='left').merge(gd, on='date', how='left')
        m['base_pnl']  = m['base_pnl'].fillna(0)
        m['strat_pnl'] = m['strat_pnl'].fillna(0)
        m['comb_eq']   = (m['base_pnl'] + m['strat_pnl']).cumsum()
        tp = round(sub_df['pnl'].sum(), 0)
        return m, tp

    lines = []
    for sub, label, color in [
        (df,    f'CE+PE Rs.{tpnl:,.0f}', '#26a69a'),
        (ce_df, f'CE only Rs.{ce_df["pnl"].sum():,.0f}', '#0ea5e9'),
        (pe_df, f'PE only Rs.{pe_df["pnl"].sum():,.0f}', '#f59e0b'),
    ]:
        m, tp = make_combined(sub, label)
        lines.append({
            "id": label[:6].replace(' ','_'),
            "label": label,
            "color": color,
            "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                     for d, v in zip(m['date'], m['comb_eq']) if pd.notna(v)],
            "seriesType": "line",
        })

    # Base alone
    bs = base_daily.sort_values('date')
    lines.append({
        "id": "base", "label": f"Base Rs.{int(base_daily['base_pnl'].sum()):,.0f}",
        "color": "#9e9e9e",
        "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                 for d, v in zip(bs['date'], bs['base_pnl'].cumsum())],
        "seriesType": "line",
    })

    tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}
    send_custom_chart("108_options_fib", tv_json,
                      title=f"Options Fib Zone Selling | {total}t | WR {wr}% | P&L Rs.{tpnl:,.0f}")

    print(f"\n  Saved → {OUT_DIR}/108_options_fib_trades.csv")
    print("\nDone.")

if __name__ == '__main__':
    main()
