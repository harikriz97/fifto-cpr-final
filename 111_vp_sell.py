"""
111_vp_sell.py — Volume Profile selling on blank days
=====================================================
Previous day tick data → VAH / VAL / POC

Signal:
  Price touches VAH zone → sell CE (expect rejection at supply)
  Price touches VAL zone → sell PE (expect bounce at demand)

Zone definition: price within VP_PROX% of VAH/VAL
Entry window: 09:30 – 13:00 (first touch only)
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
EOD_EXIT   = '15:20:00'
ENTRY_START= '09:30:00'
ENTRY_END  = '13:00:00'
TGT_PCT    = 0.30
YEARS      = 5
VP_BINS    = 200
VP_VA_PCT  = 0.70
VP_PROX    = 0.003    # 0.3% touch zone around VAH/VAL

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


def compute_vp(tks_prev):
    """Return (poc, vah, val) from previous day ticks."""
    prices = tks_prev['price'].values
    if len(prices) < 10: return None, None, None
    p_min, p_max = prices.min(), prices.max()
    if p_max <= p_min: return None, None, None
    bins = np.linspace(p_min, p_max, VP_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vols = tks_prev['volume'].values
    counts = np.zeros(VP_BINS)
    for i, p in enumerate(prices):
        idx = min(np.searchsorted(bins[1:], p), VP_BINS - 1)
        counts[idx] += max(vols[i], 1)
    poc_idx = int(np.argmax(counts))
    poc = bin_centers[poc_idx]
    total = counts.sum(); target = total * VP_VA_PCT
    vah_idx = poc_idx; val_idx = poc_idx; cum = counts[poc_idx]
    while cum < target:
        can_up = vah_idx + 1 < VP_BINS
        can_dn = val_idx - 1 >= 0
        if can_up and can_dn:
            if counts[vah_idx + 1] >= counts[val_idx - 1]:
                vah_idx += 1; cum += counts[vah_idx]
            else:
                val_idx -= 1; cum += counts[val_idx]
        elif can_up: vah_idx += 1; cum += counts[vah_idx]
        elif can_dn: val_idx -= 1; cum += counts[val_idx]
        else: break
    return r2(poc), r2(bin_centers[vah_idx]), r2(bin_centers[val_idx])


def detect_vp_touch(tks, vah, val):
    """First tick that touches VAH or VAL zone. Returns (signal, entry_time) or (None, None)."""
    scan = tks[(tks['time'] >= ENTRY_START) & (tks['time'] <= ENTRY_END)]
    for _, row in scan.iterrows():
        p = row['price']; t = row['time']
        if abs(p - vah) / p <= VP_PROX:
            h, m, s = map(int, t.split(':'))
            s = min(s + 2, 59)
            return 'CE', f'{h:02d}:{m:02d}:{s:02d}'
        if abs(p - val) / p <= VP_PROX:
            h, m, s = map(int, t.split(':'))
            s = min(s + 2, 59)
            return 'PE', f'{h:02d}:{m:02d}:{s:02d}'
    return None, None


def _worker(args):
    date_str, vp_data, blank_set = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set: return []
    poc, vah, val = vp_data if vp_data else (None, None, None)
    if vah is None or val is None: return []
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return []

        signal, entry_t = detect_vp_touch(day, vah, val)
        if signal is None: return []

        spot_at = day[day['time'] >= entry_t[:8]]
        if spot_at.empty: return []
        spot_ref = spot_at.iloc[0]['price']

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        atm   = get_atm(spot_ref)
        instr = f'NIFTY{expiries[0]}{atm}{signal}'

        res = simulate_sell(date_str, instr, entry_t)
        if res is None: return []
        pnl, reason, ep, xp = res

        return [{'date': date_str, 'signal': signal, 'entry_time': entry_t,
                 'poc': poc, 'vah': vah, 'val': val,
                 'strike': atm, 'ep': ep, 'xp': xp,
                 'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason}]
    except Exception:
        return []


def main():
    print("Building daily ticks + volume profiles...")
    t0 = time.time()
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
    seed_idx  = max(0, all_dates.index(dates_5yr[0]) - 5)
    scan_all  = all_dates[seed_idx:]

    date_list = []
    vp_map    = {}
    prev_tks  = None
    prev_date = None

    for d in scan_all:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None: continue
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
        if len(day) < 2: continue
        if prev_tks is not None:
            poc, vah, val = compute_vp(prev_tks)
            vp_map[d] = (poc, vah, val)
        date_list.append(d)
        prev_tks = day
        prev_date = d

    print(f"  VP computed for {len(vp_map)} days in {time.time()-t0:.0f}s")

    # Blank days
    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    crt_blank = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str))
    blank_rem = pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')
    blank_set = crt_blank | set(blank_rem['date'].astype(str))
    blank_days = sorted([d for d in dates_5yr if d in blank_set])
    print(f"  Blank days: {len(blank_days)}")

    # Parallel backtest
    args_list = [(d, vp_map.get(d), blank_set) for d in dates_5yr]
    print(f"\nRunning VP backtest on {len(args_list)} days...")
    t0 = datetime.now()
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_worker, args_list)
    elapsed = (datetime.now() - t0).total_seconds()

    all_trades = [t for day in raw for t in day]
    if not all_trades:
        print("No trades."); return
    df = pd.DataFrame(all_trades)

    def stats(sub, label):
        if sub.empty: return
        wr  = sub['win'].mean() * 100
        tp  = sub['pnl'].sum()
        ap  = sub['pnl'].mean()
        wst = sub['pnl'].min()
        print(f"  {label:<28} | {len(sub):>4}t | WR {wr:>5.1f}% | "
              f"Rs.{tp:>9,.0f} | Avg Rs.{ap:>5,.0f} | Worst Rs.{wst:>7,.0f}")

    ce_df = df[df['signal']=='CE']
    pe_df = df[df['signal']=='PE']

    print(f"\n{'='*78}")
    print(f"  VOLUME PROFILE SELLING — BLANK DAYS ({elapsed:.0f}s)")
    print(f"{'='*78}")
    print(f"  {'Variant':<28} | {'T':>4} | {'WR':>7} | {'Total P&L':>11} | {'Avg':>8} | {'Worst':>10}")
    print(f"  {'-'*76}")
    stats(df,     'All')
    stats(ce_df,  'CE (VAH touch)')
    stats(pe_df,  'PE (VAL touch)')

    df['year'] = df['date'].str[:4]
    print(f"\n  Year breakdown:")
    for yr, g in df.groupby('year'):
        print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    exits = df['exit_reason'].value_counts()
    print(f"\n  Exit breakdown:")
    for r, c in exits.items():
        print(f"    {r:<12}: {c} ({round(c/len(df)*100,1)}%)")

    df.to_csv(f'{OUT_DIR}/111_vp_trades.csv', index=False)

    # Equity chart
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
    base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
    base_daily.columns = ['date', 'base_pnl']

    df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    vp_daily = df.groupby('date_dt')['pnl'].sum().reset_index()
    vp_daily.columns = ['date', 'vp_pnl']

    all_dt = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(vp_daily['date']))})
    m = all_dt.merge(base_daily, on='date', how='left').merge(vp_daily, on='date', how='left').fillna(0)
    m['base_eq'] = m['base_pnl'].cumsum()
    m['vp_eq']   = m['vp_pnl'].cumsum()
    m['comb_eq'] = (m['base_pnl'] + m['vp_pnl']).cumsum()
    comb_total = round(m['comb_eq'].iloc[-1], 0)
    comb_dd    = round((m['comb_eq'] - m['comb_eq'].cummax()).min(), 0)

    print(f"\n  Combined (Base + VP): Rs.{comb_total:,.0f} | DD Rs.{comb_dd:,.0f}")

    def eq_pts(series, dates):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                for d, v in zip(dates, series) if pd.notna(v)]

    dates = m['date']
    tv_json = {
        "isTvFormat": False, "candlestick": [], "volume": [],
        "lines": [
            {"id": "combined", "label": f"Combined Rs.{comb_total:,.0f}",
             "color": "#26a69a", "data": eq_pts(m['comb_eq'], dates), "seriesType": "line"},
            {"id": "base", "label": f"Base Rs.{int(m['base_pnl'].sum()):,.0f}",
             "color": "#0ea5e9", "data": eq_pts(m['base_eq'], dates), "seriesType": "line"},
            {"id": "vp", "label": f"VP blank Rs.{int(df['pnl'].sum()):,.0f}",
             "color": "#f59e0b", "data": eq_pts(m['vp_eq'], dates), "seriesType": "line"},
            {"id": "dd", "label": f"DD max Rs.{comb_dd:,.0f}", "color": "#ef5350",
             "data": eq_pts(m['comb_eq'] - m['comb_eq'].cummax(), dates),
             "seriesType": "baseline", "baseValue": 0, "isNewPane": True},
        ]
    }
    total = len(df); wr = round(df['win'].mean()*100, 1)
    send_custom_chart("111_vp_sell", tv_json,
        title=f"Volume Profile Selling — Blank Days | {total}t | WR {wr}% | Combined Rs.{comb_total:,.0f}")

    print(f"\n  Saved → {OUT_DIR}/111_vp_trades.csv")
    print("Done.")

if __name__ == '__main__':
    main()
