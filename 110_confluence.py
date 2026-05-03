"""
110_confluence.py — Confluence blank day strategy
==================================================
Combine 4 signals for higher-conviction blank day trades:

  Entry signal : IB Failure (false breakout after 9:45)
  Filter 1     : Ichimoku cloud direction (daily)
  Filter 2     : CPR bias (open vs TC/BC)
  Filter 3     : Volume Profile (open vs VAH/VAL)

Confluence scoring:
  Score 0 = IB alone
  Score 1 = IB + 1 filter agrees
  Score 2 = IB + 2 filters agree
  Score 3 = IB + all 3 filters agree
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
VP_BINS    = 200
VP_VA_PCT  = 0.70

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
    buf = spot_open * BRK_BUF
    h_broken = False; l_broken = False
    signal = None; entry_t = None
    scan = tks[(tks['time'] >= ENTRY_START) & (tks['time'] <= ENTRY_END)]
    for _, row in scan.iterrows():
        p = row['price']; t = row['time']
        if p > ib_h + buf: h_broken = True
        if p < ib_l - buf: l_broken = True
        if h_broken and p <= ib_h and signal is None:
            h, m, s = map(int, t.split(':'))
            s = min(s + 2, 59)
            entry_t = f'{h:02d}:{m:02d}:{s:02d}'
            signal = 'CE'; break
        if l_broken and p >= ib_l and signal is None:
            h, m, s = map(int, t.split(':'))
            s = min(s + 2, 59)
            entry_t = f'{h:02d}:{m:02d}:{s:02d}'
            signal = 'PE'; break
    return signal, entry_t


def compute_volume_profile(tks_prev):
    """Return (vah, val) — Value Area High/Low from previous day ticks."""
    prices = tks_prev['price'].values
    if len(prices) < 10: return None, None
    p_min, p_max = prices.min(), prices.max()
    if p_max <= p_min: return None, None
    bins = np.linspace(p_min, p_max, VP_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vols = tks_prev['volume'].values
    counts = np.zeros(VP_BINS)
    for i, p in enumerate(prices):
        idx = min(np.searchsorted(bins[1:], p), VP_BINS - 1)
        counts[idx] += max(vols[i], 1)
    poc_idx = int(np.argmax(counts))
    total = counts.sum(); target = total * VP_VA_PCT
    vah_idx = poc_idx; val_idx = poc_idx
    cum = counts[poc_idx]
    while cum < target:
        can_up = vah_idx + 1 < VP_BINS
        can_dn = val_idx - 1 >= 0
        if can_up and can_dn:
            if counts[vah_idx + 1] >= counts[val_idx - 1]:
                vah_idx += 1; cum += counts[vah_idx]
            else:
                val_idx -= 1; cum += counts[val_idx]
        elif can_up:
            vah_idx += 1; cum += counts[vah_idx]
        elif can_dn:
            val_idx -= 1; cum += counts[val_idx]
        else:
            break
    return r2(bin_centers[vah_idx]), r2(bin_centers[val_idx])


def _worker(args):
    date_str, meta, blank_set = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set:
        return []
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return []

        ib_tks = day[day['time'] <= IB_END]
        if ib_tks.empty: return []
        ib_h = ib_tks['price'].max()
        ib_l = ib_tks['price'].min()
        ib_range = ib_h - ib_l
        spot_open = ib_tks.iloc[0]['price']
        if ib_range <= 0: return []

        signal, entry_t = detect_ib_failure(day, ib_h, ib_l, spot_open)
        if signal is None: return []

        # ── Filter 1: Ichimoku ─────────────────────────────────────────────────
        ichi_sig   = meta.get('ichi_sig')
        ichimoku_ok = (ichi_sig == signal)

        # ── Filter 2: CPR Bias (open vs TC/BC) ────────────────────────────────
        tc = meta.get('tc'); bc = meta.get('bc')
        cpr_sig = None
        if tc is not None and bc is not None:
            if   spot_open > tc: cpr_sig = 'PE'   # above TC → bullish → sell PE
            elif spot_open < bc: cpr_sig = 'CE'   # below BC → bearish → sell CE
        cpr_ok = (cpr_sig == signal) if cpr_sig else False

        # ── Filter 3: Volume Profile (open vs VAH/VAL) ────────────────────────
        vah = meta.get('vah'); val = meta.get('val')
        vp_sig = None
        if vah is not None and val is not None:
            if   spot_open > vah: vp_sig = 'PE'   # above value area → bullish → sell PE
            elif spot_open < val: vp_sig = 'CE'   # below value area → bearish → sell CE
        vp_ok = (vp_sig == signal) if vp_sig else False

        confluence = int(ichimoku_ok) + int(cpr_ok) + int(vp_ok)

        # Execute trade
        spot_at = day[day['time'] >= entry_t[:8]]
        if spot_at.empty: return []
        spot_ref = spot_at.iloc[0]['price']
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        atm  = get_atm(spot_ref)
        instr = f'NIFTY{expiries[0]}{atm}{signal}'

        res = simulate_sell(date_str, instr, entry_t)
        if res is None: return []
        pnl, reason, ep, xp = res

        return [{
            'date': date_str, 'signal': signal, 'entry_time': entry_t,
            'ib_range': r2(ib_range),
            'ichi_ok': ichimoku_ok, 'cpr_ok': cpr_ok, 'vp_ok': vp_ok,
            'confluence': confluence,
            'strike': atm, 'ep': ep, 'xp': xp,
            'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
        }]
    except Exception:
        return []


def main():
    print("Building daily seed data (Ichimoku needs ~100 day seed)...")
    t0 = time.time()
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
    seed_idx  = max(0, all_dates.index(dates_5yr[0]) - 110)
    scan_all  = all_dates[seed_idx:]

    rows = []
    prev_tks_cache = {}   # date → ticks (for VP, we keep the previous day's)
    prev_date = None
    for d in scan_all:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None: continue
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
        if len(day) < 2: continue
        row = {
            'date': d,
            'o': day.iloc[0]['price'],
            'h': day['price'].max(),
            'l': day['price'].min(),
            'c': day.iloc[-1]['price'],
        }
        # Store full ticks for VP of next day
        prev_tks_cache[d] = day
        rows.append(row)

    df_d = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  {len(df_d)} daily candles in {time.time()-t0:.0f}s")

    # ── Ichimoku ───────────────────────────────────────────────────────────────
    def rolling_mid(h, l, n):
        return ((h.rolling(n).max() + l.rolling(n).min()) / 2).round(2)

    df_d['tenkan'] = rolling_mid(df_d['h'], df_d['l'], 9)
    df_d['kijun']  = rolling_mid(df_d['h'], df_d['l'], 26)
    df_d['span_a'] = ((df_d['tenkan'].shift(26) + df_d['kijun'].shift(26)) / 2).round(2)
    df_d['span_b'] = rolling_mid(df_d['h'].shift(26), df_d['l'].shift(26), 52)
    df_d['cloud_top']    = df_d[['span_a','span_b']].max(axis=1)
    df_d['cloud_bottom'] = df_d[['span_a','span_b']].min(axis=1)

    def cloud_signal(row):
        if pd.isna(row['cloud_top']): return None
        if row['o'] > row['cloud_top']:    return 'PE'
        if row['o'] < row['cloud_bottom']: return 'CE'
        return None

    df_d['ichi_sig'] = df_d.apply(cloud_signal, axis=1)

    # ── CPR (proper TC/BC) ─────────────────────────────────────────────────────
    df_d['pdh'] = df_d['h'].shift(1)
    df_d['pdl'] = df_d['l'].shift(1)
    df_d['pdc'] = df_d['c'].shift(1)
    df_d['pvt'] = ((df_d['pdh'] + df_d['pdl'] + df_d['pdc']) / 3).round(2)
    df_d['bc']  = ((df_d['pdh'] + df_d['pdl']) / 2).round(2)
    df_d['tc']  = (2 * df_d['pvt'] - df_d['bc']).round(2)

    # ── Volume Profile (previous day VAH/VAL) ─────────────────────────────────
    print("Computing volume profiles...")
    t0 = time.time()
    dates_list = df_d['date'].tolist()
    vah_map = {}; val_map = {}
    for i in range(1, len(dates_list)):
        d = dates_list[i]
        prev = dates_list[i-1]
        prev_tks = prev_tks_cache.get(prev)
        if prev_tks is not None:
            vah, val = compute_volume_profile(prev_tks)
            vah_map[d] = vah; val_map[d] = val
    print(f"  Volume profiles done in {time.time()-t0:.0f}s")

    # ── Build meta map ─────────────────────────────────────────────────────────
    meta_map = {}
    for _, row in df_d.iterrows():
        d = row['date']
        meta_map[d] = {
            'ichi_sig': row['ichi_sig'],
            'tc': row['tc'] if pd.notna(row['tc']) else None,
            'bc': row['bc'] if pd.notna(row['bc']) else None,
            'vah': vah_map.get(d),
            'val': val_map.get(d),
        }

    # ── Load blank days ────────────────────────────────────────────────────────
    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    crt_blank = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str))
    blank_rem = pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')
    blank_set = crt_blank | set(blank_rem['date'].astype(str))
    blank_days = sorted([d for d in dates_5yr if d in blank_set])
    print(f"\nBlank days: {len(blank_days)} | Running parallel backtest...")

    # ── Parallel backtest ──────────────────────────────────────────────────────
    args_list = [(d, meta_map.get(d, {}), blank_set) for d in dates_5yr]
    t0 = datetime.now()
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_worker, args_list)
    elapsed = (datetime.now() - t0).total_seconds()

    all_trades = [t for day in raw for t in day]
    if not all_trades:
        print("No trades."); return
    df = pd.DataFrame(all_trades)
    print(f"  {len(df)} trades in {elapsed:.1f}s")

    # ── Stats by confluence tier ───────────────────────────────────────────────
    def stats(sub, label):
        if sub.empty: return
        wr  = sub['win'].mean() * 100
        tp  = sub['pnl'].sum()
        ap  = sub['pnl'].mean()
        wst = sub['pnl'].min()
        print(f"  {label:<30} | {len(sub):>4}t | WR {wr:>5.1f}% | "
              f"Rs.{tp:>9,.0f} | Avg Rs.{ap:>5,.0f} | Worst Rs.{wst:>7,.0f}")

    print(f"\n{'='*85}")
    print(f"  CONFLUENCE BLANK DAY STRATEGY — IB Failure + Ichimoku + CPR + VolumeProfile")
    print(f"{'='*85}")
    print(f"  {'Variant':<30} | {'T':>4} | {'WR':>7} | {'Total P&L':>11} | {'Avg':>8} | {'Worst':>10}")
    print(f"  {'-'*83}")
    stats(df,                               'C0: All IB (no filter)')
    print(f"  {'-'*83}")
    stats(df[df['confluence'] >= 1],        'C1: 1+ filters agree')
    stats(df[df['confluence'] >= 2],        'C2: 2+ filters agree')
    stats(df[df['confluence'] >= 3],        'C3: All 3 filters agree')
    print(f"  {'-'*83}")
    stats(df[df['ichi_ok']],                'Ichimoku only filter')
    stats(df[df['cpr_ok']],                 'CPR only filter')
    stats(df[df['vp_ok']],                  'VP only filter')
    print(f"  {'-'*83}")
    stats(df[df['ichi_ok'] & df['cpr_ok']], 'Ichi + CPR')
    stats(df[df['ichi_ok'] & df['vp_ok']],  'Ichi + VP')
    stats(df[df['cpr_ok'] & df['vp_ok']],   'CPR + VP')

    # Year breakdown for best tier
    best_label = 'C2' if len(df[df['confluence']>=2]) >= 50 else 'C1'
    best_df = df[df['confluence'] >= (2 if best_label=='C2' else 1)]
    print(f"\n  Year breakdown ({best_label}):")
    best_df['year'] = best_df['date'].str[:4]
    for yr, g in best_df.groupby('year'):
        print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    exits = best_df['exit_reason'].value_counts()
    print(f"\n  Exit breakdown ({best_label}):")
    for r, c in exits.items():
        print(f"    {r:<12}: {c} ({round(c/len(best_df)*100,1)}%)")

    df.to_csv(f'{OUT_DIR}/110_confluence_trades.csv', index=False)

    # ── Equity chart ──────────────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
    base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
    base_daily.columns = ['date', 'base_pnl']

    def make_line(sub_df, color, label):
        if sub_df.empty: return None
        sub_df = sub_df.copy()
        sub_df['date_dt'] = pd.to_datetime(sub_df['date'].astype(str), format='%Y%m%d')
        gd = sub_df.groupby('date_dt')['pnl'].sum().reset_index()
        gd.columns = ['date', 'strat_pnl']
        all_dt = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(gd['date']))})
        m = all_dt.merge(base_daily, on='date', how='left').merge(gd, on='date', how='left').fillna(0)
        m['comb_eq'] = (m['base_pnl'] + m['strat_pnl']).cumsum()
        tp = round(sub_df['pnl'].sum(), 0)
        return {
            "id": label.replace(' ','_'), "label": f"{label} Rs.{tp:,.0f}",
            "color": color, "seriesType": "line",
            "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                     for d, v in zip(m['date'], m['comb_eq']) if pd.notna(v)],
        }

    lines = []
    for sub, col, lbl in [
        (df,                              '#9e9e9e', 'C0 No filter'),
        (df[df['confluence']>=1],         '#0ea5e9', 'C1 (1+ agree)'),
        (df[df['confluence']>=2],         '#26a69a', 'C2 (2+ agree)'),
        (df[df['confluence']>=3],         '#f59e0b', 'C3 (all agree)'),
        (df[df['ichi_ok']&df['cpr_ok']], '#ab47bc', 'Ichi+CPR'),
    ]:
        ln = make_line(sub, col, lbl)
        if ln: lines.append(ln)

    # base alone
    bs = base_daily.sort_values('date')
    lines.append({
        "id": "base", "label": f"Base Rs.{int(bs['base_pnl'].sum()):,.0f}",
        "color": "#ef5350", "seriesType": "line",
        "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                 for d, v in zip(bs['date'], bs['base_pnl'].cumsum())],
    })

    tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}
    n0 = len(df); wr0 = round(df['win'].mean()*100,1)
    n2 = len(df[df['confluence']>=2]); wr2 = round(df[df['confluence']>=2]['win'].mean()*100,1) if n2 else 0
    send_custom_chart("110_confluence", tv_json,
        title=f"Confluence Blank Day | C0:{n0}t WR{wr0}% → C2:{n2}t WR{wr2}%")

    print(f"\n  Saved → {OUT_DIR}/110_confluence_trades.csv")
    print("Done.")

if __name__ == '__main__':
    main()
