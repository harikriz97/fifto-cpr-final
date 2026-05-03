"""
107_fib_patterns.py — Fibonacci patterns for options selling
=============================================================
Tests 4 strategies across all trading days (5yr):

  S1: Fib Zone (PDH/PDL daily swing)
      61.8% from PDL = resistance → sell CE
      38.2% from PDL = support    → sell PE
      Touch zone between 10:00-13:00 → entry next candle + 2s

  S2: Fib Zone (Weekly 5-day swing)
      Same logic using 5-day H/L instead of PDH/PDL

  S3: M/W Pattern (15M intraday)
      M = two 15M swing highs ±0.4% of each other → second fails → sell CE
      W = two 15M swing lows  ±0.4% of each other → second holds → sell PE
      Entry: pattern completion candle close + 2s

  S4: ABCD Pattern (15M)
      A-B impulse, B-C retracement (38.2–61.8% of AB),
      C-D extension (CD ≈ AB, within 15%)
      Entry at D completion → sell CE (bearish ABCD) or PE (bullish ABCD)
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_ohlc_15m(tks, start='09:15:00', end='14:00:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o','h','l','c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def next_entry_time(candle_time_str):
    """Given 15M candle time (e.g. 10:30:00), return next candle open + 2s"""
    h, m, s = map(int, candle_time_str.split(':'))
    m += 15
    if m >= 60: h += 1; m -= 60
    return f'{h:02d}:{m:02d}:02'

def find_swing_highs_lows(ohlc, n=2):
    """Zigzag: find swing highs/lows using n-bar pivot"""
    highs, lows = [], []
    for i in range(n, len(ohlc) - n):
        if ohlc['h'].iloc[i] == ohlc['h'].iloc[i-n:i+n+1].max():
            highs.append((i, ohlc['h'].iloc[i], ohlc['time'].iloc[i]))
        if ohlc['l'].iloc[i] == ohlc['l'].iloc[i-n:i+n+1].min():
            lows.append((i, ohlc['l'].iloc[i], ohlc['time'].iloc[i]))
    return highs, lows

# ── Per-date worker ───────────────────────────────────────────────────────────
def _worker(args):
    date_str, pdh, pdl, wk_h, wk_l = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    results = []

    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return []

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        expiry = expiries[0]

        # intraday ticks with time filter
        intra = day[(day['time'] >= '10:00:00') & (day['time'] <= '13:00:00')]

        def trade(signal, entry_time, strategy):
            spot_at = day[day['time'] >= entry_time[:8]]
            if spot_at.empty: return
            spot_ref = spot_at.iloc[0]['price']
            strike = get_atm(spot_ref)
            instr  = f'NIFTY{expiry}{strike}{signal}'
            res = simulate_sell(date_str, instr, entry_time)
            if res:
                pnl, reason, ep, xp = res
                results.append({
                    'date': date_str, 'strategy': strategy,
                    'signal': signal, 'entry_time': entry_time,
                    'strike': strike, 'ep': ep, 'xp': xp,
                    'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
                })

        # ── S1: Fib Zone (PDH/PDL) ─────────────────────────────────────────
        if pdh and pdl and pdh > pdl:
            rng = pdh - pdl
            fib618 = r2(pdl + 0.618 * rng)  # resistance zone top
            fib500 = r2(pdl + 0.500 * rng)  # half
            fib382 = r2(pdl + 0.382 * rng)  # support zone bottom

            prev_p = None
            s1_done = False
            for _, row in intra.iterrows():
                p = row['price']
                t = row['time']
                if prev_p is not None and not s1_done:
                    # CE sell: price enters 50-61.8% zone from above (resistance)
                    if prev_p >= fib618 and fib500 <= p <= fib618:
                        entry = t[:5] + ':02'
                        trade('CE', entry, 'S1_fib_pdhl_CE')
                        s1_done = True
                    # PE sell: price enters 38.2-50% zone from below (support)
                    elif prev_p <= fib382 and fib382 <= p <= fib500:
                        entry = t[:5] + ':02'
                        trade('PE', entry, 'S1_fib_pdhl_PE')
                        s1_done = True
                prev_p = p

        # ── S2: Fib Zone (Weekly 5-day swing) ─────────────────────────────
        if wk_h and wk_l and wk_h > wk_l:
            rng = wk_h - wk_l
            fib618 = r2(wk_l + 0.618 * rng)
            fib500 = r2(wk_l + 0.500 * rng)
            fib382 = r2(wk_l + 0.382 * rng)

            prev_p = None
            s2_done = False
            for _, row in intra.iterrows():
                p = row['price']
                t = row['time']
                if prev_p is not None and not s2_done:
                    if prev_p >= fib618 and fib500 <= p <= fib618:
                        entry = t[:5] + ':02'
                        trade('CE', entry, 'S2_fib_weekly_CE')
                        s2_done = True
                    elif prev_p <= fib382 and fib382 <= p <= fib500:
                        entry = t[:5] + ':02'
                        trade('PE', entry, 'S2_fib_weekly_PE')
                        s2_done = True
                prev_p = p

        # ── S3: M/W Pattern (15M) ──────────────────────────────────────────
        ohlc = build_ohlc_15m(tks, end='13:30:00')
        if len(ohlc) >= 6:
            highs, lows = find_swing_highs_lows(ohlc, n=2)
            tol = 0.004  # 0.4% tolerance for double top/bottom

            # M pattern: two swing highs close to each other, second one lower close
            if len(highs) >= 2:
                h1_idx, h1_val, h1_t = highs[-2]
                h2_idx, h2_val, h2_t = highs[-1]
                if h2_idx > h1_idx and abs(h2_val - h1_val) / h1_val < tol:
                    # Second high failed (close below high)
                    h2_close = ohlc['c'].iloc[h2_idx]
                    if h2_close < h2_val * 0.998:  # close is below high = bearish candle
                        et = next_entry_time(h2_t)
                        if '10:00' <= et[:5] <= '13:30':
                            trade('CE', et, 'S3_M_pattern')

            # W pattern: two swing lows close to each other, second holds
            if len(lows) >= 2:
                l1_idx, l1_val, l1_t = lows[-2]
                l2_idx, l2_val, l2_t = lows[-1]
                if l2_idx > l1_idx and abs(l2_val - l1_val) / l1_val < tol:
                    l2_close = ohlc['c'].iloc[l2_idx]
                    if l2_close > l2_val * 1.002:  # close above low = bullish candle
                        et = next_entry_time(l2_t)
                        if '10:00' <= et[:5] <= '13:30':
                            trade('PE', et, 'S3_W_pattern')

        # ── S4: ABCD Pattern (15M) ─────────────────────────────────────────
        if len(ohlc) >= 8:
            highs, lows = find_swing_highs_lows(ohlc, n=2)
            tol_ratio = 0.15  # CD within 15% of AB

            # Bearish ABCD: A=high, B=low, C=high, D=low (D ≈ B level)
            if len(highs) >= 2 and len(lows) >= 2:
                # Try bearish: A(high) B(low) C(high) D(low)
                for a_idx, a_val, a_t in highs[-3:]:
                    for b_idx, b_val, b_t in lows:
                        if b_idx <= a_idx: continue
                        ab = a_val - b_val
                        if ab <= 0: continue
                        for c_idx, c_val, c_t in highs:
                            if c_idx <= b_idx: continue
                            bc_ratio = (c_val - b_val) / ab
                            if not (0.382 <= bc_ratio <= 0.618): continue
                            # D should be at ≈ B level (CD ≈ AB)
                            for d_idx, d_val, d_t in lows:
                                if d_idx <= c_idx: continue
                                cd = c_val - d_val
                                if abs(cd - ab) / ab > tol_ratio: continue
                                # D complete → sell CE (price reversal up expected)
                                # Wait — ABCD bearish means price falls to D, then we sell CE?
                                # Actually at D in bearish ABCD, price should bounce → sell PE
                                # But for CE: bearish ABCD → D is a lower high/resistance → sell CE
                                # Let's go: if D is near a high (bearish ABCD), sell CE
                                et = next_entry_time(d_t)
                                if '10:00' <= et[:5] <= '13:30':
                                    trade('CE', et, 'S4_ABCD_bear')
                                break
                            break
                        break

                # Bullish ABCD: A(low) B(high) C(low) D(high)
                for a_idx, a_val, a_t in lows[-3:]:
                    for b_idx, b_val, b_t in highs:
                        if b_idx <= a_idx: continue
                        ab = b_val - a_val
                        if ab <= 0: continue
                        for c_idx, c_val, c_t in lows:
                            if c_idx <= b_idx: continue
                            bc_ratio = (b_val - c_val) / ab
                            if not (0.382 <= bc_ratio <= 0.618): continue
                            for d_idx, d_val, d_t in highs:
                                if d_idx <= c_idx: continue
                                cd = d_val - c_val
                                if abs(cd - ab) / ab > tol_ratio: continue
                                et = next_entry_time(d_t)
                                if '10:00' <= et[:5] <= '13:30':
                                    trade('PE', et, 'S4_ABCD_bull')
                                break
                            break
                        break

    except Exception as e:
        pass

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building daily OHLC for Fib levels...")
    t0 = time.time()
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
    seed_idx  = max(0, all_dates.index(dates_5yr[0]) - 10)
    scan_dates = all_dates[seed_idx:]

    daily_rows = []
    for d in scan_dates:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None: continue
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
        if len(day) < 2: continue
        daily_rows.append({'date': d, 'h': day['price'].max(), 'l': day['price'].min()})

    df_d = pd.DataFrame(daily_rows)
    df_d['pdh'] = df_d['h'].shift(1)
    df_d['pdl'] = df_d['l'].shift(1)
    df_d['wk_h'] = df_d['h'].rolling(5).max().shift(1)
    df_d['wk_l'] = df_d['l'].rolling(5).min().shift(1)
    df_d = df_d.dropna()

    level_map = {row['date']: (row['pdh'], row['pdl'], row['wk_h'], row['wk_l'])
                 for _, row in df_d.iterrows()}

    print(f"  {len(df_d)} daily candles in {time.time()-t0:.0f}s")

    args_list = []
    for d in dates_5yr:
        if d in level_map:
            pdh, pdl, wk_h, wk_l = level_map[d]
            args_list.append((d, pdh, pdl, wk_h, wk_l))

    print(f"\nRunning {len(args_list)} days in parallel...")
    t0 = time.time()
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(_worker, args_list)
    elapsed = time.time() - t0

    all_trades = [t for day_trades in raw for t in day_trades]
    if not all_trades:
        print("No trades found.")
        return

    df = pd.DataFrame(all_trades)
    df['pnl'] = df['pnl'].round(2)

    # ── Summary per strategy ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FIB PATTERNS BACKTEST — ALL DAYS 5yr ({elapsed:.0f}s)")
    print(f"{'='*72}")
    print(f"  {'Strategy':<24} | {'T':>4} | {'WR':>6} | {'Total P&L':>11} | {'Avg':>7}")
    print(f"  {'-'*70}")

    strategies = ['S1_fib_pdhl_CE','S1_fib_pdhl_PE','S2_fib_weekly_CE','S2_fib_weekly_PE',
                  'S3_M_pattern','S3_W_pattern','S4_ABCD_bear','S4_ABCD_bull']

    summary = []
    for s in strategies:
        g = df[df['strategy']==s]
        if g.empty: print(f"  {s:<24} | {'0':>4}"); continue
        wr = g['win'].mean()*100
        tp = g['pnl'].sum()
        ap = g['pnl'].mean()
        summary.append({'strategy':s,'trades':len(g),'wr':wr,'total_pnl':tp,'avg_pnl':ap})
        print(f"  {s:<24} | {len(g):>4} | {wr:>5.1f}% | Rs.{tp:>8,.0f} | Rs.{ap:>5,.0f}")

    # CE vs PE grouped
    print(f"\n  {'Grouped':<24} | {'T':>4} | {'WR':>6} | {'Total P&L':>11} | {'Avg':>7}")
    print(f"  {'-'*70}")
    for grp_label, grp_strats in [
        ('ALL S1 Fib PDH/PDL', ['S1_fib_pdhl_CE','S1_fib_pdhl_PE']),
        ('ALL S2 Fib Weekly',  ['S2_fib_weekly_CE','S2_fib_weekly_PE']),
        ('ALL S3 M+W pattern', ['S3_M_pattern','S3_W_pattern']),
        ('ALL S4 ABCD',        ['S4_ABCD_bear','S4_ABCD_bull']),
    ]:
        g = df[df['strategy'].isin(grp_strats)]
        if g.empty: continue
        wr = g['win'].mean()*100
        tp = g['pnl'].sum()
        ap = g['pnl'].mean()
        print(f"  {grp_label:<24} | {len(g):>4} | {wr:>5.1f}% | Rs.{tp:>8,.0f} | Rs.{ap:>5,.0f}")

    # Overall
    print(f"\n  {'TOTAL ALL STRATEGIES':<24} | {len(df):>4} | "
          f"{df['win'].mean()*100:>5.1f}% | Rs.{df['pnl'].sum():>8,.0f} | Rs.{df['pnl'].mean():>5,.0f}")

    # ── Save & equity chart ─────────────────────────────────────────────────
    df.to_csv(f'{OUT_DIR}/107_fib_trades.csv', index=False)

    # Combined equity vs base
    df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
    base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
    base_daily.columns = ['date', 'base_pnl']

    lines = []
    colors = ['#26a69a','#0ea5e9','#f59e0b','#ef5350','#ab47bc','#ff7043','#66bb6a','#ec407a']
    for i, (grp_label, grp_strats) in enumerate([
        ('S1 Fib PDH/PDL', ['S1_fib_pdhl_CE','S1_fib_pdhl_PE']),
        ('S2 Fib Weekly',  ['S2_fib_weekly_CE','S2_fib_weekly_PE']),
        ('S3 M+W Pattern', ['S3_M_pattern','S3_W_pattern']),
        ('S4 ABCD',        ['S4_ABCD_bear','S4_ABCD_bull']),
    ]):
        g = df[df['strategy'].isin(grp_strats)]
        if g.empty: continue
        gd = g.groupby('date_dt')['pnl'].sum().reset_index()
        gd.columns = ['date', 'pnl']
        all_dt = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(gd['date']))})
        m = all_dt.merge(base_daily, on='date', how='left').merge(gd.rename(columns={'pnl':'strat_pnl'}), on='date', how='left')
        m['base_pnl']  = m['base_pnl'].fillna(0)
        m['strat_pnl'] = m['strat_pnl'].fillna(0)
        m['comb_eq']   = (m['base_pnl'] + m['strat_pnl']).cumsum()
        tp = round(g['pnl'].sum(), 0)
        lines.append({
            "id": grp_label.replace(' ','_'),
            "label": f"{grp_label} Rs.{tp:,.0f}",
            "color": colors[i],
            "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                     for d, v in zip(m['date'], m['comb_eq']) if pd.notna(v)],
            "seriesType": "line",
        })

    # Base line
    base_daily_s = base_daily.sort_values('date')
    base_eq = base_daily_s['base_pnl'].cumsum()
    lines.append({
        "id": "base", "label": f"Base Rs.{int(base_daily['base_pnl'].sum()):,.0f}",
        "color": "#9e9e9e",
        "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                 for d, v in zip(base_daily_s['date'], base_eq) if pd.notna(v)],
        "seriesType": "line",
    })

    tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}
    send_custom_chart("107_fib_patterns", tv_json,
                      title=f"Fib Patterns — S1 PDH/PDL | S2 Weekly | S3 M/W | S4 ABCD | All Days 5yr")

    print(f"\n  Saved → {OUT_DIR}/107_fib_trades.csv")
    print("\nDone.")

if __name__ == '__main__':
    main()
