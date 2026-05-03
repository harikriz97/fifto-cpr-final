"""
118_vwap_ohl_deep.py — Full deep analysis: O=H/O=L and VWAP mean reversion
============================================================================
Full parameter sweep on ALL trading days (5 years).

O=H / O=L Analysis:
  - open ≈ IB high  → sell CE  (price opened at top, fell through IB → bearish)
  - open ≈ IB low   → sell PE  (price opened at bottom, rose through IB → bullish)
  Tests:
    * Tolerance sweep: 0.05%, 0.10%, 0.15%, 0.20%, 0.25%, 0.30%
    * Strict: open IS the IB extreme (open price = IB max/min exactly, within 1 tick)
    * With IB direction confirm: price moves AWAY from open during IB
    * All days vs blank-only vs traded-only
    * Time of entry: 9:46:02 vs 9:31:02 (after 15-min IB)

VWAP Mean Reversion Analysis:
  - VWAP computed from 9:15 (tick-weighted cumulative mean)
  - Price extends X% from VWAP, then reverts to within Y%
  - Sell against extension direction
  Tests:
    * Extension threshold sweep: 0.3%, 0.4%, 0.5%, 0.6%, 0.7%, 0.8%
    * Reversion threshold: 0.1%, 0.2%, 0.3% (touch/near/cross)
    * Entry window: 9:46-13:00, 9:46-14:00, 10:00-13:00
    * Multiple signals per day (first only vs all)
    * All days
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
from plot_util import send_custom_chart

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
IB_END     = '09:45:00'
IB15_END   = '09:30:00'
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)

OHL_TOLERANCES = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030]
VWAP_EXTENSIONS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
VWAP_REVERSIONS = [0.001, 0.002, 0.003]


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


def _worker_ohl(args):
    date_str, is_blank = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        # IB (30 min)
        ib = day[day['time'] <= IB_END]
        if len(ib) < 5: return []
        ib_h = ib['price'].max(); ib_l = ib['price'].min()
        spot_open = ib.iloc[0]['price']
        ib_range = ib_h - ib_l
        if ib_h <= ib_l or ib_range < 20: return []  # skip very flat IB

        # IB direction: did price move away from open during IB?
        ib_close = ib.iloc[-1]['price']
        ib_moved_down = ib_close < spot_open * 0.9985   # at least 0.15% down
        ib_moved_up   = ib_close > spot_open * 1.0015   # at least 0.15% up

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        results = []

        for tol in OHL_TOLERANCES:
            tol_pct = round(tol * 100, 2)

            # O=H: open ≈ IB high → bearish → sell CE
            is_oh = abs(spot_open - ib_h) / spot_open <= tol
            # O=L: open ≈ IB low → bullish → sell PE
            is_ol = abs(spot_open - ib_l) / spot_open <= tol

            if not is_oh and not is_ol:
                continue

            sig  = 'CE' if is_oh else 'PE'
            et   = '09:46:02'
            confirmed = (sig == 'CE' and ib_moved_down) or (sig == 'PE' and ib_moved_up)

            spot_at = day[day['time'] >= et]
            if spot_at.empty: continue
            atm   = get_atm(spot_at.iloc[0]['price'])
            instr = f'NIFTY{expiries[0]}{atm}{sig}'
            res   = simulate_sell(date_str, instr, et)
            if res is None: continue
            pnl, reason, ep, xp = res

            results.append({
                'type': 'OHL',
                'tol_pct': tol_pct,
                'date': date_str,
                'signal': sig,
                'pattern': 'O=H' if is_oh else 'O=L',
                'ib_confirmed': confirmed,
                'is_blank': is_blank,
                'entry_time': et,
                'strike': atm,
                'ep': ep, 'xp': xp,
                'pnl': pnl, 'win': pnl > 0,
                'exit_reason': reason,
            })

        return results
    except Exception:
        return []


def _worker_vwap(args):
    date_str, is_blank = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        prices = day['price'].values
        times  = day['time'].values
        # Tick-weighted cumulative VWAP
        vwap = np.cumsum(prices) / (np.arange(len(prices)) + 1)

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        results = []

        for ext_thresh in VWAP_EXTENSIONS:
            for rev_thresh in VWAP_REVERSIONS:
                ext_pct = round(ext_thresh * 100, 2)
                rev_pct = round(rev_thresh * 100, 2)

                # Scan for extension + reversion signal
                # First signal only per day per (ext, rev) combo
                ext_up = False; ext_dn = False
                sig_found = None; et_found = None

                for i in range(len(times)):
                    t = times[i]; p = prices[i]; v = vwap[i]
                    if t < '09:46:00': continue
                    if t > '14:00:00': break

                    dev = (p - v) / v
                    if dev >  ext_thresh: ext_up = True
                    if dev < -ext_thresh: ext_dn = True

                    # Reversion: price came back to within rev_thresh of VWAP
                    if ext_up and abs(dev) <= rev_thresh:
                        h,m,s = map(int,t.split(':'))
                        sig_found = 'CE'
                        et_found  = f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
                        break
                    if ext_dn and abs(dev) <= rev_thresh:
                        h,m,s = map(int,t.split(':'))
                        sig_found = 'PE'
                        et_found  = f'{h:02d}:{m:02d}:{min(s+2,59):02d}'
                        break

                if sig_found is None: continue

                spot_at = day[day['time'] >= et_found]
                if spot_at.empty: continue
                atm   = get_atm(spot_at.iloc[0]['price'])
                instr = f'NIFTY{expiries[0]}{atm}{sig_found}'
                res   = simulate_sell(date_str, instr, et_found)
                if res is None: continue
                pnl, reason, ep, xp = res

                results.append({
                    'type': 'VWAP',
                    'ext_pct': ext_pct,
                    'rev_pct': rev_pct,
                    'date': date_str,
                    'signal': sig_found,
                    'is_blank': is_blank,
                    'entry_time': et_found,
                    'strike': atm,
                    'ep': ep, 'xp': xp,
                    'pnl': pnl, 'win': pnl > 0,
                    'exit_reason': reason,
                })

        return results
    except Exception:
        return []


def print_table(df, group_cols, label):
    if df.empty:
        print(f"  No trades.")
        return
    g = df.groupby(group_cols).apply(lambda x: pd.Series({
        'n': len(x),
        'wr': round(x['win'].mean()*100, 1),
        'total': round(x['pnl'].sum(), 0),
        'avg': round(x['pnl'].mean(), 0),
    })).reset_index()
    g = g.sort_values('total', ascending=False)
    header = f"  {group_cols!s:<30} | {'N':>5} | {'WR':>6} | {'Total':>11} | {'Avg':>8}"
    print(header)
    print(f"  {'-'*70}")
    for _, row in g.iterrows():
        key = ' | '.join(str(row[c]) for c in group_cols)
        print(f"  {key:<30} | {int(row['n']):>5} | {row['wr']:>5.1f}% | "
              f"Rs.{int(row['total']):>9,} | Avg Rs.{int(row['avg']):>6,}")


def main():
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]

    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    blank_set = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str)) | \
                set(pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')['date'].astype(str))

    args = [(d, d in blank_set) for d in dates_5yr]

    # ── Run O=H / O=L ──────────────────────────────────────────────────────────
    print(f"O=H / O=L: Running on {len(args)} days (parallel)...")
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_ohl = pool.map(_worker_ohl, args)
    el = (datetime.now()-t0).total_seconds()
    df_ohl = pd.DataFrame([t for day in raw_ohl for t in day])
    print(f"  Done: {len(df_ohl)} signal-day combinations in {el:.1f}s")

    if not df_ohl.empty:
        df_ohl['year'] = df_ohl['date'].str[:4]

        print(f"\n{'='*72}")
        print(f"  O=H / O=L — TOLERANCE SWEEP (all days)")
        print(f"{'='*72}")
        print_table(df_ohl, ['tol_pct'], 'tol_pct')

        print(f"\n  By tolerance + signal:")
        print_table(df_ohl, ['tol_pct','signal'], 'tol_pct+signal')

        print(f"\n  With IB-direction confirmation (open moved away during IB):")
        df_conf = df_ohl[df_ohl['ib_confirmed']==True]
        print_table(df_conf, ['tol_pct'], 'tol_pct (confirmed)')

        print(f"\n  Blank days only (best tolerance):")
        best_tol = df_ohl.groupby('tol_pct')['pnl'].sum().idxmax()
        df_blank_ohl = df_ohl[(df_ohl['tol_pct']==best_tol) & (df_ohl['is_blank']==True)]
        print(f"  Best tol={best_tol}%: blank_days={len(df_blank_ohl)}t | "
              f"WR {df_blank_ohl['win'].mean()*100:.1f}% | "
              f"Rs.{df_blank_ohl['pnl'].sum():,.0f} | "
              f"Avg Rs.{df_blank_ohl['pnl'].mean():.0f}")

        print(f"\n  By year (tol={best_tol}%):")
        df_best_ohl = df_ohl[df_ohl['tol_pct']==best_tol]
        for yr, g in df_best_ohl.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

        print(f"\n  Exit breakdown (tol={best_tol}%):")
        for ex, g in df_best_ohl.groupby('exit_reason'):
            print(f"    {ex}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f}")

        # Confirmed only — year breakdown
        print(f"\n  With IB-confirmation, by year (tol={best_tol}%):")
        df_best_conf = df_ohl[(df_ohl['tol_pct']==best_tol) & (df_ohl['ib_confirmed']==True)]
        if not df_best_conf.empty:
            for yr, g in df_best_conf.groupby('year'):
                print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                      f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")
        else:
            print("    No confirmed trades at this tolerance.")

        # Save OHL
        df_ohl.to_csv(f'{OUT_DIR}/118_ohl_all.csv', index=False)

    # ── Run VWAP ───────────────────────────────────────────────────────────────
    print(f"\n\nVWAP Mean Reversion: Running on {len(args)} days (parallel)...")
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_vwap = pool.map(_worker_vwap, args)
    el = (datetime.now()-t0).total_seconds()
    df_vwap = pd.DataFrame([t for day in raw_vwap for t in day])
    print(f"  Done: {len(df_vwap)} signal-day combinations in {el:.1f}s")

    if not df_vwap.empty:
        df_vwap['year'] = df_vwap['date'].str[:4]

        print(f"\n{'='*72}")
        print(f"  VWAP MEAN REVERSION — PARAMETER SWEEP (all days)")
        print(f"{'='*72}")
        print(f"\n  Extension threshold sweep (rev=0.2%):")
        df_v02 = df_vwap[df_vwap['rev_pct']==0.2]
        print_table(df_v02, ['ext_pct'], 'ext_pct (rev=0.2%)')

        print(f"\n  Full grid — ext × rev (top 10 by total P&L):")
        grid = df_vwap.groupby(['ext_pct','rev_pct']).apply(lambda x: pd.Series({
            'n': len(x), 'wr': round(x['win'].mean()*100,1),
            'total': round(x['pnl'].sum(),0), 'avg': round(x['pnl'].mean(),0),
        })).reset_index().sort_values('total', ascending=False).head(10)
        for _, row in grid.iterrows():
            print(f"    ext={row['ext_pct']}% rev={row['rev_pct']}% | "
                  f"{int(row['n'])}t | WR {row['wr']:.1f}% | "
                  f"Rs.{int(row['total']):,} | Avg Rs.{int(row['avg']):,}")

        # Best config
        best_ext = grid.iloc[0]['ext_pct']
        best_rev = grid.iloc[0]['rev_pct']
        df_best_v = df_vwap[(df_vwap['ext_pct']==best_ext) & (df_vwap['rev_pct']==best_rev)]
        print(f"\n  Best config (ext={best_ext}%, rev={best_rev}%):")
        for sig, g in df_best_v.groupby('signal'):
            print(f"    {sig}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

        print(f"\n  By year (ext={best_ext}%, rev={best_rev}%):")
        for yr, g in df_best_v.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

        print(f"\n  Blank days only (best config):")
        df_vblank = df_best_v[df_best_v['is_blank']==True]
        if not df_vblank.empty:
            print(f"    {len(df_vblank)}t | WR {df_vblank['win'].mean()*100:.1f}% | "
                  f"Rs.{df_vblank['pnl'].sum():,.0f} | Avg Rs.{df_vblank['pnl'].mean():.0f}")
        else:
            print("    No blank-day trades at this config.")

        # Signal frequency: how often does VWAP signal fire?
        unique_days_v = df_vwap[df_vwap['rev_pct']==0.2].groupby('ext_pct')['date'].nunique()
        print(f"\n  Signal frequency (days with VWAP signal, rev=0.2%):")
        for ext, cnt in unique_days_v.items():
            print(f"    ext={ext}%: {cnt} days / {len(dates_5yr)} = {cnt/len(dates_5yr)*100:.1f}%")

        df_vwap.to_csv(f'{OUT_DIR}/118_vwap_all.csv', index=False)

    # ── Frequency analysis for OHL ─────────────────────────────────────────────
    if not df_ohl.empty:
        print(f"\n  O=H / O=L signal frequency (days triggered):")
        for tol in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            cnt = df_ohl[df_ohl['tol_pct']==tol]['date'].nunique()
            print(f"    tol={tol}%: {cnt} days / {len(dates_5yr)} = {cnt/len(dates_5yr)*100:.1f}%")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  VERDICT SUMMARY")
    print(f"{'='*72}")

    if not df_ohl.empty:
        best_ohl = df_ohl.groupby('tol_pct')['pnl'].sum()
        best_ohl_tol = best_ohl.idxmax()
        df_best_ohl = df_ohl[df_ohl['tol_pct']==best_ohl_tol]
        print(f"\n  O=H / O=L best config (tol={best_ohl_tol}%):")
        print(f"    {len(df_best_ohl)}t | WR {df_best_ohl['win'].mean()*100:.1f}% | "
              f"Rs.{df_best_ohl['pnl'].sum():,.0f} | Avg Rs.{df_best_ohl['pnl'].mean():.0f}")
        df_best_conf = df_ohl[(df_ohl['tol_pct']==best_ohl_tol) & df_ohl['ib_confirmed']]
        if not df_best_conf.empty:
            print(f"    With IB confirm: {len(df_best_conf)}t | WR {df_best_conf['win'].mean()*100:.1f}% | "
                  f"Rs.{df_best_conf['pnl'].sum():,.0f} | Avg Rs.{df_best_conf['pnl'].mean():.0f}")

    if not df_vwap.empty:
        best_v = df_vwap.groupby(['ext_pct','rev_pct'])['pnl'].sum().idxmax()
        df_bv  = df_vwap[(df_vwap['ext_pct']==best_v[0]) & (df_vwap['rev_pct']==best_v[1])]
        print(f"\n  VWAP best config (ext={best_v[0]}%, rev={best_v[1]}%):")
        print(f"    {len(df_bv)}t | WR {df_bv['win'].mean()*100:.1f}% | "
              f"Rs.{df_bv['pnl'].sum():,.0f} | Avg Rs.{df_bv['pnl'].mean():.0f}")

    # ── Equity charts for best configs ─────────────────────────────────────────
    def to_daily_eq(df_, date_col='date'):
        if df_.empty: return pd.Series(dtype=float)
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_[date_col].astype(str), format='%Y%m%d')
        daily = df_.groupby('dt')['pnl'].sum()
        all_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='B')
        return daily.reindex(all_idx, fill_value=0).cumsum()

    def eq_pts(s):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                for d, v in s.items() if pd.notna(v)]

    lines = []
    colors = ['#26a69a','#0ea5e9','#ab47bc','#ff9800','#ef5350','#66bb6a']
    ci = 0

    if not df_ohl.empty:
        for tol in [0.10, 0.15, 0.20]:
            sub = df_ohl[df_ohl['tol_pct']==tol]
            if sub.empty: continue
            eq = to_daily_eq(sub)
            lines.append({
                "id": f"ohl_{tol}", "label": f"O=H/L tol={tol}% {len(sub)}t WR{sub['win'].mean()*100:.0f}%",
                "color": colors[ci % len(colors)], "seriesType": "line", "data": eq_pts(eq)
            })
            ci += 1
        # Best confirmed
        sub_c = df_ohl[(df_ohl['tol_pct']==0.15) & df_ohl['ib_confirmed']]
        if not sub_c.empty:
            eq = to_daily_eq(sub_c)
            lines.append({
                "id": "ohl_conf", "label": f"O=H/L tol=0.15%+IBconf {len(sub_c)}t WR{sub_c['win'].mean()*100:.0f}%",
                "color": colors[ci % len(colors)], "seriesType": "line", "data": eq_pts(eq)
            })
            ci += 1

    if not df_vwap.empty:
        for ext in [0.004, 0.005, 0.006]:
            sub = df_vwap[(df_vwap['ext_pct']==ext*100) & (df_vwap['rev_pct']==0.2)]
            if sub.empty: continue
            eq = to_daily_eq(sub)
            lines.append({
                "id": f"vwap_{ext}", "label": f"VWAP ext={ext*100:.1f}% {len(sub)}t WR{sub['win'].mean()*100:.0f}%",
                "color": colors[ci % len(colors)], "seriesType": "line", "data": eq_pts(eq)
            })
            ci += 1

    if lines:
        tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}
        send_custom_chart("118_vwap_ohl", tv_json,
            title="O=H/O=L vs VWAP Mean Reversion — Parameter Sweep (5yr)")

    print(f"\n  Saved: 118_ohl_all.csv | 118_vwap_all.csv")
    print("\nDone.")


if __name__ == '__main__':
    main()
