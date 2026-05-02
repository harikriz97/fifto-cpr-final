"""
103_parallel_backtest.py — Parallel backtest: Combined CRT + MRC blank day strategy (ATM 30%)
==============================================================================================
Runs in parallel across all trading dates (5yr).
Signal lookup from pre-computed CSVs:
  - CRT blank days  (91_crt_ltf_D.csv, is_blank=True)  → CE sell ATM 30%
  - MRC unique blank (100_mrc_trades.csv, is_blank=True, not in CRT) → CE/PE sell ATM 30%
  - Base strategy days → no trade (counted as 0 P&L)

Usage:
    python3 103_parallel_backtest.py                   # full 5yr
    python3 103_parallel_backtest.py 20240101 20241231 # specific range
    python3 103_parallel_backtest.py --last 50         # last N dates
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30

def r2(v): return round(float(v), 2)

def get_atm(spot):
    return int(round(spot / STRIKE_INT) * STRIKE_INT)

def simulate_sell(date_str, instrument, entry_time):
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - TGT_PCT)); hsl = r2(ep * 1.00 + ep); sl = hsl; md = 0.0
    # rebuild hard SL correctly
    hsl = r2(ep * 2.0)  # 100% up from ep (unreachable hard cap)
    sl  = r2(ep * (1 + 1.00))  # initial SL = 100% up
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


def _worker(args):
    date_str, signal_type, signal_dir, entry_time = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        spot = load_spot_data(date_str, 'NIFTY')
        if spot is None: return None
        spot_at = spot[spot['time'] >= entry_time[:8]]
        if spot_at.empty: return None
        spot_ref = spot_at.iloc[0]['price']

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return None
        strike = get_atm(spot_ref)
        instr  = f'NIFTY{expiries[0]}{strike}{signal_dir}'

        res = simulate_sell(date_str, instr, entry_time)
        if res is None: return None
        pnl, reason, ep, xp = res

        return {
            'date': date_str,
            'signal_type': signal_type,
            'signal': signal_dir,
            'entry_time': entry_time,
            'strike': strike,
            'instrument': instr,
            'ep': ep,
            'xp': xp,
            'pnl': pnl,
            'win': pnl > 0,
            'exit_reason': reason,
        }
    except Exception as e:
        print(f"[{date_str}] ERROR: {str(e)[:80]}")
        return None


def main():
    # ── Parse CLI args ────────────────────────────────────────────────────────
    all_dates = list_trading_dates()

    if len(sys.argv) == 1:
        dates_to_process = all_dates
        from_date, to_date = dates_to_process[0], dates_to_process[-1]
    elif len(sys.argv) == 3 and sys.argv[1] == '--last':
        n = int(sys.argv[2])
        dates_to_process = all_dates[-n:]
        from_date, to_date = dates_to_process[0], dates_to_process[-1]
    elif len(sys.argv) >= 3:
        from_date, to_date = sys.argv[1], sys.argv[2]
        dates_to_process = [d for d in all_dates if from_date <= d <= to_date]
    else:
        print("Usage: python3 103_parallel_backtest.py [from_date to_date | --last N]")
        sys.exit(1)

    # ── Load pre-computed signals ──────────────────────────────────────────────
    print("Loading signal lookup tables...")
    crt_raw = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    crt_blank = crt_raw[crt_raw['is_blank']==True].copy()
    crt_blank['date'] = crt_blank['date'].astype(str)
    crt_sig = {row['date']: row['entry_time'] for _, row in crt_blank.iterrows()}

    mrc_raw = pd.read_csv(f'{OUT_DIR}/100_mrc_trades.csv')
    mrc_blank = mrc_raw[mrc_raw['is_blank']==True].copy()
    mrc_blank['date'] = mrc_blank['date'].astype(str)
    crt_dates = set(crt_sig.keys())
    mrc_unique = mrc_blank[~mrc_blank['date'].isin(crt_dates)].copy()
    mrc_sig = {row['date']: (row['entry_time'], row['signal'])
               for _, row in mrc_unique.iterrows()}

    # ── Build args list — only dates with a signal ────────────────────────────
    args_list = []
    for d in dates_to_process:
        if d in crt_sig:
            args_list.append((d, 'CRT', 'CE', crt_sig[d]))
        elif d in mrc_sig:
            et, sig = mrc_sig[d]
            args_list.append((d, 'MRC', sig, et))

    print(f"\n{'='*65}")
    print(f"PARALLEL BACKTEST — CRT + MRC Blank Days (ATM {int(TGT_PCT*100)}%)")
    print(f"{'='*65}")
    print(f"Date range:   {from_date} → {to_date}")
    print(f"Total dates:  {len(dates_to_process)}")
    print(f"Signal dates: {len(args_list)}  "
          f"(CRT: {sum(1 for a in args_list if a[1]=='CRT')} | "
          f"MRC: {sum(1 for a in args_list if a[1]=='MRC')})")
    print(f"{'='*65}\n")

    if not args_list:
        print("No signal dates in range.")
        return

    n_workers = min(16, cpu_count() or 4, len(args_list))
    t0 = datetime.now()

    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker, args_list)

    elapsed = (datetime.now() - t0).total_seconds()
    results = [r for r in results if r is not None]

    if not results:
        print("No successful trades.")
        return

    df = pd.DataFrame(results).sort_values('date')
    df['pnl'] = df['pnl'].round(2)

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs('data/consolidated', exist_ok=True)
    out_csv = f'data/consolidated/103_crt_mrc_atm30_{from_date}_{to_date}.csv'
    df.to_csv(out_csv, index=False)

    # ── Stats ──────────────────────────────────────────────────────────────────
    total   = len(df)
    wins    = df['win'].sum()
    wr      = round(wins / total * 100, 2)
    total_pnl = round(df['pnl'].sum(), 2)
    avg_pnl   = round(df['pnl'].mean(), 2)

    crt_df = df[df['signal_type']=='CRT']
    mrc_df = df[df['signal_type']=='MRC']

    print(f"\n{'='*65}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"Total trades: {total} | WR: {wr}% | P&L: Rs.{total_pnl:,.0f} | Avg: Rs.{avg_pnl:,.0f}")
    print(f"  CRT : {len(crt_df)}t | WR {crt_df['win'].mean()*100:.1f}% | "
          f"Rs.{crt_df['pnl'].sum():,.0f} | Avg Rs.{crt_df['pnl'].mean():.0f}")
    print(f"  MRC : {len(mrc_df)}t | WR {mrc_df['win'].mean()*100:.1f}% | "
          f"Rs.{mrc_df['pnl'].sum():,.0f} | Avg Rs.{mrc_df['pnl'].mean():.0f}")

    exits = df['exit_reason'].value_counts()
    print(f"\nExit breakdown:")
    for reason, cnt in exits.items():
        print(f"  {reason:<12}: {cnt} ({round(cnt/total*100,1)}%)")

    if len(df) > 0:
        best  = df.loc[df['pnl'].idxmax()]
        worst = df.loc[df['pnl'].idxmin()]
        print(f"\nBest  day: {best['date']}  Rs.{best['pnl']:,.0f}")
        print(f"Worst day: {worst['date']} Rs.{worst['pnl']:,.0f}")

    print(f"\nSaved: {out_csv}")


if __name__ == '__main__':
    main()
