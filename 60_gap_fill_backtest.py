"""
60_gap_fill_backtest.py
Gap Fill Strategy: NIFTY opens with a gap → sell ATM option expecting gap to fill
- Gap up  > threshold → sell ATM CE
- Gap down > threshold → sell ATM PE
- Entry at 09:16:02 (next candle open + 2s, rule 8)
- Target: EP * 0.75 (25% profit for seller)
- Hard SL: EP * 2.0 (option doubles)
- EOD exit: 15:20:00
"""
import sys, os, re
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart, plot_equity
from my_util import list_trading_dates, load_spot_data

DATA_ROOT  = os.environ['INTER_SERVER_DATA_PATH']
LOT_SIZE   = 75
TARGET_PCT = 0.25   # seller books 25% profit when option falls 25%
SL_MULT    = 2.0    # option doubles = hard SL

def r2(v): return round(float(v), 2)

# ─────────────────────────────────────────────
# Option file helpers
# ─────────────────────────────────────────────
def find_option_file(date_str, strike, opt_type, date_folder):
    """Find NIFTY option file for given date/strike/type (nearest expiry)."""
    pattern = re.compile(
        rf'^NIFTY(\d{{6}}){strike}({opt_type})\.csv$', re.IGNORECASE
    )
    matches = []
    for f in os.listdir(date_folder):
        m = pattern.match(f)
        if m:
            expiry_yymmdd = m.group(1)
            # parse expiry
            try:
                exp_dt = datetime.strptime('20' + expiry_yymmdd, '%Y%m%d')
                trade_dt = datetime.strptime(date_str, '%Y%m%d')
                dte = (exp_dt - trade_dt).days
                if dte >= 0:
                    matches.append((dte, f))
            except Exception:
                continue
    if not matches:
        return None
    matches.sort()
    return os.path.join(date_folder, matches[0][1])


def load_option(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=None, names=['date','time','price','vol','oi'])
    return df


def backtest_day(date_str, prev_date_str, gap_threshold):
    date_folder = os.path.join(DATA_ROOT, date_str)
    if not os.path.isdir(date_folder):
        return None

    # ── spot data ──
    spot = load_spot_data(date_str, 'NIFTY')
    prev_spot = load_spot_data(prev_date_str, 'NIFTY')
    if spot is None or spot.empty or prev_spot is None or prev_spot.empty:
        return None

    prev_close  = float(prev_spot['price'].iloc[-1])
    today_first = spot[spot['time'] >= '09:15:00']
    if today_first.empty:
        return None
    today_open = float(today_first['price'].iloc[0])

    gap_pct = r2((today_open - prev_close) / prev_close * 100)

    if abs(gap_pct) < gap_threshold:
        return None   # no signal

    direction = 'sell_ce' if gap_pct > 0 else 'sell_pe'
    opt_type  = 'CE' if direction == 'sell_ce' else 'PE'

    # ── ATM strike at 9:16 ──
    spot_916 = spot[spot['time'] >= '09:16:00']
    if spot_916.empty:
        return None
    spot_price = float(spot_916['price'].iloc[0])
    atm_strike = int(round(spot_price / 50) * 50)

    # ── load option ──
    opt_path = find_option_file(date_str, atm_strike, opt_type, date_folder)
    if opt_path is None:
        # try ±50
        for adj in [50, -50, 100, -100]:
            opt_path = find_option_file(date_str, atm_strike + adj, opt_type, date_folder)
            if opt_path:
                atm_strike += adj
                break
    if opt_path is None:
        return None

    opt_tks = load_option(opt_path)
    if opt_tks is None or opt_tks.empty:
        return None

    # ── entry at 09:16:02 ──
    entry_tks = opt_tks[opt_tks['time'] >= '09:16:02']
    if entry_tks.empty:
        return None
    ep = r2(float(entry_tks['price'].iloc[0]))
    entry_time = entry_tks.iloc[0]['time']

    if ep <= 0:
        return None

    target   = r2(ep * (1 - TARGET_PCT))
    hard_sl  = r2(ep * SL_MULT)

    # ── simulate tick by tick ──
    after_entry = opt_tks[opt_tks['time'] >= entry_time].copy()
    eod_tks     = opt_tks[opt_tks['time'] <= '15:20:00']

    exit_price  = None
    exit_time   = None
    exit_reason = 'eod'

    for _, row in after_entry.iterrows():
        cp = float(row['price'])
        t  = row['time']
        if t > '15:20:00':
            break
        if cp <= target:
            exit_price  = r2(cp)
            exit_time   = t
            exit_reason = 'target'
            break
        if cp >= hard_sl:
            exit_price  = r2(cp)
            exit_time   = t
            exit_reason = 'hard_sl'
            break

    if exit_price is None:
        eod = opt_tks[opt_tks['time'] <= '15:20:00']
        if eod.empty:
            return None
        exit_price  = r2(float(eod['price'].iloc[-1]))
        exit_time   = eod.iloc[-1]['time']

    pnl = r2((ep - exit_price) * LOT_SIZE)

    return {
        'date':        date_str,
        'gap_pct':     gap_pct,
        'direction':   direction,
        'opt_type':    opt_type,
        'strike':      atm_strike,
        'ep':          ep,
        'xp':          exit_price,
        'entry_time':  entry_time,
        'exit_time':   exit_time,
        'exit_reason': exit_reason,
        'pnl':         pnl,
    }


# ─────────────────────────────────────────────
# Full backtest
# ─────────────────────────────────────────────
def run_backtest(gap_threshold=0.5):
    dates  = list_trading_dates()
    trades = []
    skip   = 0

    for i, d in enumerate(dates):
        if i == 0:
            continue
        prev = dates[i-1]
        try:
            result = backtest_day(d, prev, gap_threshold)
            if result:
                trades.append(result)
            else:
                skip += 1
        except Exception as e:
            skip += 1

    return pd.DataFrame(trades)


print("Running gap fill backtest (threshold=0.5%)...")
df = run_backtest(gap_threshold=0.5)

if df.empty:
    print("No trades found!")
    sys.exit(1)

# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['year']    = df['date_dt'].dt.year
df = df.sort_values('date_dt').reset_index(drop=True)

total_n  = len(df)
wins     = (df.pnl > 0).sum()
wr       = r2(wins / total_n * 100)
total_pnl= r2(df.pnl.sum())
avg_pnl  = r2(df.pnl.mean())
max_win  = r2(df.pnl.max())
max_loss = r2(df.pnl.min())

equity   = df['pnl'].cumsum()
peak     = equity.cummax()
dd       = equity - peak
max_dd   = r2(dd.min())

print(f"\n{'='*60}")
print(f"  GAP FILL STRATEGY — Full Backtest Results")
print(f"{'='*60}")
print(f"  Trades     : {total_n}")
print(f"  Win Rate   : {wr}%")
print(f"  Total P&L  : Rs {total_pnl:,.0f}")
print(f"  Avg P&L    : Rs {avg_pnl:,.0f}")
print(f"  Best trade : Rs {max_win:,.0f}")
print(f"  Worst trade: Rs {max_loss:,.0f}")
print(f"  Max DD     : Rs {max_dd:,.0f}")

print(f"\n{'='*60}")
print(f"  Exit reason breakdown")
print(f"{'='*60}")
for reason, grp in df.groupby('exit_reason'):
    n = len(grp); w = (grp.pnl>0).sum()
    print(f"  {reason:<12} {n:>4} trades | WR {w/n*100:>5.1f}% | P&L Rs {grp.pnl.sum():>10,.0f}")

print(f"\n{'='*60}")
print(f"  Direction breakdown")
print(f"{'='*60}")
for d, grp in df.groupby('direction'):
    n = len(grp); w = (grp.pnl>0).sum()
    print(f"  {d:<12} {n:>4} trades | WR {w/n*100:>5.1f}% | P&L Rs {grp.pnl.sum():>10,.0f}")

print(f"\n{'='*60}")
print(f"  Year-wise breakdown")
print(f"{'='*60}")
for yr, grp in df.groupby('year'):
    n = len(grp); w = (grp.pnl>0).sum()
    print(f"  {yr}  {n:>4} trades | WR {w/n*100:>5.1f}% | P&L Rs {grp.pnl.sum():>10,.0f}")

print(f"\n{'='*60}")
print(f"  Gap size sensitivity")
print(f"{'='*60}")
for thresh in [0.3, 0.5, 0.75, 1.0, 1.5]:
    sub = df[df['gap_pct'].abs() >= thresh]
    if len(sub) == 0: continue
    print(f"  gap >= {thresh:.1f}%  {len(sub):>4} trades | WR {(sub.pnl>0).mean()*100:.1f}% | P&L Rs {sub.pnl.sum():>10,.0f}")

# ─────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────
out_csv = 'data/20260428/60_gap_fill_trades.csv'
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ─────────────────────────────────────────────
# Chart 1: Equity curve
# ─────────────────────────────────────────────
eq_data = [{'time': int(r.date_dt.timestamp()), 'value': round(eq, 2)}
           for (_, r), eq in zip(df.iterrows(), equity)]
dd_data = [{'time': int(r.date_dt.timestamp()), 'value': round(d, 2)}
           for (_, r), d in zip(df.iterrows(), dd)]

tv_eq = {
    'lines': [
        {'id': 'equity',   'label': 'Equity',   'data': eq_data,
         'seriesType': 'baseline', 'baseValue': 0},
        {'id': 'drawdown', 'label': 'Drawdown', 'data': dd_data,
         'seriesType': 'baseline', 'baseValue': 0, 'isNewPane': True},
    ],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('gap_fill_equity', tv_eq,
                  title=f'Gap Fill Strategy — {total_n} trades | WR {wr}% | P&L Rs {total_pnl:,.0f}')
print("📊 Equity chart sent")

# ─────────────────────────────────────────────
# Chart 2: Year-wise P&L bars
# ─────────────────────────────────────────────
yr_stats = df.groupby('year').agg(pnl=('pnl','sum'), n=('pnl','count'),
                                   wins=('pnl', lambda x: (x>0).sum())).reset_index()
ts_base = int(pd.Timestamp('2021-01-01').timestamp())
bar_yr  = []
for i, row in yr_stats.iterrows():
    bar_yr.append({
        'time':  ts_base + i * 86400 * 365,
        'value': round(row['pnl'], 2),
        'color': '#26a69a' if row['pnl'] >= 0 else '#ef5350',
        'label': str(int(row['year'])),
    })

tv_yr = {'lines': [{'id':'yr_pnl','label':'Year P&L','seriesType':'bar',
                    'data': bar_yr, 'xLabels': [b['label'] for b in bar_yr]}],
         'candlestick':[],'volume':[],'isTvFormat':False}
send_custom_chart('gap_yr_pnl', tv_yr, title='Gap Fill — Year-wise P&L')
print("📊 Year-wise chart sent")

# ─────────────────────────────────────────────
# Chart 3: Compare gap fill vs v17a
# ─────────────────────────────────────────────
v17a_df = pd.read_csv('data/56_combined_trades.csv', parse_dates=['date'])
v17a_df = v17a_df[v17a_df['strategy']=='v17a'].sort_values('date').reset_index(drop=True)
v17a_eq = v17a_df['pnl'].cumsum()

ts_v17a   = [int(pd.Timestamp(r.date).timestamp()) for _, r in v17a_df.iterrows()]
ts_gap    = [int(r.date_dt.timestamp()) for _, r in df.iterrows()]

tv_cmp = {
    'lines': [
        {'id': 'v17a', 'label': 'v17a CPR+EMA', 'color': '#26a69a',
         'data': [{'time': t, 'value': round(e,2)} for t,e in zip(ts_v17a, v17a_eq)]},
        {'id': 'gap',  'label': 'Gap Fill',      'color': '#4BC0C0',
         'data': [{'time': t, 'value': round(e,2)} for t,e in zip(ts_gap, equity)]},
    ],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('gap_vs_v17a', tv_cmp, title='Gap Fill vs v17a CPR — Equity Comparison')
print("📊 Comparison chart sent")

print("\nAll done!")
