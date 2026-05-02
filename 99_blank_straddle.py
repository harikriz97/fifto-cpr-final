"""
99_blank_straddle.py — Short Straddle on Blank Days
=====================================================
Sell ATM CE + ATM PE simultaneously on days where base strategy has no signal.

Simulation logic (tick-by-tick, both legs):
  Entry  : 10:00:02 (market settled, first signal checked)
  Strike : ATM (round spot to nearest 50)
  Target : combined premium decays 30% → exit both legs
  SL     : combined value rises 50% → exit both legs (hard loss)
  Trailing: after 20% profit locked, trail to BE; after 40% → trail to 20%
  EOD    : 15:20 exit regardless

Filters tested in parallel:
  A. All blank days (no filter)
  B. CPR neutral: open inside TC–BC band
  C. Narrow range expected: CPR width < avg (sideways bias)
  D. B + C combined

Lots: 1 CE lot + 1 PE lot (SCALE applies to both)
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

EOD_EXIT   = '15:20:00'
ENTRY_TIME = '10:00:02'
YEARS      = 5
OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50

def r2(v): return round(float(v), 2)

def get_atm(spot):
    return int(round(spot / STRIKE_INT) * STRIKE_INT)

# ── Straddle simulation ───────────────────────────────────────────────────────
def simulate_straddle(date_str, expiry, atm, entry_time,
                      tgt_pct=0.30, sl_pct=0.50):
    """
    Sell 1 CE + 1 PE at same ATM strike.
    Returns (pnl_ce, pnl_pe, total_pnl, exit_reason, ep_ce, ep_pe, xp_ce, xp_pe, exit_time)
    or None if data unavailable.
    """
    ce_instr = f'NIFTY{expiry}{atm}CE'
    pe_instr = f'NIFTY{expiry}{atm}PE'

    ce_raw = load_tick_data(date_str, ce_instr, entry_time)
    pe_raw = load_tick_data(date_str, pe_instr, entry_time)

    # Both legs must have data
    if ce_raw is None or ce_raw.empty: return None
    if pe_raw is None or pe_raw.empty: return None

    ce_raw = ce_raw[ce_raw['time'] >= entry_time].reset_index(drop=True)
    pe_raw = pe_raw[pe_raw['time'] >= entry_time].reset_index(drop=True)

    if ce_raw.empty or pe_raw.empty: return None

    ep_ce = r2(ce_raw.iloc[0]['price'])
    ep_pe = r2(pe_raw.iloc[0]['price'])

    if ep_ce <= 0 or ep_pe <= 0: return None
    combined_ep = ep_ce + ep_pe
    if combined_ep < 20: return None  # too illiquid / near-expiry junk

    # Build time-indexed series, deduplicate then forward-fill gaps
    ce_s = ce_raw.groupby('time')['price'].last()
    pe_s = pe_raw.groupby('time')['price'].last()
    all_times = sorted(set(ce_s.index) | set(pe_s.index))

    ce_ff = ce_s.reindex(all_times).ffill()
    pe_ff = pe_s.reindex(all_times).ffill()

    # Track trailing SL on combined
    max_profit_pct = 0.0
    trail_sl_pct   = None    # None = hard SL only; else = combined must stay above this

    for t in all_times:
        if t > EOD_EXIT:
            break
        if t >= EOD_EXIT:
            ce_p = r2(ce_ff[t]); pe_p = r2(pe_ff[t])
            pnl_ce = r2((ep_ce - ce_p) * LOT_SIZE * SCALE)
            pnl_pe = r2((ep_pe - pe_p) * LOT_SIZE * SCALE)
            return pnl_ce, pnl_pe, r2(pnl_ce + pnl_pe), 'eod', ep_ce, ep_pe, ce_p, pe_p, t

        ce_p = r2(ce_ff[t]); pe_p = r2(pe_ff[t])
        combined_now = ce_p + pe_p

        profit_pct = (combined_ep - combined_now) / combined_ep

        # Update max profit and set trailing SL
        if profit_pct > max_profit_pct:
            max_profit_pct = profit_pct
        if max_profit_pct >= 0.40:
            trail_sl_pct = max(0.20, max_profit_pct - 0.10)   # lock 10% below max
        elif max_profit_pct >= 0.20:
            trail_sl_pct = 0.0                                  # at least break even

        # Target hit
        if profit_pct >= tgt_pct:
            pnl_ce = r2((ep_ce - ce_p) * LOT_SIZE * SCALE)
            pnl_pe = r2((ep_pe - pe_p) * LOT_SIZE * SCALE)
            return pnl_ce, pnl_pe, r2(pnl_ce + pnl_pe), 'target', ep_ce, ep_pe, ce_p, pe_p, t

        # Trail SL hit (locked profit being given back)
        if trail_sl_pct is not None and profit_pct < trail_sl_pct:
            pnl_ce = r2((ep_ce - ce_p) * LOT_SIZE * SCALE)
            pnl_pe = r2((ep_pe - pe_p) * LOT_SIZE * SCALE)
            return pnl_ce, pnl_pe, r2(pnl_ce + pnl_pe), 'trail_sl', ep_ce, ep_pe, ce_p, pe_p, t

        # Hard SL: combined value rises 50% (big directional move)
        if combined_now >= combined_ep * (1 + sl_pct):
            pnl_ce = r2((ep_ce - ce_p) * LOT_SIZE * SCALE)
            pnl_pe = r2((ep_pe - pe_p) * LOT_SIZE * SCALE)
            return pnl_ce, pnl_pe, r2(pnl_ce + pnl_pe), 'hard_sl', ep_ce, ep_pe, ce_p, pe_p, t

    # Ran out of ticks before EOD time
    if len(all_times) == 0: return None
    t    = all_times[-1]
    ce_p = r2(ce_ff.iloc[-1]); pe_p = r2(pe_ff.iloc[-1])
    pnl_ce = r2((ep_ce - ce_p) * LOT_SIZE * SCALE)
    pnl_pe = r2((ep_pe - pe_p) * LOT_SIZE * SCALE)
    return pnl_ce, pnl_pe, r2(pnl_ce + pnl_pe), 'eod', ep_ce, ep_pe, ce_p, pe_p, t

# ── Build daily CPR for filter ────────────────────────────────────────────────
print("Building daily OHLC + CPR...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - 5)
rows  = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({'date': d,
                 'o': day.iloc[0]['price'],
                 'h': day['price'].max(),
                 'l': day['price'].min(),
                 'c': day.iloc[-1]['price']})

df_d = pd.DataFrame(rows)
ph   = df_d['h'].shift(1); pl = df_d['l'].shift(1); pc = df_d['c'].shift(1)
df_d['pvt'] = ((ph + pl + pc) / 3).round(2)
df_d['bc']  = ((ph + pl) / 2).round(2)
df_d['tc']  = (df_d['pvt'] + (df_d['pvt'] - df_d['bc'])).round(2)
df_d['cpr_width'] = (df_d['tc'] - df_d['bc']).round(2)
avg_cpr_width = df_d['cpr_width'].mean()

df_d = df_d.dropna().reset_index(drop=True)
df_5yr = df_d[df_d['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(df_5yr)} days | avg CPR width: {avg_cpr_width:.1f} pts | {time.time()-t0:.0f}s")

# ── Blank day set ─────────────────────────────────────────────────────────────
base_df    = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base_dates = set(base_df['date'].astype(str).str.replace('-', ''))
blank_days = df_5yr[~df_5yr['date'].isin(base_dates)].reset_index(drop=True)
print(f"  Blank days: {len(blank_days)}")

# ── Filter tags ───────────────────────────────────────────────────────────────
blank_days['inside_cpr'] = (blank_days['o'] >= blank_days['bc']) & \
                            (blank_days['o'] <= blank_days['tc'])
blank_days['narrow_cpr'] = blank_days['cpr_width'] < avg_cpr_width * 0.8
blank_days['cpr_bias']   = 'neutral'
blank_days.loc[blank_days['o'] > blank_days['tc'], 'cpr_bias'] = 'bull'
blank_days.loc[blank_days['o'] < blank_days['bc'], 'cpr_bias'] = 'bear'

inside_count = blank_days['inside_cpr'].sum()
narrow_count = blank_days['narrow_cpr'].sum()
print(f"  Inside CPR: {inside_count} | Narrow CPR: {narrow_count}")

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nRunning straddle simulation on all blank days...")
t0      = time.time()
records = []
skipped = 0

for idx, row in blank_days.iterrows():
    dstr = row['date']
    year = dstr[:4]

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None:
        skipped += 1; continue

    # Spot price at 10:00
    spot_10 = spot[spot['time'] >= '10:00:00']
    if spot_10.empty:
        skipped += 1; continue
    spot_ref = spot_10.iloc[0]['price']

    atm      = get_atm(spot_ref)
    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries:
        skipped += 1; continue
    expiry   = expiries[0]

    res = simulate_straddle(dstr, expiry, atm, ENTRY_TIME)
    if res is None:
        skipped += 1; continue

    pnl_ce, pnl_pe, total_pnl, reason, ep_ce, ep_pe, xp_ce, xp_pe, xt = res

    records.append(dict(
        date=dstr, year=year,
        inside_cpr=row['inside_cpr'],
        narrow_cpr=row['narrow_cpr'],
        cpr_bias=row['cpr_bias'],
        atm=atm, expiry=expiry,
        ep_ce=ep_ce, ep_pe=ep_pe,
        xp_ce=xp_ce, xp_pe=xp_pe,
        combined_ep=r2(ep_ce + ep_pe),
        exit_reason=reason, exit_time=xt,
        pnl_ce=pnl_ce, pnl_pe=pnl_pe,
        total_pnl=r2(total_pnl),
        win=total_pnl > 0
    ))

    if idx % 50 == 0:
        print(f"  {idx}/{len(blank_days)} | {len(records)} trades | skip {skipped} | {time.time()-t0:.0f}s")

print(f"  Done | {len(records)} trades | skipped {skipped} | {time.time()-t0:.0f}s")

# ── Results ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

def stats(g, label=''):
    if g.empty:
        print(f"  {label}: 0 trades"); return
    wr   = g['win'].mean() * 100
    pnl  = g['total_pnl'].sum()
    avg  = g['total_pnl'].mean()
    ep   = g['combined_ep'].mean()
    ex   = dict(g['exit_reason'].value_counts())
    print(f"  {label}: {len(g):>4}t | WR {wr:>5.1f}% | Rs.{pnl:>9,.0f} | "
          f"Avg Rs.{avg:>6,.0f} | Avg EP {ep:.0f} | {ex}")

sep = '─' * 72
print(f"\n{'='*72}")
print(f"  SHORT STRADDLE — BLANK DAYS  (entry {ENTRY_TIME}, TGT 30%, SL 50%)")
print(f"{'='*72}")

stats(df,                                                     'A. All blank days  ')
stats(df[df['inside_cpr']],                                   'B. Inside CPR      ')
stats(df[df['narrow_cpr']],                                   'C. Narrow CPR      ')
stats(df[df['inside_cpr'] & df['narrow_cpr']],                'D. Inside+Narrow   ')

print(f"\n{sep}")
print("  BY CPR BIAS")
print(sep)
for bias, g in df.groupby('cpr_bias'):
    stats(g, f"  {bias:<10}")

print(f"\n{sep}")
print("  YEAR-WISE (all blank days)")
print(sep)
for yr, g in df.groupby('year'):
    stats(g, f"  {yr}")

print(f"\n{sep}")
print("  EXIT BREAKDOWN")
print(sep)
for reason, g in df.groupby('exit_reason'):
    pnl = g['total_pnl'].sum()
    avg = g['total_pnl'].mean()
    print(f"  {reason:<12}: {len(g):>4}t | Rs.{pnl:>9,.0f} | Avg Rs.{avg:>6,.0f}")

# DTE breakdown (0 vs 1+ DTE)
def parse_expiry(exp):
    if len(exp) == 6:
        return pd.Timestamp(f"20{exp[:2]}-{exp[2:4]}-{exp[4:6]}")
    return pd.Timestamp(f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}")

df['dte'] = df.apply(lambda r:
    (parse_expiry(str(r['expiry'])) -
     pd.Timestamp(r['date'][:4]+'-'+r['date'][4:6]+'-'+r['date'][6:])).days, axis=1)
print(f"\n{sep}")
print("  DTE BREAKDOWN")
print(sep)
for dte, g in df.groupby('dte'):
    stats(g, f"  DTE={dte}")

# ── Compare with CRT ──────────────────────────────────────────────────────────
crt_blank = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = crt_blank[crt_blank['is_blank'] == True]

print(f"\n{'='*72}")
print("  STRADDLE vs CRT Approach D — BLANK DAYS")
print(f"{'='*72}")
print(f"  {'Strategy':<28} | Trades | WR     | P&L         | Avg")
print(f"  {'-'*68}")

for label, g, col in [
    ("CRT Approach D",         crt_blank,  'pnl_65'),
    ("Straddle (all blank)",   df,         'total_pnl'),
    ("Straddle (inside CPR)",  df[df['inside_cpr']], 'total_pnl'),
    ("Straddle (narrow CPR)",  df[df['narrow_cpr']], 'total_pnl'),
]:
    if g.empty: print(f"  {label:<28} | no trades"); continue
    print(f"  {label:<28} | {len(g):>6} | {g['win'].mean()*100:>5.1f}% | "
          f"Rs.{g[col].sum():>10,.0f} | Rs.{g[col].mean():>6,.0f}")

df.to_csv(f'{OUT_DIR}/99_blank_straddle.csv', index=False)
print(f"\n  Saved → {OUT_DIR}/99_blank_straddle.csv")
print("\nDone.")
