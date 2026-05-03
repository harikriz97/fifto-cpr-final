"""
85_crt_scalping.py — CRT Full-Day 15-Min Scalping
==================================================
Instead of only morning PDH/PDL sweep, scan ALL 15-min candles:
  Reference: PRIOR 15-min candle high/low (not PDH/PDL)
  Signal:    Current candle sweeps prior candle H or L AND closes back inside
  Entry:     Next 15-min candle open + 1 min (wait for full close)
  Max:       Up to 4 trades/day, sequential (exit first before new entry)

Option type:
  Bearish CRT (sweep high, close back) → Sell CE OTM1 OR Buy PE ATM
  Bullish CRT (sweep low, close back)  → Sell PE OTM1 OR Buy CE ATM

Tested configs:
  A: Sell  | TGT 20%  | SL 100% | 3-tier trailing
  B: Sell  | TGT 15%  | SL 50%  | hard SL (scalp)
  C: Buy   | TGT 50%  | SL 25%  | hard SL (scalp)
  D: Buy   | TGT 30%  | SL 20%  | hard SL (quick scalp)

Scan window: 09:15 – 13:00 (avoid afternoon chop)
EOD: 13:30 for buying, 15:20 for selling
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
CANDLE_MIN = 15
SCAN_END   = '13:00:00'   # stop looking for new signals after this
MAX_TRADES = 4            # max per day
YEARS      = 5
OUT_DIR    = 'data/20260430'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def get_otm1(s, opt):
    atm = get_atm(s)
    return atm + STRIKE_INT if opt == 'CE' else atm - STRIKE_INT

def build_15min_full(tks, start='09:15:00', end='13:15:00'):
    df = tks[(tks['time'] >= start) & (tks['time'] <= end)].copy()
    if df.empty: return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.set_index('ts').sort_index()
    ohlc = df['price'].resample('15min').ohlc().dropna()
    ohlc.columns = ['o', 'h', 'l', 'c']
    ohlc['time'] = ohlc.index.strftime('%H:%M:%S')
    return ohlc.reset_index(drop=True)

def simulate_sell(date_str, expiry, strike, opt, entry_time,
                  tgt_pct=0.20, sl_pct=1.00, eod='15:20:00', trail=True):
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + sl_pct)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= eod: return r2((ep-p)*LOT_SIZE), 'eod', r2(ep), r2(p), t
        d = (ep-p)/ep
        if d > md: md = d
        if trail:
            if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
            elif md >= 0.40: sl = min(sl, r2(ep*0.80))
            elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE), 'lockin_sl' if (trail and sl<hsl) else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

def simulate_buy(date_str, expiry, strike, opt, entry_time,
                 tgt_pct=0.50, sl_pct=0.25, eod='13:30:00'):
    instr = f'NIFTY{expiry}{strike}{opt}'
    tks = load_tick_data(date_str, instr, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 + tgt_pct))
    sl  = r2(ep * (1 - sl_pct))
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= eod: return r2((p-ep)*LOT_SIZE), 'eod', r2(ep), r2(p), t
        if p >= tgt: return r2((p-ep)*LOT_SIZE), 'target', r2(ep), r2(p), t
        if p <= sl:  return r2((p-ep)*LOT_SIZE), 'hard_sl', r2(ep), r2(p), t
    return r2((ps[-1]-ep)*LOT_SIZE), 'eod', r2(ep), r2(ps[-1]), ts[-1]

# ── Build daily OHLC ──────────────────────────────────────────────────────────
print("Building daily OHLC...")
t0 = time.time()
all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra      = max(0, all_dates.index(dates_5yr[0]) - 60)

rows = []
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')]
    if len(day)<2: continue
    rows.append({'date':d,'o':day.iloc[0]['price'],
                 'h':day['price'].max(),'l':day['price'].min(),'c':day.iloc[-1]['price']})

ohlc = pd.DataFrame(rows)
ohlc = ohlc.dropna().reset_index(drop=True)
ohlc_5yr = ohlc[ohlc['date'].isin(dates_5yr)].reset_index(drop=True)
print(f"  {len(ohlc_5yr)} days | {time.time()-t0:.0f}s")

# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS = {
    'sell_20_trail': dict(side='sell', tgt=0.20, sl=1.00, eod='15:20:00', trail=True),
    'sell_15_hard':  dict(side='sell', tgt=0.15, sl=0.50, eod='15:20:00', trail=False),
    'sell_10_hard':  dict(side='sell', tgt=0.10, sl=0.30, eod='15:20:00', trail=False),
    'buy_50_25':     dict(side='buy',  tgt=0.50, sl=0.25, eod='13:30:00'),
    'buy_30_20':     dict(side='buy',  tgt=0.30, sl=0.20, eod='13:30:00'),
}

# ── Main scan ─────────────────────────────────────────────────────────────────
print("\nScanning CRT signals full day (15-min candles)...")
t0 = time.time()

all_signals = []   # all signal metadata (no option sim)
results = {cfg: [] for cfg in CONFIGS}

for idx, row in ohlc_5yr.iterrows():
    dstr = row['date']
    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: continue
    candles = build_15min_full(spot)
    if len(candles) < 2: continue

    expiries = list_expiry_dates(dstr)
    if not expiries: continue
    expiry = expiries[0]

    day_signals = []
    for ci in range(1, len(candles)):
        prev_c = candles.iloc[ci-1]
        curr_c = candles.iloc[ci]
        ct     = curr_c['time']
        if ct > SCAN_END: break

        prev_h = prev_c['h']; prev_l = prev_c['l']
        ch = curr_c['h']; cl = curr_c['l']; cc = curr_c['c']

        # Entry: 1 min after current candle closes
        h_str = ct[:2]; m_str = ct[3:5]
        entry_min = int(h_str)*60 + int(m_str) + CANDLE_MIN + 1
        entry_t   = f"{entry_min//60:02d}:{entry_min%60:02d}:02"
        if entry_t >= '15:15:00': break

        crt_dir = None
        # Bearish: swept above prior high, closed back below
        if ch > prev_h and cc < prev_h:
            crt_dir = 'bearish'
        # Bullish: swept below prior low, closed back above
        elif cl < prev_l and cc > prev_l:
            crt_dir = 'bullish'

        if crt_dir:
            day_signals.append({
                'date': dstr, 'candle_time': ct, 'entry_time': entry_t,
                'crt_dir': crt_dir, 'price_at_entry': cc,
                'sweep_h': r2(ch) if crt_dir=='bearish' else None,
                'sweep_l': r2(cl) if crt_dir=='bullish' else None,
                'year': dstr[:4]
            })

    all_signals.extend(day_signals)

    # Now simulate: sequential, max MAX_TRADES per day
    if not day_signals:
        if idx % 100 == 0: print(f"  {idx}/{len(ohlc_5yr)} | signals:{len(all_signals)} | {time.time()-t0:.0f}s")
        continue

    for cfg_name, cfg in CONFIGS.items():
        occupied_until = '00:00:00'
        day_count      = 0
        for sig in day_signals:
            if day_count >= MAX_TRADES: break
            if sig['entry_time'] <= occupied_until: continue  # previous trade still open

            crt_dir  = sig['crt_dir']
            entry_t  = sig['entry_time']
            ep_approx = sig['price_at_entry']

            if crt_dir == 'bearish':
                opt = 'CE' if cfg['side']=='sell' else 'PE'
                strike = get_otm1(ep_approx, 'CE') if cfg['side']=='sell' else get_atm(ep_approx)
            else:
                opt = 'PE' if cfg['side']=='sell' else 'CE'
                strike = get_otm1(ep_approx, 'PE') if cfg['side']=='sell' else get_atm(ep_approx)

            if cfg['side'] == 'sell':
                res = simulate_sell(dstr, expiry, strike, opt, entry_t,
                                    tgt_pct=cfg['tgt'], sl_pct=cfg['sl'],
                                    eod=cfg['eod'], trail=cfg.get('trail', False))
            else:
                res = simulate_buy(dstr, expiry, strike, opt, entry_t,
                                   tgt_pct=cfg['tgt'], sl_pct=cfg['sl'],
                                   eod=cfg['eod'])

            if res:
                pnl75, reason, ep, xp, xt = res
                pnl65 = r2(pnl75 * SCALE)
                results[cfg_name].append({
                    'date': dstr, 'candle_time': sig['candle_time'],
                    'entry_time': entry_t, 'crt': crt_dir,
                    'opt': opt, 'strike': strike, 'ep': ep, 'xp': xp,
                    'exit_time': xt, 'exit_reason': reason,
                    'pnl_65': pnl65, 'win': pnl65 > 0, 'year': dstr[:4]
                })
                occupied_until = xt
                day_count += 1

    if idx % 100 == 0:
        print(f"  {idx}/{len(ohlc_5yr)} | signals:{len(all_signals)} | {time.time()-t0:.0f}s")

print(f"Done | {time.time()-t0:.0f}s")

# ── Signal frequency analysis ─────────────────────────────────────────────────
df_sig = pd.DataFrame(all_signals)
print(f"\n{'='*62}")
print("  CRT SIGNAL FREQUENCY (15-min, all day, prior candle sweep)")
print(f"{'='*62}")
print(f"  Trading days:     {len(ohlc_5yr)}")
print(f"  Total signals:    {len(df_sig)}")
print(f"  Avg per day:      {len(df_sig)/len(ohlc_5yr):.1f}")
print(f"  Days with 0 sig:  {len(ohlc_5yr) - df_sig['date'].nunique()}")
print(f"  Days with 1 sig:  {(df_sig.groupby('date').size()==1).sum()}")
print(f"  Days with 2 sig:  {(df_sig.groupby('date').size()==2).sum()}")
print(f"  Days with 3+ sig: {(df_sig.groupby('date').size()>=3).sum()}")
print(f"\n  By candle time:")
print(df_sig['candle_time'].value_counts().sort_index().to_string())
print(f"\n  Bull vs Bear signals:")
print(df_sig['crt_dir'].value_counts().to_string())

# ── Results per config ────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  BACKTEST RESULTS — Full Day CRT Scalping")
print(f"{'='*62}")

for cfg_name, recs in results.items():
    if not recs:
        print(f"\n{cfg_name}: no trades"); continue
    df_r = pd.DataFrame(recs)
    wr   = df_r['win'].mean()*100
    pnl  = df_r['pnl_65'].sum()
    avg  = df_r['pnl_65'].mean()
    cfg  = CONFIGS[cfg_name]
    print(f"\n  [{cfg_name}] TGT={int(cfg['tgt']*100)}% SL={int(cfg['sl']*100)}% side={cfg['side']}")
    print(f"  Trades:    {len(df_r)} ({len(df_r)/len(ohlc_5yr):.1f}/day avg)")
    print(f"  Win Rate:  {wr:.1f}%")
    print(f"  Total P&L: Rs.{pnl:,.0f}")
    print(f"  Avg/trade: Rs.{avg:,.0f}")
    print(f"  Best:      Rs.{df_r['pnl_65'].max():,.0f}")
    print(f"  Worst:     Rs.{df_r['pnl_65'].min():,.0f}")
    print(f"  Exits: {dict(df_r['exit_reason'].value_counts())}")
    for yr in sorted(df_r['year'].unique()):
        g = df_r[df_r['year']==yr]
        print(f"    {yr}: {len(g):>4}t | WR {g['win'].mean()*100:.0f}% | Rs.{g['pnl_65'].sum():,.0f}")

    if not df_r.empty:
        df_r.to_csv(f'{OUT_DIR}/85_crt_{cfg_name}.csv', index=False)

print("\nDone.")
