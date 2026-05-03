"""
119_final_combined.py — Final combined system + comprehensive Excel
====================================================================
Combines every bias-free validated strategy into one system:

  A: Base trades (from 75_live_simulation.csv)
  B: Blank day S3 TrendIB (corrected — IB expansion pre-entry only)
  C: Blank day S1 OHL — only on days where B didn't fire
       C_both: CE + PE signals
       C_pe:   PE signals only (CE direction consistently loses)
  D: S4 2nd trade — after base TARGET exit (pullback re-entry)
  E: S4-style 2nd trade after BLANK day target exit (B or C)

Excel sheets: summary | all_trades | by_year | by_month | by_weekday |
              by_daytype | by_open_pos | by_ib | by_cpr_width |
              exit_analysis | win_loss | drawdown | equity_daily |
              blank_deep | ohl_sweep | vwap_sweep
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
S3_ENTRY   = '09:46:02'
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
OHL_TOL    = 0.0015     # 0.15% — best from sweep
PULLBACK_LO= 0.60; PULLBACK_HI = 0.75
REENTRY_CUT= '13:30:00'

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)
def t2m(t):
    h,m,s = map(int,t.split(':'))
    return h*60+m+s/60


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
            return r2((ep-p)*LOT_SIZE*SCALE), 'eod', r2(ep), r2(p), ts[i-1] if i>0 else t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE*SCALE), 'target', r2(ep), r2(p), t
        if p >= sl:  return r2((ep-p)*LOT_SIZE*SCALE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(ep), r2(p), t
    return r2((ep-ps[-1])*LOT_SIZE*SCALE), 'eod', r2(ep), r2(ps[-1]), ts[-1]


def try_2nd_trade(date_str, opt, ep1, exit_time, expiries, spot_day, opt_tks):
    """S4/E style: pullback to 60-75% of ep1 after first trade exit."""
    if exit_time is None or t2m(exit_time) > t2m(REENTRY_CUT):
        return None
    pb_lo = ep1 * PULLBACK_LO; pb_hi = ep1 * PULLBACK_HI
    scan = opt_tks[(opt_tks['time'] >= exit_time) & (opt_tks['time'] <= REENTRY_CUT)]
    reentry_t = None
    for _, row in scan.iterrows():
        if pb_lo <= row['price'] <= pb_hi:
            h,m,s = map(int, row['time'].split(':'))
            reentry_t = f'{h:02d}:{m:02d}:{min(s+2,59):02d}'; break
    if reentry_t is None: return None
    spot_re = spot_day[spot_day['time'] >= reentry_t]
    if spot_re.empty: return None
    atm2 = get_atm(spot_re.iloc[0]['price'])
    instr2 = f'NIFTY{expiries[0]}{atm2}{opt}'
    res = simulate_sell(date_str, instr2, reentry_t)
    if res is None: return None
    pnl, reason, ep2, xp2, xt2 = res
    return {'pnl': pnl, 'win': pnl > 0, 'ep': ep2, 'xp': xp2,
            'exit_reason': reason, 'entry_time': reentry_t, 'strike': atm2}


def _worker_blank(args):
    date_str, meta, blank_set = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    if date_str not in blank_set: return []
    tc = meta.get('tc'); bc = meta.get('bc')

    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return []
        day = tks[(tks['time']>='09:15:00')&(tks['time']<='15:30:00')].copy()
        if len(day) < 30: return []

        ib = day[day['time'] <= IB_END]
        if len(ib) < 5: return []
        ib_h = ib['price'].max(); ib_l = ib['price'].min()
        spot_open = ib.iloc[0]['price']
        ib_range_pts = r2(ib_h - ib_l)
        ib_range_pct = r2((ib_h - ib_l) / spot_open * 100)
        if ib_h <= ib_l: return []

        # Pre-entry IB expansion (bias-free for S3)
        pre = day[(day['time'] > IB_END) & (day['time'] < S3_ENTRY)]
        ib_exp_up_pre   = (not pre.empty) and (pre['price'].max() > ib_h)
        ib_exp_down_pre = (not pre.empty) and (pre['price'].min() < ib_l)

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []

        results = []
        traded_signal = None  # track which strategy fired to avoid double trade

        # ── S3: TrendIB (corrected) ─────────────────────────────────────────
        if tc is not None and bc is not None:
            bullish_open = spot_open > tc
            bearish_open = spot_open < bc
            s3_sig = None
            if bullish_open and ib_exp_up_pre and not ib_exp_down_pre:
                s3_sig = 'PE'
            elif bearish_open and ib_exp_down_pre and not ib_exp_up_pre:
                s3_sig = 'CE'

            if s3_sig is not None:
                spot_at = day[day['time'] >= S3_ENTRY]
                if not spot_at.empty:
                    atm = get_atm(spot_at.iloc[0]['price'])
                    instr = f'NIFTY{expiries[0]}{atm}{s3_sig}'
                    res = simulate_sell(date_str, instr, S3_ENTRY)
                    if res is not None:
                        pnl, reason, ep, xp, xt = res
                        results.append({
                            'strategy': 'B_S3', 'date': date_str, 'signal': s3_sig,
                            'entry_time': S3_ENTRY, 'strike': atm, 'ep': ep, 'xp': xp,
                            'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
                            'ib_range_pts': ib_range_pts, 'ib_range_pct': ib_range_pct,
                        })
                        traded_signal = s3_sig

                        # Strategy E: 2nd trade after S3 target
                        if reason == 'target':
                            opt_tks_s3 = load_tick_data(date_str, instr, S3_ENTRY)
                            if opt_tks_s3 is not None and not opt_tks_s3.empty:
                                spot_day = day
                                r2t = try_2nd_trade(date_str, s3_sig, ep, xt,
                                                    expiries, spot_day, opt_tks_s3)
                                if r2t is not None:
                                    r2t.update({
                                        'strategy': 'E_blank2nd', 'date': date_str,
                                        'signal': s3_sig,
                                        'ib_range_pts': ib_range_pts, 'ib_range_pct': ib_range_pct,
                                    })
                                    results.append(r2t)

        # ── S1: OHL — only if S3 didn't fire ────────────────────────────────
        if traded_signal is None:
            is_oh = abs(spot_open - ib_h) / spot_open <= OHL_TOL
            is_ol = abs(spot_open - ib_l) / spot_open <= OHL_TOL

            for label, condition, sig in [
                ('C_both', is_oh or is_ol, 'CE' if is_oh else ('PE' if is_ol else None)),
                ('C_pe',   is_ol,         'PE'),
            ]:
                if not condition or sig is None: continue
                spot_at = day[day['time'] >= S3_ENTRY]
                if spot_at.empty: continue
                atm = get_atm(spot_at.iloc[0]['price'])
                instr = f'NIFTY{expiries[0]}{atm}{sig}'
                res = simulate_sell(date_str, instr, S3_ENTRY)
                if res is None: continue
                pnl, reason, ep, xp, xt = res
                results.append({
                    'strategy': label, 'date': date_str, 'signal': sig,
                    'entry_time': S3_ENTRY, 'strike': atm, 'ep': ep, 'xp': xp,
                    'pnl': pnl, 'win': pnl > 0, 'exit_reason': reason,
                    'ib_range_pts': ib_range_pts, 'ib_range_pct': ib_range_pct,
                })

                # Strategy E: 2nd trade after S1 target
                if reason == 'target' and label == 'C_both':
                    opt_tks_s1 = load_tick_data(date_str, instr, S3_ENTRY)
                    if opt_tks_s1 is not None and not opt_tks_s1.empty:
                        r2t = try_2nd_trade(date_str, sig, ep, xt,
                                            expiries, day, opt_tks_s1)
                        if r2t is not None:
                            r2t.update({
                                'strategy': 'E_blank2nd', 'date': date_str,
                                'signal': sig,
                                'ib_range_pts': ib_range_pts, 'ib_range_pct': ib_range_pct,
                            })
                            results.append(r2t)

        return results
    except Exception:
        return []


def _worker_s4(trade_row):
    """D: 2nd trade after base TARGET exit."""
    date_str = trade_row['date_str']
    opt = trade_row['opt']
    ep1 = trade_row.get('entry_price', trade_row.get('ep', None))
    if ep1 is None: return []
    ep1 = float(ep1)
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return []
        spot_all = load_spot_data(date_str, 'NIFTY')
        if spot_all is None: return []
        spot_day = spot_all[(spot_all['time']>='09:15:00')&(spot_all['time']<='15:30:00')]
        spot_at  = spot_day[spot_day['time']>=trade_row['entry_time']]
        if spot_at.empty: return []
        atm1  = get_atm(spot_at.iloc[0]['price'])
        instr = f'NIFTY{expiries[0]}{atm1}{opt}'
        opt_tks = load_tick_data(date_str, instr, trade_row['entry_time'])
        if opt_tks is None or opt_tks.empty: return []
        os_ = opt_tks[opt_tks['time']>=trade_row['entry_time']].reset_index(drop=True)
        if os_.empty: return []
        ps = os_['price'].values; ts = os_['time'].values

        tgt = ep1*(1-TGT_PCT); hsl = ep1*2.0; sl = hsl; md = 0.0
        exit_t = None
        for t_, p in zip(ts, ps):
            if t_ >= EOD_EXIT: exit_t = t_; break
            d = (ep1-p)/ep1
            if d > md: md = d
            if   md >= 0.60: sl = min(sl, ep1*(1-md*0.95))
            elif md >= 0.40: sl = min(sl, ep1*0.80)
            elif md >= 0.25: sl = min(sl, ep1)
            if p <= tgt: exit_t = t_; break
            if p >= sl:  exit_t = t_; break

        r2t = try_2nd_trade(date_str, opt, ep1, exit_t, expiries, spot_day, opt_tks)
        if r2t is None: return []
        r2t.update({'strategy': 'D_base2nd', 'date': date_str, 'signal': opt,
                    'ib_range_pts': None, 'ib_range_pct': None})
        return [r2t]
    except Exception:
        return []


def enrich(df, beh_map):
    """Add day behavior metadata to trade dataframe."""
    if df.empty: return df
    df = df.copy()
    df['date_str'] = df['date'].astype(str)
    for col in ['day_type','open_pos','ib_expand','cpr_class','tc','bc','pvt',
                'gap_pct','ib_class','ichi_sig']:
        df[col] = df['date_str'].map(lambda d: beh_map.get(d, {}).get(col))
    dt = pd.to_datetime(df['date_str'], format='mixed')
    df['year']    = dt.dt.year.astype(str)
    df['month']   = dt.dt.strftime('%b')
    df['month_n'] = dt.dt.month
    df['weekday'] = dt.dt.strftime('%a')
    df['weekday_n'] = dt.dt.dayofweek
    return df


def pivot_summary(df, rows, label=''):
    if df.empty: return pd.DataFrame()
    return df.groupby(rows).apply(lambda g: pd.Series({
        'trades': len(g),
        'win_rate_%': round(g['win'].mean()*100, 1),
        'total_pnl': round(g['pnl'].sum(), 0),
        'avg_pnl': round(g['pnl'].mean(), 0),
        'avg_win': round(g[g['win']]['pnl'].mean(), 0) if g['win'].any() else 0,
        'avg_loss': round(g[~g['win']]['pnl'].mean(), 0) if (~g['win']).any() else 0,
        'max_win': round(g['pnl'].max(), 0),
        'max_loss': round(g['pnl'].min(), 0),
    })).reset_index()


def daily_equity(df):
    if df.empty: return pd.Series(dtype=float)
    d = df.copy()
    d['dt'] = pd.to_datetime(d['date'].astype(str), format='%Y%m%d')
    return d.groupby('dt')['pnl'].sum()


def eq_pts(s):
    return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
            for d, v in s.items() if pd.notna(v)]


def main():
    t_start = datetime.now()

    # ── Load base trades ───────────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str),format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv':'pnl'})
    base['strategy'] = 'A_base'
    base['signal']   = base['opt']
    base['entry_time'] = base['entry_time'] if 'entry_time' in base.columns else '09:16:02'
    if 'entry_price' in base.columns:
        base = base.rename(columns={'entry_price':'ep'})
    if 'exit_price' in base.columns:
        base = base.rename(columns={'exit_price':'xp'})
    if 'strike' not in base.columns:
        base['strike'] = None
    base['ib_range_pts'] = None; base['ib_range_pct'] = None

    # ── Load day behavior metadata ─────────────────────────────────────────────
    beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                        sheet_name='all_days_observations', dtype={'date':str})
    beh['date'] = beh['date'].astype(str)
    beh_map = {}
    for _, row in beh.iterrows():
        beh_map[row['date']] = {k: (None if pd.isna(v) else v) for k, v in row.items()}

    meta_map = {d: {k: v for k, v in beh_map[d].items()
                    if k in ('tc','bc','pvt','r1','s1','ichi_sig','cpr_class','ib_class')}
                for d in beh_map}

    # ── Date range + blank set ─────────────────────────────────────────────────
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]
    crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
    blank_set = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str)) | \
                set(pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')['date'].astype(str))

    # ── B/C/E: Blank day strategies ────────────────────────────────────────────
    print(f"B/C/E: Blank day strategies ({len([d for d in dates_5yr if d in blank_set])} blank days)...")
    args_blank = [(d, meta_map.get(d,{}), blank_set) for d in dates_5yr]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_blank = pool.map(_worker_blank, args_blank)
    el = (datetime.now()-t0).total_seconds()
    df_blank = pd.DataFrame([t for day in raw_blank for t in day])
    print(f"  {len(df_blank)} trades in {el:.1f}s")

    if not df_blank.empty:
        for strat, g in df_blank.groupby('strategy'):
            print(f"  {strat:<15}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── D: S4 2nd trade (base) ─────────────────────────────────────────────────
    print(f"\nD: 2nd trade after base target...")
    base_target = base[base['exit_reason']=='target']
    args_s4 = [row for _, row in base_target.iterrows()]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_s4 = pool.map(_worker_s4, args_s4)
    el = (datetime.now()-t0).total_seconds()
    df_s4 = pd.DataFrame([t for day in raw_s4 for t in day])
    print(f"  {len(df_s4)} trades in {el:.1f}s")
    if not df_s4.empty:
        print(f"  D_base2nd  : {len(df_s4)}t | WR {df_s4['win'].mean()*100:.1f}% | "
              f"Rs.{df_s4['pnl'].sum():,.0f} | Avg Rs.{df_s4['pnl'].mean():.0f}")

    # ── Assemble all trades ────────────────────────────────────────────────────
    base_cols = ['strategy','date','date_str','signal','entry_time','ep','xp',
                 'pnl','win','exit_reason','strike','ib_range_pts','ib_range_pct']
    # Standardise base
    b_std = base[['strategy','date','date_str','signal','ep','xp','pnl','win',
                  'exit_reason','strike','ib_range_pts','ib_range_pct']].copy()
    if 'entry_time' not in base.columns:
        b_std['entry_time'] = '09:16:02'
    else:
        b_std['entry_time'] = base['entry_time']

    def std(df_):
        for c in ['strategy','date','signal','entry_time','ep','xp','pnl','win',
                  'exit_reason','strike','ib_range_pts','ib_range_pct']:
            if c not in df_.columns: df_[c] = None
        df_['date'] = df_['date'].astype(str)
        if 'date_str' not in df_.columns:
            df_['date_str'] = df_['date']
        return df_

    frames = [std(b_std)]
    if not df_blank.empty: frames.append(std(df_blank.copy()))
    if not df_s4.empty:    frames.append(std(df_s4.copy()))
    all_trades = pd.concat(frames, ignore_index=True)
    all_trades = enrich(all_trades, beh_map)

    # ── Scenario comparison ────────────────────────────────────────────────────
    # Scenario 1: Base only
    # Scenario 2: Base + S3 + D_s4
    # Scenario 3: Base + S3 + C_both + D_s4 + E_blank2nd
    # Scenario 4: Base + S3 + C_pe   + D_s4 + E_blank2nd

    def scenario_pnl(strats):
        sub = all_trades[all_trades['strategy'].isin(strats)]
        return sub['pnl'].sum(), len(sub), sub['win'].mean()*100 if len(sub) else 0

    scenarios = {
        'S1: Base only':          ['A_base'],
        'S2: Base + D(s4)':       ['A_base','D_base2nd'],
        'S3: Base + B(s3) + D':   ['A_base','B_S3','D_base2nd'],
        'S4: +C_both + E':        ['A_base','B_S3','C_both','D_base2nd','E_blank2nd'],
        'S5: +C_pe + E':          ['A_base','B_S3','C_pe','D_base2nd','E_blank2nd'],
    }

    print(f"\n{'='*72}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'='*72}")
    for name, strats in scenarios.items():
        pnl, n, wr = scenario_pnl(strats)
        print(f"  {name:<35} | {n:>4}t | WR {wr:>5.1f}% | Rs.{pnl:>12,.0f}")

    # Pick best scenario for final charts
    best_strats = ['A_base','B_S3','C_pe','D_base2nd','E_blank2nd']
    df_final = all_trades[all_trades['strategy'].isin(best_strats)]
    df_base_only = all_trades[all_trades['strategy']=='A_base']

    print(f"\n  FINAL SYSTEM (S5): {len(df_final)}t | WR {df_final['win'].mean()*100:.1f}% | "
          f"Rs.{df_final['pnl'].sum():,.0f}")

    # ── COMPREHENSIVE EXCEL ────────────────────────────────────────────────────
    print(f"\nBuilding comprehensive Excel...")
    xls_path = f'{OUT_DIR}/119_final_analysis.xlsx'
    with pd.ExcelWriter(xls_path, engine='openpyxl') as writer:

        # Sheet 1: summary — all strategies
        print("  Sheet: summary")
        s_rows = []
        for strat, g in all_trades.groupby('strategy'):
            wins = g[g['win']]; losses = g[~g['win']]
            s_rows.append({
                'strategy': strat,
                'trades': len(g),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate_%': round(g['win'].mean()*100, 1),
                'total_pnl': round(g['pnl'].sum(), 0),
                'avg_pnl': round(g['pnl'].mean(), 0),
                'avg_win': round(wins['pnl'].mean(), 0) if len(wins) else 0,
                'avg_loss': round(losses['pnl'].mean(), 0) if len(losses) else 0,
                'max_win': round(g['pnl'].max(), 0),
                'max_loss': round(g['pnl'].min(), 0),
                'profit_factor': round(wins['pnl'].sum() / abs(losses['pnl'].sum()), 2)
                                 if len(losses) and losses['pnl'].sum() != 0 else 0,
                'expectancy': round(g['win'].mean()*wins['pnl'].mean() +
                                    (1-g['win'].mean())*losses['pnl'].mean(), 0)
                              if len(wins) and len(losses) else round(g['pnl'].mean(), 0),
            })
        # Add scenario totals
        for name, strats in scenarios.items():
            sub = all_trades[all_trades['strategy'].isin(strats)]
            wins = sub[sub['win']]; losses = sub[~sub['win']]
            s_rows.append({
                'strategy': f'SCENARIO: {name}',
                'trades': len(sub), 'wins': len(wins), 'losses': len(losses),
                'win_rate_%': round(sub['win'].mean()*100, 1),
                'total_pnl': round(sub['pnl'].sum(), 0),
                'avg_pnl': round(sub['pnl'].mean(), 0),
                'avg_win': round(wins['pnl'].mean(), 0) if len(wins) else 0,
                'avg_loss': round(losses['pnl'].mean(), 0) if len(losses) else 0,
                'max_win': round(sub['pnl'].max(), 0),
                'max_loss': round(sub['pnl'].min(), 0),
                'profit_factor': round(wins['pnl'].sum() / abs(losses['pnl'].sum()), 2)
                                 if len(losses) and losses['pnl'].sum() != 0 else 0,
                'expectancy': 0,
            })
        pd.DataFrame(s_rows).to_excel(writer, sheet_name='summary', index=False)

        # Sheet 2: all_trades — every single trade with full metadata
        print("  Sheet: all_trades")
        cols = ['strategy','date','year','month','weekday','signal','entry_time',
                'ep','xp','pnl','win','exit_reason','strike',
                'day_type','open_pos','ib_expand','cpr_class','ib_range_pts','ib_range_pct',
                'tc','bc','pvt','gap_pct','ichi_sig']
        out_cols = [c for c in cols if c in all_trades.columns]
        all_trades[out_cols].sort_values(['date','strategy']).to_excel(
            writer, sheet_name='all_trades', index=False)

        # Sheet 3: by_strategy_year
        print("  Sheet: by_strategy_year")
        sy = pivot_summary(all_trades, ['strategy','year'])
        sy.to_excel(writer, sheet_name='by_strategy_year', index=False)

        # Sheet 4: by_year (final system)
        print("  Sheet: by_year")
        by_yr = pivot_summary(df_final, ['year'])
        by_yr_base = pivot_summary(df_base_only, ['year']).rename(
            columns={'total_pnl':'base_pnl','win_rate_%':'base_wr','trades':'base_trades'})
        by_yr_m = by_yr.merge(by_yr_base[['year','base_pnl','base_wr','base_trades']],
                               on='year', how='left')
        by_yr_m['incremental_pnl'] = by_yr_m['total_pnl'] - by_yr_m['base_pnl']
        by_yr_m.to_excel(writer, sheet_name='by_year', index=False)

        # Sheet 5: by_month (final system)
        print("  Sheet: by_month")
        bm = df_final.sort_values('month_n')
        pivot_summary(bm, ['month_n','month']).to_excel(
            writer, sheet_name='by_month', index=False)

        # Sheet 6: by_weekday (final system)
        print("  Sheet: by_weekday")
        bwd = df_final.sort_values('weekday_n')
        pivot_summary(bwd, ['weekday_n','weekday']).to_excel(
            writer, sheet_name='by_weekday', index=False)

        # Sheet 7: by_daytype
        print("  Sheet: by_daytype")
        pivot_summary(all_trades.dropna(subset=['day_type']), ['strategy','day_type']).to_excel(
            writer, sheet_name='by_daytype', index=False)

        # Sheet 8: by_open_pos
        print("  Sheet: by_open_pos")
        pivot_summary(all_trades.dropna(subset=['open_pos']), ['strategy','open_pos','signal']).to_excel(
            writer, sheet_name='by_open_pos', index=False)

        # Sheet 9: by_ib_expand
        print("  Sheet: by_ib_expand")
        pivot_summary(all_trades.dropna(subset=['ib_expand']), ['strategy','ib_expand']).to_excel(
            writer, sheet_name='by_ib_expand', index=False)

        # Sheet 10: by_cpr_class
        print("  Sheet: by_cpr_class")
        pivot_summary(all_trades.dropna(subset=['cpr_class']), ['strategy','cpr_class']).to_excel(
            writer, sheet_name='by_cpr_class', index=False)

        # Sheet 11: exit_analysis
        print("  Sheet: exit_analysis")
        pivot_summary(all_trades, ['strategy','exit_reason']).to_excel(
            writer, sheet_name='exit_analysis', index=False)

        # Sheet 12: win_loss
        print("  Sheet: win_loss")
        wl_rows = []
        for strat, g in all_trades.groupby('strategy'):
            wins = g[g['win']]; losses = g[~g['win']]
            if len(wins) == 0 or len(losses) == 0: continue
            wl_rows.append({
                'strategy': strat,
                'avg_win': round(wins['pnl'].mean(), 0),
                'avg_loss': round(losses['pnl'].mean(), 0),
                'wl_ratio': round(abs(wins['pnl'].mean() / losses['pnl'].mean()), 2),
                'profit_factor': round(wins['pnl'].sum() / abs(losses['pnl'].sum()), 2),
                'win_rate_%': round(g['win'].mean()*100, 1),
                'expectancy': round(g['win'].mean()*wins['pnl'].mean() +
                                    (1-g['win'].mean())*losses['pnl'].mean(), 0),
                'max_consec_wins': int((g['win'] != g['win'].shift()).cumsum()
                                       .where(g['win']).dropna()
                                       .groupby((g['win'] != g['win'].shift()).cumsum()).count().max()),
                'max_consec_losses': int((g['win'] != g['win'].shift()).cumsum()
                                         .where(~g['win']).dropna()
                                         .groupby((g['win'] != g['win'].shift()).cumsum()).count().max()),
            })
        pd.DataFrame(wl_rows).to_excel(writer, sheet_name='win_loss', index=False)

        # Sheet 13: drawdown analysis
        print("  Sheet: drawdown")
        dd_rows = []
        for strat, g in all_trades.groupby('strategy'):
            g = g.sort_values('date')
            eq = g['pnl'].cumsum()
            dd = eq - eq.cummax()
            dd_rows.append({
                'strategy': strat,
                'max_drawdown': round(dd.min(), 0),
                'total_pnl': round(g['pnl'].sum(), 0),
                'calmar_ratio': round(g['pnl'].sum() / abs(dd.min()), 2) if dd.min() != 0 else 0,
                'recovery_factor': round(g['pnl'].sum() / abs(dd.min()), 2) if dd.min() != 0 else 0,
            })
        pd.DataFrame(dd_rows).to_excel(writer, sheet_name='drawdown', index=False)

        # Sheet 14: equity_daily — daily P&L and cumsum per strategy
        print("  Sheet: equity_daily")
        all_strats = sorted(all_trades['strategy'].unique())
        eq_daily = pd.DataFrame()
        for strat in all_strats:
            sub = all_trades[all_trades['strategy']==strat].copy()
            sub['dt'] = pd.to_datetime(sub['date'].astype(str), format='mixed')
            d = sub.groupby('dt')['pnl'].sum()
            eq_daily[f'{strat}_pnl'] = d
            eq_daily[f'{strat}_cum'] = d.cumsum()
        # Final system equity
        df_final2 = df_final.copy()
        df_final2['dt'] = pd.to_datetime(df_final2['date'].astype(str), format='mixed')
        d_final = df_final2.groupby('dt')['pnl'].sum()
        eq_daily['final_system_pnl'] = d_final
        eq_daily['final_system_cum'] = d_final.cumsum()
        d_base2 = df_base_only.copy()
        d_base2['dt'] = pd.to_datetime(d_base2['date'].astype(str), format='mixed')
        d_b = d_base2.groupby('dt')['pnl'].sum()
        eq_daily['base_pnl'] = d_b
        eq_daily['base_cum'] = d_b.cumsum()
        eq_daily = eq_daily.reset_index().rename(columns={'index':'date'})
        eq_daily.to_excel(writer, sheet_name='equity_daily', index=False)

        # Sheet 15: blank_deep — blank day analysis
        print("  Sheet: blank_deep")
        if not df_blank.empty:
            df_blank_e = enrich(df_blank.copy(), beh_map)
            pivot_summary(df_blank_e, ['strategy','day_type']).to_excel(
                writer, sheet_name='blank_deep', index=False)

        # Sheet 16: ohl_sweep — from script 118
        print("  Sheet: ohl_sweep")
        try:
            ohl_csv = pd.read_csv(f'{OUT_DIR}/118_ohl_all.csv')
            ohl_pivot = ohl_csv.groupby(['tol_pct','signal']).apply(lambda g: pd.Series({
                'trades': len(g), 'wr_%': round(g['win'].mean()*100,1),
                'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
            })).reset_index()
            ohl_pivot.to_excel(writer, sheet_name='ohl_sweep', index=False)
        except Exception:
            pd.DataFrame({'note':['118_ohl_all.csv not found']}).to_excel(
                writer, sheet_name='ohl_sweep', index=False)

        # Sheet 17: vwap_sweep — from script 118
        print("  Sheet: vwap_sweep")
        try:
            vwap_csv = pd.read_csv(f'{OUT_DIR}/118_vwap_all.csv')
            vwap_pivot = vwap_csv.groupby(['ext_pct','rev_pct']).apply(lambda g: pd.Series({
                'trades': len(g), 'wr_%': round(g['win'].mean()*100,1),
                'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
            })).reset_index().sort_values('total_pnl', ascending=False)
            vwap_pivot.to_excel(writer, sheet_name='vwap_sweep', index=False)
        except Exception:
            pd.DataFrame({'note':['118_vwap_all.csv not found']}).to_excel(
                writer, sheet_name='vwap_sweep', index=False)

        # Sheet 18: signal_calendar — every blank day classified
        print("  Sheet: signal_calendar")
        cal_rows = []
        blank_dates = sorted([d for d in dates_5yr if d in blank_set])
        blank_trades_map = {}
        if not df_blank.empty:
            for d, g in df_blank.groupby('date'):
                blank_trades_map[str(d)] = list(g['strategy'].unique())
        for d in blank_dates:
            b = beh_map.get(d, {})
            cal_rows.append({
                'date': d, 'day_type': b.get('day_type'), 'open_pos': b.get('open_pos'),
                'ib_expand': b.get('ib_expand'), 'cpr_class': b.get('cpr_class'),
                'strategies_fired': ', '.join(blank_trades_map.get(d, ['none'])),
            })
        pd.DataFrame(cal_rows).to_excel(writer, sheet_name='signal_calendar', index=False)

    print(f"  Saved: {xls_path}")
    el_total = (datetime.now()-t_start).total_seconds()

    # ── Final combined picture ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FINAL COMBINED PICTURE")
    print(f"{'='*72}")
    for name, strats in scenarios.items():
        sub = all_trades[all_trades['strategy'].isin(strats)]
        pnl = sub['pnl'].sum()
        base_pnl = all_trades[all_trades['strategy']=='A_base']['pnl'].sum()
        print(f"  {name:<35} | Rs.{pnl:>12,.0f} | +Rs.{pnl-base_pnl:>10,.0f} vs base")

    final_pnl = df_final['pnl'].sum()
    base_pnl2 = df_base_only['pnl'].sum()
    print(f"\n  Base original:       Rs.{base_pnl2:>12,.0f}  {len(df_base_only)}t | WR {df_base_only['win'].mean()*100:.1f}%")

    strat_totals = df_final.groupby('strategy')['pnl'].sum()
    for s in ['B_S3','C_pe','D_base2nd','E_blank2nd']:
        if s in strat_totals.index:
            g = df_final[df_final['strategy']==s]
            print(f"  + {s:<16}: Rs.{strat_totals[s]:>+12,.0f}  {len(g)}t | WR {g['win'].mean()*100:.1f}%")
    print(f"  {'─'*50}")
    print(f"  GRAND TOTAL:         Rs.{final_pnl:>12,.0f}  {len(df_final)}t")
    print(f"  Improvement:         Rs.{final_pnl-base_pnl2:>+12,.0f}  ({(final_pnl/base_pnl2-1)*100:+.1f}%)")

    # ── Equity chart ──────────────────────────────────────────────────────────
    all_idx = pd.date_range(
        start=pd.to_datetime(df_base_only['date'].min(), format='mixed'),
        end=pd.to_datetime(df_final['date'].max(), format='mixed'),
        freq='B'
    )

    def to_cum(df_, fmt='mixed'):
        if df_.empty: return pd.Series(0, index=all_idx)
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_['date'].astype(str), format=fmt)
        d = df_.groupby('dt')['pnl'].sum()
        return d.reindex(all_idx, fill_value=0).cumsum()

    eq_base_c  = to_cum(df_base_only)
    eq_final_c = to_cum(df_final)
    dd_base    = eq_base_c  - eq_base_c.cummax()
    dd_final   = eq_final_c - eq_final_c.cummax()

    print(f"\n  Equity:")
    print(f"    Base:   Rs.{int(eq_base_c.iloc[-1]):,.0f} | DD Rs.{int(dd_base.min()):,.0f} | "
          f"Calmar {eq_base_c.iloc[-1]/abs(dd_base.min()):.1f}")
    print(f"    Final:  Rs.{int(eq_final_c.iloc[-1]):,.0f} | DD Rs.{int(dd_final.min()):,.0f} | "
          f"Calmar {eq_final_c.iloc[-1]/abs(dd_final.min()):.1f}")

    # Add per-year equity breakdown in chart
    lines = [
        {"id":"final","label":f"Final System Rs.{int(eq_final_c.iloc[-1]):,.0f}",
         "color":"#26a69a","seriesType":"line","data":eq_pts(eq_final_c)},
        {"id":"base","label":f"Base Rs.{int(eq_base_c.iloc[-1]):,.0f}",
         "color":"#9e9e9e","seriesType":"line","data":eq_pts(eq_base_c)},
    ]
    # Per strategy incremental
    colors = ['#0ea5e9','#ab47bc','#ff9800','#66bb6a']
    ci = 0
    for s in ['B_S3','C_pe','D_base2nd','E_blank2nd']:
        sub = df_final[df_final['strategy'].isin(['A_base'] +
              [s_ for s_ in ['B_S3','C_pe','D_base2nd','E_blank2nd'] if s_ <= s])]
        eq_s = to_cum(sub)
        if eq_s.iloc[-1] == 0: continue
        lines.append({
            "id": s, "label": f"Cumul +{s} Rs.{int(eq_s.iloc[-1]):,.0f}",
            "color": colors[ci % len(colors)], "seriesType":"line","data": eq_pts(eq_s)
        })
        ci += 1
    lines += [
        {"id":"dd_final","label":f"Final DD Rs.{int(dd_final.min()):,.0f}",
         "color":"#ef5350","seriesType":"baseline","baseValue":0,"isNewPane":True,
         "data":eq_pts(dd_final)},
        {"id":"dd_base","label":f"Base DD Rs.{int(dd_base.min()):,.0f}",
         "color":"#ff9800","seriesType":"baseline","baseValue":0,"isNewPane":True,
         "data":eq_pts(dd_base)},
    ]

    tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}
    send_custom_chart("119_final",  tv_json,
        title=f"Final System | Base {int(eq_base_c.iloc[-1]):,} → "
              f"Combined {int(eq_final_c.iloc[-1]):,} | DD {int(dd_final.min()):,}")

    all_trades.to_csv(f'{OUT_DIR}/119_all_trades.csv', index=False)
    print(f"\n  Saved: 119_final_analysis.xlsx | 119_all_trades.csv")
    print(f"\n  Total runtime: {el_total:.0f}s")
    print("\nDone.")


if __name__ == '__main__':
    main()
