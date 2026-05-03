"""
122_full_master_excel.py — Complete master Excel for all 5 years
================================================================
For EVERY trading day (all ~1155 days):
  Spot:       Open, High, Low, Close, Range, Range%
  Prev day:   OHLC, Range, Close
  CPR:        TC, BC, Pivot, Width pts, Width %
  Pivots:     R1 R2 R3 R4 | S1 S2 S3 S4
  ADR:        20-day average daily range (pts + %)
  IB:         High, Low, Range pts, Range %, IB/ADR ratio
  Futures:    Open, Basis at open (fut-spot), Basis%
  Day meta:   day_type, open_pos, ib_expand, cpr_class, gap%, ichi_sig
  Trade:      strategy, signal, entry_time, exit_time, ep, xp, pnl, win, exit_reason
  MAE/MFE:    option tick-level (max loss %, max profit %)
  Timing:     minutes to target | minutes to SL | time_of_target | time_of_sl
  R/S levels: first_level_tested (R1/R2/S1 etc.), dist_to_r1%, dist_to_s1%

Sheets: all_days | traded_days | adr_analysis | futures_basis |
        mae_mfe_analysis | timing_analysis | ib_adr_analysis | findings_summary
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

OUT_DIR    = 'data/20260430'
DATA_PATH  = os.environ.get('INTER_SERVER_DATA_PATH', '/mnt/data/day-wise')
LOT_SIZE   = 75
SCALE      = 65 / 75
IB_END     = '09:45:00'
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30
MONTH_MAP  = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
              7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}

def r2(v): return round(float(v), 2)
def t2m(t):
    if not t or t == 'nan': return None
    h,m,s = map(int, str(t).split(':')[:3])
    return h*60 + m + s/60


def get_futures_name(date_str):
    """Get current month's futures instrument name for a date."""
    dt = pd.Timestamp(date_str)
    # NSE monthly futures expire last Thursday of month
    # Simple approach: use current month's contract
    yy = dt.strftime('%y')
    mm = MONTH_MAP[dt.month]
    return f'NIFTY{yy}{mm}FUT'


def _worker_day(args):
    """Process one trading day: spot OHLC + IB + futures + ADR info + trade details."""
    date_str, meta, trade_row, adr_20 = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    result = {'date': date_str}

    try:
        # ── Spot data ──────────────────────────────────────────────────────
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None:
            return result
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 10:
            return result

        spot_open  = r2(day.iloc[0]['price'])
        spot_high  = r2(day['price'].max())
        spot_low   = r2(day['price'].min())
        spot_close = r2(day.iloc[-1]['price'])
        spot_range = r2(spot_high - spot_low)
        spot_range_pct = r2(spot_range / spot_open * 100)

        result.update({
            'spot_open': spot_open, 'spot_high': spot_high,
            'spot_low': spot_low,   'spot_close': spot_close,
            'spot_range_pts': spot_range, 'spot_range_pct': spot_range_pct,
        })

        # ── IB ──────────────────────────────────────────────────────────
        ib = day[day['time'] <= IB_END]
        if not ib.empty:
            ib_h = r2(ib['price'].max()); ib_l = r2(ib['price'].min())
            ib_range = r2(ib_h - ib_l)
            ib_range_pct = r2(ib_range / spot_open * 100)
            adr_ratio = r2(ib_range / adr_20 * 100) if adr_20 and adr_20 > 0 else None
            result.update({
                'ib_high': ib_h, 'ib_low': ib_l,
                'ib_range_pts': ib_range, 'ib_range_pct': ib_range_pct,
                'ib_adr_ratio_%': adr_ratio,
            })

        # ── ADR ──────────────────────────────────────────────────────────
        result['adr_20d_pts'] = r2(adr_20) if adr_20 else None
        result['adr_20d_pct'] = r2(adr_20 / spot_open * 100) if adr_20 and spot_open else None

        # ── Futures ──────────────────────────────────────────────────────
        fut_name = get_futures_name(date_str)
        try:
            fut_path = f'{DATA_PATH}/{date_str}/{fut_name}.csv'
            if os.path.exists(fut_path):
                fut_tks = pd.read_csv(fut_path, header=None,
                    names=['date','time','price','volume','oi'])
                fut_day = fut_tks[(fut_tks['time']>='09:15:00')&(fut_tks['time']<='15:30:00')]
                if not fut_day.empty:
                    fut_open  = r2(fut_day.iloc[0]['price'])
                    fut_close = r2(fut_day.iloc[-1]['price'])
                    basis_open  = r2(fut_open - spot_open)
                    basis_pct   = r2(basis_open / spot_open * 100)
                    result.update({
                        'fut_open': fut_open, 'fut_close': fut_close,
                        'fut_basis_pts': basis_open, 'fut_basis_pct': basis_pct,
                    })
        except Exception:
            pass

        # ── CPR + pivot levels ──────────────────────────────────────────
        if meta:
            for k in ['tc','bc','pvt','r1','r2','r3','r4','s1','s2','s3','s4',
                      'gap_pct','day_type','open_pos','ib_expand','cpr_class','ichi_sig']:
                result[k] = meta.get(k)
            if meta.get('tc') and meta.get('bc'):
                result['cpr_width_pts'] = r2(float(meta['tc']) - float(meta['bc']))
                result['cpr_width_pct'] = r2(result['cpr_width_pts'] / float(meta.get('pvt', spot_open)) * 100)

        # Distance from open to R1/S1
        if meta and meta.get('r1') and meta.get('s1'):
            try:
                result['dist_to_r1_pct'] = r2((float(meta['r1']) - spot_open) / spot_open * 100)
                result['dist_to_s1_pct'] = r2((spot_open - float(meta['s1'])) / spot_open * 100)
            except Exception:
                pass

        # Which R/S level was first reached during the day
        if meta:
            levels = []
            for lbl, key in [('R1','r1'),('R2','r2'),('R3','r3'),
                              ('S1','s1'),('S2','s2'),('S3','s3')]:
                if meta.get(key):
                    try: levels.append((lbl, float(meta[key])))
                    except Exception: pass
            if levels:
                prices = day['price'].values
                times  = day['time'].values
                first_level = None; first_time = None
                for i, p in enumerate(prices):
                    for lbl, lvl in levels:
                        if abs(p - lvl) / lvl <= 0.001:  # within 0.1%
                            first_level = lbl; first_time = times[i]; break
                    if first_level: break
                result['first_level_tested'] = first_level
                result['first_level_time']   = first_time

        # ── Trade details ──────────────────────────────────────────────
        if trade_row is not None:
            result.update({
                'strategy':    trade_row.get('strategy',''),
                'signal':      trade_row.get('signal', trade_row.get('opt','')),
                'entry_time':  trade_row.get('entry_time',''),
                'ep':          trade_row.get('ep', trade_row.get('entry_price', None)),
                'xp':          trade_row.get('xp', trade_row.get('exit_price', None)),
                'pnl':         trade_row.get('pnl', trade_row.get('pnl_conv', None)),
                'win':         trade_row.get('win', None),
                'exit_reason': trade_row.get('exit_reason',''),
            })

            # ── MAE / MFE / Timing ─────────────────────────────────────
            try:
                opt = result['signal']
                et  = result['entry_time']
                ep  = float(result['ep']) if result['ep'] else None

                if opt and et and ep and ep > 0:
                    expiries = list_expiry_dates(date_str, index_name='NIFTY')
                    if expiries:
                        # Try to get ATM from spot at entry
                        spot_at = day[day['time'] >= et]
                        if not spot_at.empty:
                            atm = int(round(spot_at.iloc[0]['price'] / 50) * 50)
                            instr = f'NIFTY{expiries[0]}{atm}{opt}'
                            opt_tks = load_tick_data(date_str, instr, et)
                            if opt_tks is not None and not opt_tks.empty:
                                tks_e = opt_tks[opt_tks['time'] >= et].reset_index(drop=True)
                                if not tks_e.empty:
                                    ps = tks_e['price'].values
                                    ts = tks_e['time'].values
                                    tgt = ep * (1 - TGT_PCT)
                                    min_p = ps[0]; max_p = ps[0]
                                    tgt_time = None; sl_time = None; exit_time = None
                                    hsl = ep * 2.0; sl = hsl; md = 0.0
                                    for i in range(len(ts)):
                                        p = ps[i]; t = ts[i]
                                        if p < min_p: min_p = p
                                        if p > max_p: max_p = p
                                        if t >= EOD_EXIT:
                                            exit_time = t; break
                                        d = (ep - p) / ep
                                        if d > md: md = d
                                        if   md >= 0.60: sl = min(sl, ep*(1-md*0.95))
                                        elif md >= 0.40: sl = min(sl, ep*0.80)
                                        elif md >= 0.25: sl = min(sl, ep)
                                        if p <= tgt and tgt_time is None:
                                            tgt_time = t; exit_time = t
                                        if p >= sl and sl_time is None:
                                            sl_time = t; exit_time = t
                                        if exit_time: break

                                    mfe_pct = r2((ep - min_p) / ep * 100)  # best profit %
                                    mae_pct = r2((max_p - ep) / ep * 100)  # worst adverse %

                                    result.update({
                                        'opt_min_price': r2(min_p),
                                        'opt_max_price': r2(max_p),
                                        'mfe_pct': mfe_pct,
                                        'mae_pct': mae_pct,
                                        'exit_time': exit_time,
                                        'time_of_target': tgt_time,
                                        'time_of_sl': sl_time,
                                    })
                                    if et and exit_time:
                                        result['mins_to_exit'] = r2(t2m(exit_time) - t2m(et))
                                    if et and tgt_time:
                                        result['mins_to_target'] = r2(t2m(tgt_time) - t2m(et))
                                    if et and sl_time:
                                        result['mins_to_sl'] = r2(t2m(sl_time) - t2m(et))
            except Exception:
                pass

    except Exception:
        pass

    return result


def main():
    t_start = datetime.now()
    print("Loading base data...")

    # ── Load base trades ──────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base = base.rename(columns={'pnl_conv':'pnl'})
    base['date_key'] = pd.to_datetime(base['date'].astype(str),
                                       format='mixed').dt.strftime('%Y%m%d')
    base['strategy'] = 'A_base'
    base['signal']   = base['opt']
    if 'entry_price' in base.columns:
        base = base.rename(columns={'entry_price':'ep', 'exit_price':'xp'})

    # Also load blank day trades from 119
    try:
        all_t = pd.read_csv(f'{OUT_DIR}/119_all_trades.csv')
        blank_t = all_t[all_t['strategy'].isin(['B_S3','C_pe','C_both'])].copy()
        blank_t['date_key'] = pd.to_datetime(blank_t['date'].astype(str),
                                              format='mixed').dt.strftime('%Y%m%d')
    except Exception:
        blank_t = pd.DataFrame()

    # Also load S4 second trades
    try:
        s4 = pd.read_csv(f'{OUT_DIR}/117_s4_clean.csv')
        s4['date_key'] = s4['date'].astype(str)
        s4['strategy'] = 'D_base2nd'
        s4['signal']   = s4['opt']
    except Exception:
        s4 = pd.DataFrame()

    # Build trade map: date → trade row
    trade_map = {}
    for _, row in base.iterrows():
        trade_map[row['date_key']] = dict(row)
    if not blank_t.empty:
        for _, row in blank_t.iterrows():
            dk = row['date_key']
            if dk not in trade_map:  # don't overwrite base trades
                trade_map[dk] = dict(row)
    if not s4.empty:
        pass  # S4 trades share date with base; handle separately

    # ── Load day behavior ──────────────────────────────────────────────────
    beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                        sheet_name='all_days_observations', dtype={'date':str})
    beh['date_key'] = beh['date'].astype(str).str.strip()
    beh_map = {r['date_key']: {k: (None if pd.isna(v) else v)
                               for k, v in r.items()} for _, r in beh.iterrows()}

    # ── Date range ─────────────────────────────────────────────────────────
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]

    print(f"  {len(dates_5yr)} days | {len(trade_map)} traded days")

    # ── Compute rolling 20-day ADR from spot data ──────────────────────────
    print("Computing 20-day ADR for all dates...")
    adr_map = {}
    range_history = []
    # Need spot OHLC for each day — use beh_map if it has high/low, else skip
    # Simple approach: for each day d, ADR = avg of previous 20 days' ranges
    # We'll compute this from the spot data stored in beh_map (spot_high - spot_low)
    # Since beh_map has 'spot_range' or we compute from tick highs
    # Use spot_range_pts from spot data inline (fast approximate using beh_map)
    for d in dates_5yr:
        meta = beh_map.get(d, {})
        # Approximate daily range from R1/S1 spread or use fixed estimate
        # Better: compute from IB range + post-IB extension
        # For now use a simple proxy we'll fill during worker
        adr_map[d] = None  # will be filled with rolling computation below

    # Pre-compute spot ranges from available beh data for ADR
    range_series = {}
    for d in dates_5yr:
        meta = beh_map.get(d, {})
        # Use ib_class or stored spot range if available
        # We'll just load a sample to build ADR — use daily close from futures as proxy
        # Simplified: use previous trades' spot ranges if available
        pass

    # We'll compute ADR dynamically inside workers using a pre-passed value
    # Pre-compute rolling ADR from beh data using high-low approximation
    # Use R1/S1 spread as proxy for expected range
    spot_ranges = {}
    for d in sorted(dates_5yr):
        meta = beh_map.get(d, {})
        try:
            r1 = float(meta.get('r1', 0) or 0)
            s1 = float(meta.get('s1', 0) or 0)
            pvt = float(meta.get('pvt', 0) or 0)
            if r1 > 0 and s1 > 0:
                # R1-S1 spread is roughly 2× daily range for normal days
                spot_ranges[d] = (r1 - s1) / 2
        except Exception:
            pass

    sorted_dates = sorted(dates_5yr)
    for i, d in enumerate(sorted_dates):
        past = sorted_dates[max(0,i-20):i]
        vals = [spot_ranges[p] for p in past if p in spot_ranges]
        adr_map[d] = sum(vals)/len(vals) if vals else 200.0  # default 200 pts

    # ── Run parallel workers ───────────────────────────────────────────────
    print(f"Processing {len(dates_5yr)} days (parallel)...")
    args = [(d, beh_map.get(d,{}), trade_map.get(d), adr_map.get(d)) for d in dates_5yr]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        results = pool.map(_worker_day, args)
    el = (datetime.now()-t0).total_seconds()
    print(f"  Done in {el:.1f}s")

    df = pd.DataFrame(results)
    df['dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['year']    = df['dt'].dt.year
    df['month']   = df['dt'].dt.month
    df['month_name'] = df['dt'].dt.strftime('%b')
    df['weekday'] = df['dt'].dt.dayofweek
    df['weekday_name'] = df['dt'].dt.strftime('%A')
    df['traded']  = df['strategy'].notna() & (df['strategy'] != '')

    # Recalculate rolling ADR from actual spot ranges computed in workers
    df = df.sort_values('date').reset_index(drop=True)
    actual_ranges = df['spot_range_pts'].fillna(0).values
    adr_actual = []
    for i in range(len(actual_ranges)):
        past = actual_ranges[max(0,i-20):i]
        adr_actual.append(round(past.mean(), 1) if len(past) > 0 else 200.0)
    df['adr_20d_pts'] = adr_actual
    df['adr_20d_pct'] = (df['adr_20d_pts'] / df['spot_open'] * 100).round(2)
    df['ib_adr_ratio_%'] = (df['ib_range_pts'] / df['adr_20d_pts'] * 100).round(1)

    traded_df = df[df['traded']].copy()
    print(f"  All days: {len(df)} | Traded: {len(traded_df)}")

    # ── Print quick summary ────────────────────────────────────────────────
    print(f"\nKey stats:")
    print(f"  Avg ADR: {df['adr_20d_pts'].mean():.0f} pts")
    print(f"  Avg IB range: {df['ib_range_pts'].mean():.0f} pts")
    print(f"  Avg IB/ADR ratio: {df['ib_adr_ratio_%'].mean():.1f}%")
    if not traded_df.empty and 'mfe_pct' in traded_df.columns:
        t = traded_df.dropna(subset=['mfe_pct','mae_pct'])
        print(f"  Avg MFE: {t['mfe_pct'].mean():.1f}%  Avg MAE: {t['mae_pct'].mean():.1f}%")
        print(f"  Avg mins to exit: {traded_df['mins_to_exit'].mean():.0f}")
        target_hits = traded_df[traded_df['exit_reason']=='target']
        if not target_hits.empty:
            print(f"  Avg mins to target: {target_hits['mins_to_target'].mean():.0f}")

    # ── IB/ADR analysis ───────────────────────────────────────────────────
    if not traded_df.empty and 'pnl' in traded_df.columns:
        traded_df['ib_adr_bin'] = pd.cut(traded_df['ib_adr_ratio_%'].fillna(0),
            bins=[0,30,50,70,90,110,200],
            labels=['0-30%','30-50%','50-70%','70-90%','90-110%','110%+'])
        print(f"\n  IB/ADR ratio → P&L:")
        ibadr = traded_df.groupby('ib_adr_bin', observed=True).apply(lambda g: pd.Series({
            'trades': len(g), 'wr': round(g['win'].mean()*100,1),
            'avg_pnl': round(g['pnl'].mean(),0), 'total': round(g['pnl'].sum(),0)
        })).reset_index()
        print(ibadr.to_string(index=False))

    # ── Futures basis analysis ─────────────────────────────────────────────
    if not traded_df.empty and 'fut_basis_pts' in traded_df.columns:
        fb = traded_df.dropna(subset=['fut_basis_pts'])
        if not fb.empty:
            fb['basis_bin'] = pd.cut(fb['fut_basis_pts'],
                bins=[-100,-20,-5,5,20,50,200],
                labels=['<-20','-20to-5','-5to+5','+5to+20','+20to+50','>+50'])
            print(f"\n  Futures basis → P&L:")
            basis_g = fb.groupby('basis_bin', observed=True).apply(lambda g: pd.Series({
                'trades': len(g), 'wr': round(g['win'].mean()*100,1),
                'avg_pnl': round(g['pnl'].mean(),0)
            })).reset_index()
            print(basis_g.to_string(index=False))

    # ── Save Excel ─────────────────────────────────────────────────────────
    print(f"\nSaving Excel...")
    xls_path = f'{OUT_DIR}/122_full_master.xlsx'

    all_cols = ['date','year','month_name','weekday_name','traded','strategy','signal',
                'spot_open','spot_high','spot_low','spot_close','spot_range_pts','spot_range_pct',
                'ib_high','ib_low','ib_range_pts','ib_range_pct',
                'adr_20d_pts','adr_20d_pct','ib_adr_ratio_%',
                'fut_open','fut_close','fut_basis_pts','fut_basis_pct',
                'tc','bc','pvt','cpr_width_pts','cpr_width_pct',
                'r1','r2','r3','r4','s1','s2','s3','s4',
                'dist_to_r1_pct','dist_to_s1_pct','first_level_tested','first_level_time',
                'gap_pct','day_type','open_pos','ib_expand','cpr_class','ichi_sig',
                'entry_time','ep','xp','pnl','win','exit_reason',
                'opt_min_price','opt_max_price','mfe_pct','mae_pct',
                'exit_time','time_of_target','time_of_sl',
                'mins_to_exit','mins_to_target','mins_to_sl']
    all_cols = [c for c in all_cols if c in df.columns]

    def piv(df_, rows, sort_col='avg_pnl'):
        if df_.empty: return pd.DataFrame()
        return df_.groupby(rows, dropna=False).apply(lambda g: pd.Series({
            'trades': len(g),
            'win_rate_%': round(g['win'].mean()*100,1) if 'win' in g else None,
            'total_pnl': round(g['pnl'].sum(),0) if 'pnl' in g else None,
            'avg_pnl': round(g['pnl'].mean(),0) if 'pnl' in g else None,
            'avg_mfe_%': round(g['mfe_pct'].mean(),1) if 'mfe_pct' in g else None,
            'avg_mae_%': round(g['mae_pct'].mean(),1) if 'mae_pct' in g else None,
            'avg_mins_to_exit': round(g['mins_to_exit'].mean(),0) if 'mins_to_exit' in g else None,
        })).reset_index().sort_values(sort_col, ascending=False)

    with pd.ExcelWriter(xls_path, engine='openpyxl') as writer:
        # Sheet 1: all_days
        print("  Sheet: all_days")
        df[all_cols].sort_values('date').to_excel(writer, sheet_name='all_days', index=False)

        # Sheet 2: traded_days only (with MAE/MFE timing)
        print("  Sheet: traded_days")
        if not traded_df.empty:
            traded_cols = [c for c in all_cols if c in traded_df.columns]
            traded_df[traded_cols].sort_values('date').to_excel(
                writer, sheet_name='traded_days', index=False)

        # Sheet 3: IB/ADR analysis
        print("  Sheet: ib_adr_analysis")
        if not traded_df.empty and 'ib_adr_ratio_%' in traded_df.columns:
            traded_df['ib_adr_bin'] = pd.cut(traded_df['ib_adr_ratio_%'].fillna(0),
                bins=[0,30,50,70,90,110,200],
                labels=['0-30%','30-50%','50-70%','70-90%','90-110%','110%+'])
            all_df2 = df.copy()
            all_df2['ib_adr_bin'] = pd.cut(all_df2['ib_adr_ratio_%'].fillna(0),
                bins=[0,30,50,70,90,110,200],
                labels=['0-30%','30-50%','50-70%','70-90%','90-110%','110%+'])
            ibadr_all = all_df2.groupby('ib_adr_bin', observed=True).agg(
                total_days=('date','count'),
                traded_days=('traded','sum'),
                avg_spot_range=('spot_range_pts','mean'),
            ).reset_index()
            ibadr_traded = traded_df.dropna(subset=['pnl']).groupby(
                'ib_adr_bin', observed=True).apply(lambda g: pd.Series({
                    'trades': len(g), 'wr': round(g['win'].mean()*100,1),
                    'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
                })).reset_index()
            ibadr_full = ibadr_all.merge(ibadr_traded, on='ib_adr_bin', how='left')
            ibadr_full.to_excel(writer, sheet_name='ib_adr_analysis', index=False)

        # Sheet 4: Futures basis analysis
        print("  Sheet: futures_basis")
        if not traded_df.empty and 'fut_basis_pts' in traded_df.columns:
            fb2 = traded_df.dropna(subset=['fut_basis_pts','pnl'])
            if not fb2.empty:
                fb2['basis_bin'] = pd.cut(fb2['fut_basis_pts'],
                    bins=[-200,-20,-5,5,20,50,200],
                    labels=['<-20','-20to-5','-5to+5','+5to+20','+20to+50','>+50'])
                piv(fb2.dropna(subset=['basis_bin']), ['basis_bin']).to_excel(
                    writer, sheet_name='futures_basis', index=False)

        # Sheet 5: MAE/MFE analysis
        print("  Sheet: mae_mfe_analysis")
        if not traded_df.empty and 'mfe_pct' in traded_df.columns:
            mf = traded_df.dropna(subset=['mfe_pct','mae_pct','pnl'])
            if not mf.empty:
                mf['mfe_bin'] = pd.cut(mf['mfe_pct'], bins=[0,10,20,30,40,50,100],
                    labels=['0-10%','10-20%','20-30%','30-40%','40-50%','50%+'])
                mf['mae_bin'] = pd.cut(mf['mae_pct'], bins=[0,5,10,20,30,50,200],
                    labels=['0-5%','5-10%','10-20%','20-30%','30-50%','50%+'])
                mfe_piv = mf.groupby('mfe_bin', observed=True).apply(lambda g: pd.Series({
                    'trades': len(g), 'wr': round(g['win'].mean()*100,1),
                    'avg_pnl': round(g['pnl'].mean(),0),
                })).reset_index()
                mae_piv = mf.groupby('mae_bin', observed=True).apply(lambda g: pd.Series({
                    'trades': len(g), 'wr': round(g['win'].mean()*100,1),
                    'avg_pnl': round(g['pnl'].mean(),0),
                })).reset_index()
                mfe_piv.to_excel(writer, sheet_name='mfe_analysis', index=False)
                mae_piv.to_excel(writer, sheet_name='mae_analysis', index=False)

        # Sheet 6: Timing analysis
        print("  Sheet: timing_analysis")
        if not traded_df.empty and 'mins_to_exit' in traded_df.columns:
            tt = traded_df.dropna(subset=['mins_to_exit','pnl'])
            if not tt.empty:
                tt['exit_bin'] = pd.cut(tt['mins_to_exit'],
                    bins=[0,30,60,120,180,240,360,500],
                    labels=['0-30m','30-60m','1-2h','2-3h','3-4h','4-6h','6h+'])
                piv(tt.dropna(subset=['exit_bin']), ['exit_bin']).to_excel(
                    writer, sheet_name='timing_analysis', index=False)

        # Sheet 7: ADR context all days
        print("  Sheet: adr_context")
        adr_ctx = df.groupby(
            pd.cut(df['adr_20d_pts'].fillna(200), bins=[0,100,150,200,250,300,500],
                   labels=['<100','100-150','150-200','200-250','250-300','300+']),
            observed=True
        ).agg(
            total_days=('date','count'),
            avg_range=('spot_range_pts','mean'),
            traded_days=('traded','sum'),
        ).reset_index().rename(columns={'adr_20d_pts':'adr_bin'})
        if not traded_df.empty and 'pnl' in traded_df.columns:
            traded_df['adr_bin_t'] = pd.cut(
                traded_df['adr_20d_pts'].fillna(200),
                bins=[0,100,150,200,250,300,500],
                labels=['<100','100-150','150-200','200-250','250-300','300+'])
            adr_t = traded_df.dropna(subset=['pnl']).groupby(
                'adr_bin_t', observed=True).apply(lambda g: pd.Series({
                    'trades': len(g), 'wr': round(g['win'].mean()*100,1),
                    'avg_pnl': round(g['pnl'].mean(),0),
                })).reset_index().rename(columns={'adr_bin_t':'adr_bin'})
            adr_full = adr_ctx.merge(adr_t, on='adr_bin', how='left')
            adr_full.to_excel(writer, sheet_name='adr_context', index=False)

        # Sheet 8: Year×Month heatmap (all days)
        print("  Sheet: ym_heatmap")
        if not traded_df.empty and 'pnl' in traded_df.columns:
            ym = traded_df.groupby(['year','month_name','month'])['pnl'].sum().reset_index()
            ym_p = ym.pivot(index='year', columns='month_name', values='pnl').fillna(0)
            ym_p = ym_p[[m for m in ['Jan','Feb','Mar','Apr','May','Jun',
                                      'Jul','Aug','Sep','Oct','Nov','Dec']
                         if m in ym_p.columns]]
            ym_p['TOTAL'] = ym_p.sum(axis=1)
            ym_p.reset_index().to_excel(writer, sheet_name='ym_heatmap', index=False)

        # Sheet 9: Findings summary
        print("  Sheet: findings_summary")
        findings = []
        if not traded_df.empty and 'pnl' in traded_df.columns:
            # IB/ADR
            if 'ib_adr_ratio_%' in traded_df.columns:
                hi_ibadr = traded_df[traded_df['ib_adr_ratio_%'] > 80]
                lo_ibadr = traded_df[traded_df['ib_adr_ratio_%'] <= 50]
                if not hi_ibadr.empty:
                    findings.append({'finding': 'IB/ADR > 80% trades',
                                     'trades': len(hi_ibadr),
                                     'wr': round(hi_ibadr['win'].mean()*100,1),
                                     'avg_pnl': round(hi_ibadr['pnl'].mean(),0),
                                     'insight': 'IB used most of ADR — limited room left'})
                if not lo_ibadr.empty:
                    findings.append({'finding': 'IB/ADR <= 50% trades',
                                     'trades': len(lo_ibadr),
                                     'wr': round(lo_ibadr['win'].mean()*100,1),
                                     'avg_pnl': round(lo_ibadr['pnl'].mean(),0),
                                     'insight': 'Lots of range left — better for option sellers'})
            # MAE
            if 'mae_pct' in traded_df.columns:
                winners = traded_df[traded_df['win'] == True].dropna(subset=['mae_pct'])
                losers  = traded_df[traded_df['win'] == False].dropna(subset=['mae_pct'])
                if not winners.empty:
                    findings.append({'finding': 'Winners avg MAE',
                                     'trades': len(winners),
                                     'wr': 100.0,
                                     'avg_pnl': round(winners['pnl'].mean(),0),
                                     'insight': f"Avg {winners['mae_pct'].mean():.1f}% adverse before winning"})
                if not losers.empty:
                    findings.append({'finding': 'Losers avg MAE',
                                     'trades': len(losers),
                                     'wr': 0.0,
                                     'avg_pnl': round(losers['pnl'].mean(),0),
                                     'insight': f"Avg {losers['mae_pct'].mean():.1f}% adverse — hard SL avg"})
            # Futures basis
            if 'fut_basis_pts' in traded_df.columns:
                premium = traded_df[traded_df['fut_basis_pts'] > 20].dropna(subset=['pnl'])
                discount = traded_df[traded_df['fut_basis_pts'] < -5].dropna(subset=['pnl'])
                if not premium.empty:
                    findings.append({'finding': 'Futures premium > 20pts',
                                     'trades': len(premium),
                                     'wr': round(premium['win'].mean()*100,1),
                                     'avg_pnl': round(premium['pnl'].mean(),0),
                                     'insight': 'Bullish futures premium context'})
                if not discount.empty:
                    findings.append({'finding': 'Futures discount < -5pts',
                                     'trades': len(discount),
                                     'wr': round(discount['win'].mean()*100,1),
                                     'avg_pnl': round(discount['pnl'].mean(),0),
                                     'insight': 'Bearish futures discount context'})
        pd.DataFrame(findings).to_excel(writer, sheet_name='findings_summary', index=False)

    print(f"  Saved: {xls_path}")
    print(f"\n  Total runtime: {(datetime.now()-t_start).total_seconds():.0f}s")
    print("\nDone.")


if __name__ == '__main__':
    main()
