"""
Artha — Vyuha Zone Charts (Matplotlib PNG)
===========================================
Per v17a trade → 3 clean PNG images in year/date subfolder:
  01_spot.png    — NIFTY 1-min line + all levels
  02_option.png  — Option CE/PE price + SL + Target + Entry/Exit
  03_summary.png — Trade card (all key numbers)

Folder: artha/vyuha/{tamil_zone}/{year}/{YYYYMMDD}/
"""
import sys, os, warnings, argparse
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np

ZONE_TAMIL = {
    'above_r4':'thor','r3_to_r4':'ironman','r2_to_r3':'captainmarvel',
    'r1_to_r2':'spiderman','pdh_to_r1':'blackpanther','tc_to_pdh':'hawkeye',
    'within_cpr':'vision','pdl_to_bc':'antman','pdl_to_s1':'blackwidow',
    's1_to_s2':'hulk','s2_to_s3':'wintersoldier','s3_to_s4':'loki',
    'below_s4':'thanos',
}
ZONE_DISPLAY = {
    'above_r4':'Thor (above R4)','r3_to_r4':'Iron Man (R3-R4)',
    'r2_to_r3':'Captain Marvel (R2-R3)','r1_to_r2':'Spider-Man (R1-R2)',
    'pdh_to_r1':'Black Panther (PDH-R1)','tc_to_pdh':'Hawkeye (TC-PDH)',
    'within_cpr':'Vision (CPR)','pdl_to_bc':'Ant-Man (PDL-BC)',
    'pdl_to_s1':'Black Widow (PDL-S1)','s1_to_s2':'Hulk (S1-S2)',
    's2_to_s3':'Winter Soldier (S2-S3)','s3_to_s4':'Loki (S3-S4)',
    'below_s4':'Thanos (below S4)',
}

BG   = '#0d1117'
GRID = '#21262d'
TEXT = '#e6edf3'
DIM  = '#8b949e'

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp=r2((h+l+c)/3); bc=r2((h+l)/2); tc=r2(2*pp-bc)
    r1=r2(2*pp-l); r2_=r2(pp+(h-l)); r3=r2(r1+(h-l)); r4=r2(r2_+(h-l))
    s1=r2(2*pp-h); s2_=r2(pp-(h-l)); s3=r2(s1-(h-l)); s4=r2(s2_-(h-l))
    return dict(pp=pp,bc=bc,tc=tc,r1=r1,r2=r2_,r3=r3,r4=r4,s1=s1,s2=s2_,s3=s3,s4=s4)

def compute_camarilla(h, l, c):
    rng=h-l; return dict(h3=r2(c+rng*1.1/4), l3=r2(c-rng*1.1/4))

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=7)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.7)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')


def save_spot(path, dstr, ohlc, pvt, cam, ph, pl, e20, entry_dt, exit_dt, pnl, zone, bias):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    won = pnl > 0
    spot_col = '#26a69a' if ohlc.iloc[-1] >= ohlc.iloc[0] else '#ef5350'
    ax.plot(ohlc.index, ohlc.values, color=spot_col, linewidth=1.2, label='NIFTY', zorder=5)

    levels = [
        ('R4', pvt['r4'], '#c62828', ':', 0.8),
        ('R3', pvt['r3'], '#e53935', ':', 0.8),
        ('R2', pvt['r2'], '#ef5350', '--', 1.0),
        ('R1', pvt['r1'], '#ef5350', '-', 1.3),
        ('TC', pvt['tc'], '#FFA726', '-', 1.8),
        ('PP', pvt['pp'], '#78909C', '--', 0.9),
        ('BC', pvt['bc'], '#FFA726', '-', 1.8),
        ('S1', pvt['s1'], '#26a69a', '-', 1.3),
        ('S2', pvt['s2'], '#26a69a', '--', 1.0),
        ('S3', pvt['s3'], '#1b7a6e', ':', 0.8),
        ('S4', pvt['s4'], '#1b7a6e', ':', 0.8),
        ('PDH', ph,       '#e91e63', '-.', 1.4),
        ('PDL', pl,       '#1e88e5', '-.', 1.4),
        (f'EMA {e20:.0f}', e20, '#4BC0C0', '-', 1.4),
        ('H3', cam['h3'], '#ce93d8', ':', 1.0),
        ('L3', cam['l3'], '#80cbc4', ':', 1.0),
    ]
    t0, t1 = ohlc.index[0], ohlc.index[-1]
    for lbl, val, col, ls, lw in levels:
        ax.axhline(val, color=col, linestyle=ls, linewidth=lw, alpha=0.85, label=f'{lbl} {val:.0f}')

    # Entry marker
    if entry_dt in ohlc.index or (entry_dt >= ohlc.index[0] and entry_dt <= ohlc.index[-1]):
        ev = float(ohlc.asof(entry_dt))
        ax.scatter([entry_dt], [ev], marker='v', color='#FF9F40', s=120, zorder=10)
        ax.annotate('Entry', (entry_dt, ev), textcoords='offset points',
                    xytext=(0, 8), color='#FF9F40', fontsize=7, ha='center')

    # Exit marker
    if exit_dt >= ohlc.index[0] and exit_dt <= ohlc.index[-1]:
        exv = float(ohlc.asof(exit_dt))
        ecol = '#26a69a' if won else '#ef5350'
        ax.scatter([exit_dt], [exv], marker='o', color=ecol, s=100, zorder=10)
        ax.annotate('Exit', (exit_dt, exv), textcoords='offset points',
                    xytext=(0, -12), color=ecol, fontsize=7, ha='center')

    tamil = ZONE_DISPLAY.get(zone, zone)
    status = 'WIN' if won else 'LOSS'
    ax.set_title(f"NIFTY Spot | {dstr} | {tamil} | Bias: {bias.upper()} | {status} Rs{pnl:+,.0f}",
                 color=TEXT, fontsize=10, pad=8)
    ax.set_ylabel('Price', color=DIM, fontsize=8)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=6.5,
              facecolor='#161b22', edgecolor=GRID, labelcolor=TEXT, ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def save_option(path, dstr, opt_tks, ep, pnl, reason, opt, strike, stype):
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    won    = pnl > 0
    tgt    = r2(ep * 0.50)
    sl     = r2(ep * 1.50)
    xp     = float(opt_tks['price'].iloc[-1])
    t0, t1 = opt_tks['dt'].iloc[0], opt_tks['dt'].iloc[-1]

    ax.plot(opt_tks['dt'], opt_tks['price'].astype(float),
            color='#FF9F40', linewidth=1.8, label=f'{opt} {strike}')
    ax.axhline(ep,  color='#FF9F40', linestyle=':', linewidth=1.2, label=f'EP {ep:.0f}')
    ax.axhline(tgt, color='#26a69a', linestyle='--', linewidth=1.4, label=f'Target {tgt:.0f}')
    ax.axhline(sl,  color='#ef5350', linestyle='--', linewidth=1.4, label=f'SL {sl:.0f}')

    ax.scatter([t0], [ep],  marker='^', color='#FF9F40', s=120, zorder=10)
    ax.annotate(f'Entry {ep:.0f}', (t0, ep), textcoords='offset points',
                xytext=(0, 8), color='#FF9F40', fontsize=7, ha='center')

    ecol = '#26a69a' if won else '#ef5350'
    ax.scatter([t1], [xp], marker='o', color=ecol, s=100, zorder=10)
    ax.annotate(f'Exit {xp:.0f}', (t1, xp), textcoords='offset points',
                xytext=(0, -12), color=ecol, fontsize=7, ha='center')

    status = 'WIN' if won else 'LOSS'
    ax.set_title(f"{opt} {strike} ({stype}) | EP={ep} XP={xp:.0f} | "
                 f"{status} Rs{pnl:+,.0f} | {reason.upper()}",
                 color=TEXT, fontsize=10, pad=8)
    ax.set_ylabel('Premium (Rs)', color=DIM, fontsize=8)
    ax.legend(loc='upper right', fontsize=8, facecolor='#161b22',
              edgecolor=GRID, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def save_summary(path, row, pvt, cam, ph, pl, e20, expiry, strike, dte):
    fig, ax = plt.subplots(figsize=(5, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    won   = row['pnl'] > 0
    color = '#26a69a' if won else '#ef5350'
    pnl   = float(row['pnl'])
    ep    = float(row['ep'])
    xp    = float(row['xp'])
    tamil = ZONE_DISPLAY.get(row['zone'], row['zone'])

    ax.text(0.5, 0.97, 'WIN' if won else 'LOSS',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=22, fontweight='bold', color=color)
    ax.text(0.5, 0.90, f"Rs {pnl:+,.0f}",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=16, color=color)

    ax.axhline(0.845, color=GRID, linewidth=0.8, xmin=0.05, xmax=0.95)

    rows = [
        ('Date',         row['date'].strftime('%d %b %Y (%A)')),
        ('Zone',         tamil),
        ('Bias',         row['bias'].upper()),
        ('Signal',       f"Sell {row['opt']} {row['strike_type']}"),
        ('Entry Time',   row['entry_time']),
        ('Expiry',       expiry),
        ('Strike',       str(strike)),
        ('DTE',          str(dte)),
        ('Entry Price',  f"Rs {ep:.2f}"),
        ('Exit Price',   f"Rs {xp:.2f}"),
        ('Exit Reason',  row['exit_reason'].upper()),
        ('', ''),
        ('PP',           f"{pvt['pp']:.0f}"),
        ('TC / BC',      f"{pvt['tc']:.0f} / {pvt['bc']:.0f}"),
        ('R1 / S1',      f"{pvt['r1']:.0f} / {pvt['s1']:.0f}"),
        ('PDH / PDL',    f"{ph:.0f} / {pl:.0f}"),
        ('EMA(20)',       f"{e20:.0f}"),
        ('Cam H3 / L3',  f"{cam['h3']:.0f} / {cam['l3']:.0f}"),
    ]

    y = 0.80
    step = 0.044
    for lbl, val in rows:
        if lbl == '':
            ax.axhline(y + step*0.4, color=GRID, linewidth=0.6,
                       xmin=0.05, xmax=0.95)
            y -= step; continue
        ax.text(0.38, y, lbl, transform=ax.transAxes,
                ha='right', va='center', fontsize=8, color=DIM)
        ax.text(0.41, y, val, transform=ax.transAxes,
                ha='left', va='center', fontsize=8, color=TEXT, fontweight='bold')
        y -= step

    ax.set_title(f"Trade Summary — {row['date'].strftime('%Y-%m-%d')}",
                 color=TEXT, fontsize=10, pad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def process_trade(row, all_dates, daily_ohlc, ema_s):
    date  = row['date'].strftime('%Y%m%d')
    dstr  = row['date'].strftime('%Y-%m-%d')
    year  = row['date'].strftime('%Y')
    zone  = row['zone']
    tamil = ZONE_TAMIL.get(zone, zone)

    out_dir = f'artha/vyuha/{tamil}/{year}/{date}'
    os.makedirs(out_dir, exist_ok=True)

    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty: return False
    spot_tks = spot_tks[(spot_tks['time'] >= '09:15:00') &
                        (spot_tks['time'] <= '15:30:00')].copy()
    spot_tks['dt'] = pd.to_datetime(dstr + ' ' + spot_tks['time'])
    ohlc = spot_tks.set_index('dt')['price'].resample('1min').last().dropna()
    if ohlc.empty: return False

    idx  = all_dates.index(date)
    prev = all_dates[idx-1]
    if prev not in daily_ohlc: return False
    ph, pl, pc, _ = daily_ohlc[prev]
    _, _, _, today_op = daily_ohlc[date]
    pvt = compute_pivots(ph, pl, pc)
    cam = compute_camarilla(ph, pl, pc)
    e20 = float(ema_s.get(date, np.nan))
    if np.isnan(e20): return False

    exps = list_expiry_dates(date)
    if not exps: return False
    expiry = exps[0]
    exp_dt = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte    = (exp_dt - row['date']).days

    atm    = int(round(today_op / 50) * 50)
    off    = {'ATM': 0, 'OTM1': 50, 'ITM1': -50}
    opt    = row['opt']
    stype  = row['strike_type']
    strike = atm + off.get(stype, 0) if opt == 'CE' else atm - off.get(stype, 0)
    inst   = f'NIFTY{expiry}{strike}{opt}'

    opt_tks = load_tick_data(date, inst, '09:15:00', '15:30:00')
    if opt_tks is None or opt_tks.empty: return False
    opt_tks['dt'] = pd.to_datetime(dstr + ' ' + opt_tks['time'])
    opt_tks = opt_tks[opt_tks['time'] >= row['entry_time']].sort_values('dt')
    if opt_tks.empty: return False

    entry_dt = pd.Timestamp(dstr + ' ' + row['entry_time'])
    exit_dt  = opt_tks['dt'].iloc[-1]

    save_spot(f'{out_dir}/01_spot.png', dstr, ohlc, pvt, cam, ph, pl, e20,
              entry_dt, exit_dt, float(row['pnl']), zone, row['bias'])

    save_option(f'{out_dir}/02_option.png', dstr, opt_tks, float(row['ep']),
                float(row['pnl']), row['exit_reason'], opt, strike, stype)

    save_summary(f'{out_dir}/03_summary.png', row, pvt, cam, ph, pl, e20,
                 expiry, strike, dte)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--zone', default=None)
    args = parser.parse_args()

    trades = pd.read_csv('data/20260428/56_combined_trades.csv', parse_dates=['date'])
    trades = trades[trades['strategy'] == 'v17a'].sort_values('date').reset_index(drop=True)

    if args.zone:
        rev = {v: k for k, v in ZONE_TAMIL.items()}
        eng = rev.get(args.zone.lower())
        if eng: trades = trades[trades['zone'] == eng]

    if args.test:
        trades = trades.head(1)

    print("Loading OHLC + EMA...")
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]
    extra     = max(0, all_dates.index(dates_5yr[0]) - 40)

    daily_ohlc = {}
    for d in all_dates[extra:]:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None or tks.empty: continue
        daily_ohlc[d] = (
            float(tks['price'].max()),
            float(tks['price'].min()),
            float(tks[tks['time'] <= '15:30:00']['price'].iloc[-1]),
            float(tks[tks['time'] >= '09:15:00']['price'].iloc[0]),
        )

    close_s = pd.Series({d:v[2] for d,v in daily_ohlc.items()}).sort_index()
    ema_s   = close_s.ewm(span=20, adjust=False).mean().shift(1)

    print(f"Generating 3 charts x {len(trades)} trades...")
    ok = 0; skip = 0

    for _, row in trades.iterrows():
        try:
            if process_trade(row, all_dates, daily_ohlc, ema_s):
                ok += 1
                tamil = ZONE_TAMIL.get(row['zone'], row['zone'])
                print(f"  [{ok}] {tamil}/{row['date'].strftime('%Y/%Y%m%d')} "
                      f"{row['bias']} {row['opt']} Rs{row['pnl']:+,.0f}")
            else:
                skip += 1
        except Exception as e:
            print(f"  ERR {row['date'].strftime('%Y-%m-%d')}: {e}")
            skip += 1

    print(f"\nDone: {ok} trades ({ok*3} PNGs), {skip} skipped")


if __name__ == '__main__':
    main()
