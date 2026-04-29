import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart, plot_equity

DATA_DIR = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell/data'
CSV_PATH = os.path.join(DATA_DIR, '56_combined_trades.csv')

ZONE_MARVEL = {
    'above_r4':    'Thor',
    'r3_to_r4':    'Iron Man',
    'r2_to_r3':    'Captain Marvel',
    'r1_to_r2':    'Spider-Man',
    'pdh_to_r1':   'Black Panther',
    'tc_to_pdh':   'Hawkeye',
    'within_cpr':  'Vision',
    'pdl_to_bc':   'Ant-Man',
    'pdl_to_s1':   'Black Widow',
    's1_to_s2':    'Hulk',
    's2_to_s3':    'Winter Soldier',
    's3_to_s4':    'Loki',
    'below_s4':    'Thanos',
}

df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df = df[df['strategy'] == 'v17a'].copy()
df['marvel'] = df['zone'].map(ZONE_MARVEL).fillna(df['zone'])

# -------------------------------------------------------
# Zone order (top to bottom: above_r4 → below_s4)
# -------------------------------------------------------
ZONE_ORDER = [
    'above_r4', 'r3_to_r4', 'r2_to_r3', 'r1_to_r2',
    'pdh_to_r1', 'tc_to_pdh', 'within_cpr', 'pdl_to_bc',
    'pdl_to_s1', 's1_to_s2', 's2_to_s3', 's3_to_s4', 'below_s4',
]
MARVEL_ORDER = [ZONE_MARVEL[z] for z in ZONE_ORDER if z in ZONE_MARVEL]

# -------------------------------------------------------
# Per-zone stats
# -------------------------------------------------------
stats = []
for zone in ZONE_ORDER:
    zdf = df[df['zone'] == zone]
    if len(zdf) == 0:
        continue
    n      = len(zdf)
    wins   = (zdf['pnl'] > 0).sum()
    wr     = round(wins / n * 100, 1)
    total  = round(zdf['pnl'].sum(), 2)
    avg    = round(zdf['pnl'].mean(), 2)
    marvel = ZONE_MARVEL.get(zone, zone)
    stats.append({'zone': zone, 'marvel': marvel, 'n': n, 'wr': wr,
                  'total_pnl': total, 'avg_pnl': avg})

stats_df = pd.DataFrame(stats)
print("\n" + "=" * 70)
print("  ZONE-WISE PERFORMANCE — Marvel Edition (v17a, 356 trades)")
print("=" * 70)
print(f"{'Zone':<25} {'N':>5} {'WR%':>7} {'Avg PNL':>10} {'Total PNL':>13}")
print("-" * 70)
for _, r in stats_df.iterrows():
    print(f"{r['marvel']:<25} {r['n']:>5} {r['wr']:>6.1f}% {r['avg_pnl']:>10,.0f} {r['total_pnl']:>13,.0f}")
totals = df['pnl']
total_n   = len(df)
total_wr  = round((df['pnl'] > 0).sum() / total_n * 100, 1)
total_pnl = round(totals.sum(), 2)
print("-" * 70)
print(f"{'TOTAL':<25} {total_n:>5} {total_wr:>6.1f}% {'':>10} {total_pnl:>13,.0f}")

# -------------------------------------------------------
# Year-wise breakdown
# -------------------------------------------------------
print("\n" + "=" * 55)
print("  YEAR-WISE PERFORMANCE (v17a)")
print("=" * 55)
yrdf = df.groupby('year').agg(
    n=('pnl', 'count'),
    wins=('pnl', lambda x: (x > 0).sum()),
    total_pnl=('pnl', 'sum'),
).reset_index()
yrdf['wr'] = (yrdf['wins'] / yrdf['n'] * 100).round(1)
yrdf['total_pnl'] = yrdf['total_pnl'].round(2)
for _, r in yrdf.iterrows():
    print(f"  {int(r['year'])}: {r['n']:>3} trades | WR {r['wr']}% | P&L Rs {r['total_pnl']:>10,.0f}")

# -------------------------------------------------------
# Chart 1: WR% per zone (bar chart)
# -------------------------------------------------------
ts_base = int(pd.Timestamp('2021-01-01').timestamp())
bar_wr = []
for i, row in stats_df.iterrows():
    color = '#26a69a' if row['wr'] >= 70 else ('#f59e0b' if row['wr'] >= 60 else '#ef5350')
    bar_wr.append({
        'time': ts_base + i * 86400 * 90,
        'value': round(row['wr'], 2),
        'color': color,
        'label': row['marvel'],
    })

tv_wr = {
    'lines': [{
        'id': 'zone_wr',
        'label': 'Win Rate %',
        'seriesType': 'bar',
        'data': bar_wr,
        'xLabels': [r['label'] for r in bar_wr],
    }],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('zone_wr', tv_wr, title='Zone Win Rate % — Marvel Edition (v17a)')
print('\n📊 WR chart sent')

# -------------------------------------------------------
# Chart 2: Total P&L per zone (bar chart)
# -------------------------------------------------------
bar_pnl = []
for i, row in stats_df.iterrows():
    pnl_val = row['total_pnl']
    color = '#26a69a' if pnl_val >= 0 else '#ef5350'
    bar_pnl.append({
        'time': ts_base + i * 86400 * 90,
        'value': round(pnl_val, 2),
        'color': color,
        'label': row['marvel'],
    })

tv_pnl = {
    'lines': [{
        'id': 'zone_pnl',
        'label': 'Total P&L (Rs)',
        'seriesType': 'bar',
        'data': bar_pnl,
        'xLabels': [r['label'] for r in bar_pnl],
    }],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('zone_pnl', tv_pnl, title='Zone Total P&L — Marvel Edition (v17a)')
print('📊 P&L chart sent')

# -------------------------------------------------------
# Chart 3: Trade count per zone
# -------------------------------------------------------
bar_count = []
for i, row in stats_df.iterrows():
    bar_count.append({
        'time': ts_base + i * 86400 * 90,
        'value': float(row['n']),
        'color': '#4BC0C0',
        'label': row['marvel'],
    })

tv_count = {
    'lines': [{
        'id': 'zone_count',
        'label': 'Trade Count',
        'seriesType': 'bar',
        'data': bar_count,
        'xLabels': [r['label'] for r in bar_count],
    }],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('zone_count', tv_count, title='Zone Trade Count — Marvel Edition (v17a)')
print('📊 Count chart sent')

# -------------------------------------------------------
# Chart 4: Equity curve (v17a sorted by date)
# -------------------------------------------------------
v17a_sorted = df.sort_values('date').reset_index(drop=True)
equity = v17a_sorted['pnl'].cumsum()
running_max = equity.cummax()
drawdown = equity - running_max

equity_data = [
    {'time': int(pd.Timestamp(row['date']).timestamp()), 'value': round(eq, 2)}
    for (_, row), eq in zip(v17a_sorted.iterrows(), equity)
]
dd_data = [
    {'time': int(pd.Timestamp(row['date']).timestamp()), 'value': round(dd, 2)}
    for (_, row), dd in zip(v17a_sorted.iterrows(), drawdown)
]

tv_eq = {
    'lines': [
        {'id': 'equity', 'label': 'Equity', 'data': equity_data,
         'seriesType': 'baseline', 'baseValue': 0},
        {'id': 'drawdown', 'label': 'Drawdown', 'data': dd_data,
         'seriesType': 'baseline', 'baseValue': 0, 'isNewPane': True},
    ],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('v17a_equity', tv_eq,
                  title=f'v17a Equity Curve — {total_n} trades | WR {total_wr}% | P&L Rs {total_pnl:,.0f}')
print('📊 Equity chart sent')

# -------------------------------------------------------
# Chart 5: Zone-wise P&L heatmap (zone × year)
# -------------------------------------------------------
years_list = sorted(df['year'].dropna().unique().astype(int).tolist())
heatmap_z = []
heatmap_text = []
for zone in ZONE_ORDER:
    row_z = []
    row_t = []
    for yr in years_list:
        sub = df[(df['zone'] == zone) & (df['year'] == yr)]
        if len(sub) == 0:
            row_z.append(None)
            row_t.append('')
        else:
            pval = round(sub['pnl'].sum(), 0)
            row_z.append(pval)
            row_t.append(f'{pval:+,.0f}')
    heatmap_z.append(row_z)
    heatmap_text.append(row_t)

tv_hm = {
    'lines': [{
        'id': 'zone_year_heatmap',
        'label': 'Zone × Year P&L',
        'seriesType': 'heatmap',
        'data': [{'time': ts_base, 'value': 0}],
        'z': heatmap_z,
        'xLabels': [str(y) for y in years_list],
        'yLabels': [ZONE_MARVEL.get(z, z) for z in ZONE_ORDER],
        'colorscale': [[0, '#ef5350'], [0.5, '#1e293b'], [1, '#26a69a']],
        'zmid': 0,
        'text': heatmap_text,
        'textTemplate': '%{text}',
        'xTitle': 'Year',
        'yTitle': 'Zone',
        'colorbarTitle': 'P&L (Rs)',
        'yReversed': False,
    }],
    'candlestick': [], 'volume': [], 'isTvFormat': False,
}
send_custom_chart('zone_year_heatmap', tv_hm,
                  title='Zone × Year P&L Heatmap — Marvel Edition (v17a)')
print('📊 Heatmap chart sent')

print('\nAll charts pushed!')
