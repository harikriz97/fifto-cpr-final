# CLAUDE.md - Project Guidelines
# VERSION: 32
## Rules

**Rule 0**: Keep it simple - choose the simplest solution that works

**Rule 1**: Always check `my_util.py` first for existing functions. For chart/plotting, use the `sa-kron-chart` skill — it handles the `plot_util` import. Do NOT create `plot_util.py` locally or import it directly.

**Rule 2**: The data folder path is defined by the environment variable `INTER_SERVER_DATA_PATH` (e.g., `/mnt/data/day-wise`). This folder is READ ONLY at all times.

**Rule 3**: The filename and folder name must follow naming convention of XX_filename where XX is the serial number. Whenever creating a csv or png file put it in the data/YYYYMMDD folder where YYYYMMDD is the trading date.

**Rule 4**: When working with notebook files, always read it, delete it, and recreate it with the same name

**Rule 5**: Use Plotly for all visualizations. Save using `super_plotter()` with formats: SVG and/or JSON. Read `kron-chart.md` for chart plotting patterns and templates.

**Rule 6**: All DataFrame column names, python variable names and CSV headers must strictly follow snake_case convention (e.g., `date_time`, `spot_open`, `ce_close`, `entry_price`, `stop_loss`)

**Rule 7**: All computed values must be rounded to 2 decimal points using `round(value, 2)`

**Rule 8**: Forward Bias Prevention - Always use `entry_signal_time` variable (next candle open + 2 seconds, NOT current candle time). Example: Signal at 09:24:00 → `entry_signal_time = 09:25:02`

**Rule 9**: Sequential Trading - Trade one position at a time. Only look for new signals AFTER current position exits. No overlapping trades allowed.

**Rule 10**: Intraday Trading Window Default - If no trading window specified, use entry_signal_time between 09:16:00 and 15:15:00 with EOD exit at 15:20:00

**Rule 11**: Instrument Naming Format - NIFTY options follow the pattern `NIFTY{expiry}{strike}{option_type}` where expiry is YYYYMMDD format, strike is the price level, and option_type is CE or PE. Example: `NIFTY2018102510350CE`. Use `fetch_option_chain()` to get ATM strike and expiry. Tick data files are stored as `${INTER_SERVER_DATA_PATH}/{date}/{instrument_name}.csv`. Use `load_tick_data()` to load tick data.

**Rule 12**: Target and Stoploss Validation - Always check every target and stoploss at the tick level only. Use tick-by-tick data to validate when price levels are hit, not candle-level data.

**Rule 13**: Chart Push is Mandatory - Every chart MUST be sent to the chat via `super_plotter(..., file_formats=['json'])`, `plot_equity()`, or `send_custom_chart()`. These functions are provided by the `sa-kron-chart` skill — do NOT import or create `plot_util.py` locally. Never call `pio.write_image()`, `fig.write_image()`, or `fig.show()` directly. Never save a chart file without also calling the push callback. `MESSAGE_CALLBACK_URL` must be set in the environment; if it is not, the code will raise and the chart must not be silently skipped.

**Rule 14**: Custom Chart Renderers - For chart features beyond standard candlestick/volume/lines/markers (e.g., zones, session highlights, Ichimoku cloud, custom indicators), use the `renderers` parameter. Send JavaScript renderer functions alongside the data — the chart engine executes them at runtime. No deployment or engine update needed.

```python
# Standard chart (Plotly → TradingView JSON → push)
super_plotter(folder_path, fig, "chart_name", "20250115", title="NIFTY — 1min")

# Standard chart with custom renderer
super_plotter(folder_path, fig, "chart_name", "20250115", title="NIFTY — 1min", renderers={
    "zones": '''function(chart, data, series, LWC) {
        for (var i = 0; i < data.length; i++) {
            series.createPriceLine({price: data[i].from, color: data[i].color});
        }
    }'''
})

# Direct tvJson with custom renderer (no Plotly figure)
send_custom_chart("chart_id", tv_json, renderers={...}, title="NIFTY — Custom")

# Equity curve
plot_equity(equity_series, drawdown_series, "equity", title="Equity Curve")
```

**Rule 15**: No Chart Config Files - Do not create chart-preferences.json or any separate chart configuration files. All chart styling, renderers, and preferences travel with the data in each chart call. The agent decides styling from conversation context and these rules.

---

## Data Structure at INTER_SERVER_DATA_PATH

### Folder Hierarchy
```
/mnt/data/day-wise/
├── YYYYMMDD/                        # Trading date folder (e.g., 20250115)
│   ├── NIFTY.csv                    # NIFTY spot tick data
│   ├── BANKNIFTY.csv                # BANKNIFTY spot tick data
│   ├── FINNIFTY.csv                 # FINNIFTY spot tick data
│   ├── SENSEX.csv                   # SENSEX spot tick data
│   ├── RELIANCE.csv                 # Stock tick data
│   ├── NIFTY25011621350CE.csv       # Option: NIFTY, expiry 250116, strike 21350, CE
│   ├── NIFTY25011621350PE.csv       # Option: NIFTY, expiry 250116, strike 21350, PE
│   ├── BANKNIFTY2501164900CE.csv    # Option: BANKNIFTY, expiry 250116, strike 4900, CE
│   └── ...
```

### CSV Schema (No Header Row)
All tick data files have the same 5-column structure:

| Column | Name           | Example       | Description                |
|--------|----------------|---------------|----------------------------|
| 1      | date           | 20250115      | Trading date (YYYYMMDD)    |
| 2      | time           | 09:15:02      | Tick time (HH:MM:SS)       |
| 3      | price          | 23253.25      | Last traded price (LTP)    |
| 4      | volume         | 0             | Traded volume              |
| 5      | open_interest  | 1800          | Open interest              |

### Option File Naming Convention
`{INDEX}{YYMMDD}{STRIKE}{CE|PE}.csv`
- INDEX: NIFTY, BANKNIFTY, FINNIFTY, SENSEX
- YYMMDD: Expiry date (short format)
- STRIKE: Strike price
- CE/PE: Call or Put

Example: `NIFTY25011621350CE.csv` = NIFTY Call, Expiry 2025-01-16, Strike 21350

### Data Range
- Available dates: 20181010 to present
- Market hours: 09:15:00 to 15:30:00
