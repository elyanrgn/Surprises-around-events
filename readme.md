# Surprises Around Events - Beginner Guide

This project helps you measure how financial instruments move around specific events (for example inflation releases, central bank announcements, etc.).

You give it:
- minute-by-minute market files (`.csv.gz`)
- a list of event dates/times (`events.csv`)

It gives you:
- one output table with the variation for each `(instrument, event)`
- one filter log table to understand removed/invalid rows

No Python coding is needed for normal use. You mainly edit `config.yaml` and your `events.csv`.

## 1. What the pipeline does

For each instrument, the code:
1. Reads and standardizes bid/ask columns from source files.
2. Applies quality filters (missing quotes, invalid spreads, outliers).
3. Converts timestamps to the event timezone.
4. Resamples to a 1-minute grid and forward-fills gaps.
5. Computes the variation around each event window.

### Output unit
- Rate instruments: variation in `bp`.
- Price instruments (FX/equity-like): variation in `%`.

## 2. Project structure

- `main.py`: entry point you run.
- `config.yaml`: your main configuration file.
- `src/`: pipeline modules.
- `output/`: generated results and logs (created automatically).

## 3. Prerequisites

- Python 3.9+
- Basic terminal usage

Install dependencies in your environment:

```bash
pip install pandas numpy pyyaml
```

Optional (recommended for faster/smaller cache files):

```bash
pip install pyarrow
```

## 4. Prepare your data

### 4.1 Market files

Put all your market `.csv.gz` files under the folder defined by `data_root` in `config.yaml`.

The code scans folders recursively, so subfolders are fine.

Expected filename pattern:
- With maturity: `{NAME}_{MATURITY}_{YYYY-YYYY}.csv.gz`
- Without maturity: `{NAME}_{YYYY-YYYY}.csv.gz`

Examples:
- `AT_10Y_1999-2009.csv.gz`
- `EURUSD_2010-2019.csv.gz`
- `EUROSTOXX_BANKS_2010-2019.csv.gz`
- `USD_ILS_3M_2010-2019.csv.gz`
- `EONIA_1M_2010-2019.csv.gz`
- `ESTR_1M_2020-2026.csv.gz`

Notes:
- If both `EONIA_<mat>` and `ESTR_<mat>` exist, the code builds one spliced `OIS_<mat>` series automatically.
- Files with invalid names are skipped with a warning.

### 4.2 Events file (`events.csv`)

Create a CSV file (path defined by `events_csv` in config) with at least:
- `datetime` (required)
- `label` (required)

Optional columns:
- `tz` (timezone, e.g. `Europe/Brussels`)
- `window_minutes` (integer)

If optional columns are missing, defaults from `config.yaml` are used.

Example:

```csv
datetime,label,tz,window_minutes
2025-03-03 11:00:00,HICP_flash,Europe/Brussels,60
2025-03-06 14:15:00,ECB_rate_decision,Europe/Brussels,90
```

## 5. Configure `config.yaml`

Most users only need these fields:
- `data_root`: where your `.csv.gz` files are
- `events_csv`: your events file path
- `output`: output CSV path
- `default_tz`, `default_window_minutes`: defaults for events

You can also tune filtering and gap fill parameters in the same file.

## 6. Run the pipeline

From the repository root:

```bash
python main.py
```

Use a specific config file:

```bash
python main.py --config my_config.yaml
```

Generate an example config file:

```bash
python main.py --init-config
```

## 7. Output files

After a successful run:
- Main results CSV at path `output` from config.
- Filter log CSV next to it, with suffix `_filter_log.csv`.

Typical result columns:
- `instrument_label`
- `event_label`
- `event_datetime`
- `asset_name`
- `maturity`
- `mid_before`
- `mid_after`
- `variation`
- `unit`
- `before_observed`
- `after_observed`

Interpretation:
- Positive `variation`: instrument moved up over the event window.
- Negative `variation`: instrument moved down.
- Check `unit` (`bp` or `pct`) before comparing values.

## 8. Troubleshooting (common)

### "Aucun instrument découvert sous ..."
- Your `data_root` path is wrong, or no `.csv.gz` files were found.

### "colonnes obligatoires manquantes" for events
- `events.csv` must contain at least `datetime` and `label`.

### Files are ignored due to naming
- Rename files to follow expected pattern.

### Timezone confusion
- If event datetimes are written without timezone, the pipeline applies `tz` (or `default_tz`).

## 9. Quick start checklist

1. Put your market `.csv.gz` files under `data_root`.
2. Create `events.csv` with `datetime,label`.
3. Update `config.yaml` paths.
4. Run `python main.py`.
5. Open output CSV and `_filter_log.csv`.
