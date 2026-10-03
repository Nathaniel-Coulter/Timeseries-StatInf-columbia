#hw1_problem6.py
"""
Disclaimer:
I had chatgpt go in and annotate / make minimal organizational changes to the slop that I created so you wouldn't see all the errors and my janky paths despite it still running. 
He also did some fancy shit with the .xlsx because my original conversion to csv was archaic apparently idek. 

STAT 4263/5263 — Homework 1
Problem 6: Australian retail time-series analysis

This script:
  1. Reads retail.xlsx from the same directory as this script.
  2. Uses the FIRST data series only:
       Column A = monthly date
       Column B = first numeric series
     (The workbook header and Series ID are preserved in the summary.)
  3. Verifies the monthly sequence and missing values.
  4. Produces:
       hw1_problem6_outputs/
       ├── p6_data_clean.csv
       ├── p6_summary.txt
       ├── p6a_time_series.png
       ├── p6b_seasonal_linear.png
       ├── p6b_seasonal_cyclical.png
       ├── p6c_acf.png
       └── p6_acf_values.csv

Important:
- The original Excel file is NEVER modified.
- The sample ACF is computed directly from the formula stated in the homework:
      gamma_hat(h) = (1/n) sum_{t=1}^{n-h} (x_{t+h}-xbar)(x_t-xbar)
      rho_hat(h)   = gamma_hat(h) / gamma_hat(0)
- Problem 6(d) and 6(e) are interpretation questions; this script supplies
  the empirical facts/plots, but the written reasoning belongs in the solution.

AI disclosure:
This script was developed with AI assistance. If submitted, disclose that use
in accordance with the course instructions.

Dependencies:
    numpy
    matplotlib

No pandas/openpyxl is required. The .xlsx is read directly using Python's
standard ZIP/XML libraries.

Run from the project root:
    python .\columbia\stat_inf_class\hw1_problem6.py
"""

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
import csv
import math
import re
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy\n"
        "Install it in the active environment with: pip install numpy"
    ) from exc

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib\n"
        "Install it in the active environment with: pip install matplotlib"
    ) from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
SOURCE_XLSX = BASE_DIR / "retail.xlsx"
OUT_DIR = BASE_DIR / "hw1_problem6_outputs"

MAX_ACF_LAG = 60  # five years of monthly lags; includes seasonal lags 12,24,...
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

summary_lines = []


def log(line=""):
    print(line)
    summary_lines.append(str(line))


def heading(title):
    bar = "=" * 78
    log()
    log(bar)
    log(title)
    log(bar)


# =============================================================================
# MINIMAL XLSX READER
# =============================================================================
#
# We only need columns A and B from the first worksheet. XLSX files are ZIP
# archives containing XML, so this avoids any dependency on pandas/openpyxl.
# =============================================================================

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

NS = {"m": MAIN_NS, "r": REL_NS}
REL_MAP_NS = {"p": PKG_REL_NS}


def _shared_strings(zf):
    """Return Excel shared-string table, if present."""
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    result = []
    for si in root.findall("m:si", NS):
        text_parts = [node.text or "" for node in si.findall(".//m:t", NS)]
        result.append("".join(text_parts))
    return result


def _first_sheet_path(zf):
    """Resolve the first worksheet path from workbook.xml relationships."""
    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    first_sheet = workbook_root.find("m:sheets/m:sheet", NS)
    if first_sheet is None:
        raise ValueError("Workbook contains no worksheets.")

    rel_id = first_sheet.attrib.get(f"{{{REL_NS}}}id")
    if not rel_id:
        raise ValueError("Could not resolve the first worksheet relationship.")

    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    target = None
    for rel in rel_root.findall("p:Relationship", REL_MAP_NS):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target")
            break

    if not target:
        raise ValueError("Could not resolve the first worksheet target.")

    # Typical target is "worksheets/sheet1.xml".
    target = target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return f"xl/{target}"


def _workbook_uses_1904_dates(zf):
    """Read Excel's workbook date system."""
    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    workbook_pr = workbook_root.find("m:workbookPr", NS)
    if workbook_pr is None:
        return False
    return workbook_pr.attrib.get("date1904", "0") in {"1", "true", "True"}


def _cell_value(cell, shared):
    """Extract a cell's raw value."""
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        node = cell.find(".//m:t", NS)
        return node.text if node is not None else ""

    value_node = cell.find("m:v", NS)
    if value_node is None:
        return None

    raw = value_node.text

    if cell_type == "s":
        return shared[int(raw)]

    if cell_type == "b":
        return raw == "1"

    return raw


def _column_from_ref(cell_ref):
    match = re.match(r"([A-Z]+)", cell_ref or "")
    return match.group(1) if match else None


def _excel_serial_to_datetime(serial, date1904=False):
    """
    Convert an Excel serial date to datetime.

    Excel's common 1900 system is reproduced with origin 1899-12-30,
    which handles Excel's historical leap-year quirk.
    """
    serial = float(serial)
    base = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
    return base + timedelta(days=serial)


def _parse_date(raw, date1904=False):
    """Accept Excel serial dates or common month/date strings."""
    if raw is None:
        raise ValueError("Missing date")

    # Numeric Excel date serial.
    try:
        return _excel_serial_to_datetime(float(raw), date1904=date1904)
    except (ValueError, TypeError):
        pass

    text = str(raw).strip()
    formats = (
        "%b-%Y",
        "%B-%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unrecognized date value: {raw!r}")


def read_first_retail_series(path):
    """
    Read:
      B1 = series description
      B2 = series ID
      rows 3+:
        A = monthly date
        B = first data series

    Returns:
      series_header, series_id, list[(datetime, float)], skipped_rows
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find:\n  {path}\n\n"
            "Place retail.xlsx in the same directory as this script."
        )

    with ZipFile(path) as zf:
        shared = _shared_strings(zf)
        sheet_path = _first_sheet_path(zf)
        date1904 = _workbook_uses_1904_dates(zf)

        root = ET.fromstring(zf.read(sheet_path))
        rows = root.findall(".//m:sheetData/m:row", NS)

        if len(rows) < 3:
            raise ValueError("The worksheet does not contain the expected header/data rows.")

        def row_map(row):
            out = {}
            for cell in row.findall("m:c", NS):
                col = _column_from_ref(cell.attrib.get("r"))
                if col:
                    out[col] = _cell_value(cell, shared)
            return out

        header_row = row_map(rows[0])
        id_row = row_map(rows[1])

        series_header = header_row.get("B", "")
        series_id = id_row.get("B", "")

        data = []
        skipped = []

        for row in rows[2:]:
            values = row_map(row)
            raw_date = values.get("A")
            raw_value = values.get("B")

            # Ignore fully blank rows.
            if raw_date is None and raw_value is None:
                continue

            try:
                date = _parse_date(raw_date, date1904=date1904)
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError("non-finite numeric value")
            except Exception as exc:
                skipped.append(
                    (row.attrib.get("r", "?"), raw_date, raw_value, str(exc))
                )
                continue

            data.append((date, value))

    if not data:
        raise ValueError("No valid observations were found in columns A/B.")

    data.sort(key=lambda item: item[0])
    return series_header, series_id, data, skipped


# =============================================================================
# DATA VALIDATION / HELPERS
# =============================================================================

def month_index(dt):
    return dt.year * 12 + (dt.month - 1)


def missing_months_between(dates):
    """Return missing YYYY-MM months inside the observed first-to-last interval."""
    present = {month_index(d) for d in dates}
    first = min(present)
    last = max(present)

    missing = []
    for idx in range(first, last + 1):
        if idx not in present:
            year, month0 = divmod(idx, 12)
            missing.append(f"{year:04d}-{month0 + 1:02d}")
    return missing


def sample_acf_exact(values, max_lag):
    """
    Homework definition:
      gamma_hat(h) = (1/n) sum_{t=1}^{n-h}
                     (x_{t+h} - xbar)(x_t - xbar)
      rho_hat(h) = gamma_hat(h) / gamma_hat(0)
    """
    x = np.asarray(values, dtype=float)
    n = len(x)

    if n < 2:
        raise ValueError("At least two observations are required.")

    max_lag = min(max_lag, n - 1)
    xbar = x.mean()
    centered = x - xbar

    gamma = np.empty(max_lag + 1, dtype=float)

    for h in range(max_lag + 1):
        gamma[h] = np.dot(centered[h:], centered[: n - h]) / n

    if gamma[0] == 0:
        raise ValueError("Sample variance is zero; ACF is undefined.")

    rho = gamma / gamma[0]
    return gamma, rho


# =============================================================================
# LOAD DATA
# =============================================================================

heading("PROBLEM 6 — DATA LOAD AND VALIDATION")

series_header, series_id, data, skipped_rows = read_first_retail_series(SOURCE_XLSX)

dates = [d for d, _ in data]
values = np.asarray([v for _, v in data], dtype=float)
n = len(values)

missing_months = missing_months_between(dates)

log(f"Source file: {SOURCE_XLSX}")
log(f"Workbook first data-series header: {series_header}")
log(f"Series ID: {series_id}")
log(f"Observations: {n}")
log(f"Start: {dates[0].strftime('%b-%Y')}")
log(f"End: {dates[-1].strftime('%b-%Y')}")
log(f"Rows skipped during parsing: {len(skipped_rows)}")
log(f"Missing months inside observed span: {len(missing_months)}")
if missing_months:
    log("Missing month(s): " + ", ".join(missing_months))

# Save cleaned series.
clean_csv = OUT_DIR / "p6_data_clean.csv"
with clean_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "year", "month", "month_name", "turnover"])
    for dt, value in data:
        writer.writerow(
            [dt.strftime("%Y-%m-%d"), dt.year, dt.month, MONTH_NAMES[dt.month - 1], value]
        )

log(f"Saved cleaned data: {clean_csv.name}")


# =============================================================================
# DESCRIPTIVE FACTS
# =============================================================================

heading("DESCRIPTIVE FACTS")

overall_mean = float(values.mean())
overall_std = float(values.std(ddof=0))
overall_min = float(values.min())
overall_max = float(values.max())

time_index = np.arange(n, dtype=float)
trend_slope_month, trend_intercept = np.polyfit(time_index, values, 1)
trend_slope_year = float(trend_slope_month * 12.0)

first12_mean = float(values[: min(12, n)].mean())
last12_mean = float(values[max(0, n - 12):].mean())

log(f"Overall mean: {overall_mean:.4f}")
log(f"Population-style sample SD (ddof=0): {overall_std:.4f}")
log(f"Minimum observed value: {overall_min:.4f}")
log(f"Maximum observed value: {overall_max:.4f}")
log(f"Simple linear trend slope: {trend_slope_month:.4f} units/month")
log(f"Simple linear trend slope: {trend_slope_year:.4f} units/year")
log(f"Mean of first 12 observations: {first12_mean:.4f}")
log(f"Mean of last 12 observations: {last12_mean:.4f}")


# =============================================================================
# PROBLEM 6(a): TIME-SERIES PLOT
# =============================================================================

heading("Problem 6(a) — Sequential time plot")

fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.plot(dates, values, linewidth=1.2)
ax.set_title("Problem 6(a): New South Wales — first retail data series")
ax.set_xlabel("Time")
ax.set_ylabel("Turnover (source units)")
ax.grid(alpha=0.25)

# Keep the 40-year x-axis readable.
ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()

fig.tight_layout()
fig.savefig(OUT_DIR / "p6a_time_series.png", dpi=180)
plt.close(fig)

log("Saved p6a_time_series.png")


# =============================================================================
# PROBLEM 6(b): SEASONAL PLOTS + MONTHLY AVERAGES
# =============================================================================

heading("Problem 6(b) — Seasonal plots and calendar-month averages")

# Organize values by year.
year_to_months = {}
for dt, value in data:
    year_to_months.setdefault(dt.year, {})[dt.month] = value

# Raw calendar-month averages.
monthly_values = {m: [] for m in range(1, 13)}
for dt, value in data:
    monthly_values[dt.month].append(value)

monthly_avg = {
    m: float(np.mean(monthly_values[m])) if monthly_values[m] else float("nan")
    for m in range(1, 13)
}
monthly_counts = {m: len(monthly_values[m]) for m in range(1, 13)}

for m in range(1, 13):
    log(
        f"{MONTH_NAMES[m-1]}: mean={monthly_avg[m]:.4f} "
        f"(n={monthly_counts[m]})"
    )

highest_month = max(monthly_avg, key=lambda m: monthly_avg[m])
log(
    f"Highest-average month: {MONTH_NAMES[highest_month - 1]} "
    f"({monthly_avg[highest_month]:.4f})"
)
log("Note: these are raw calendar-month averages; the series is not detrended first.")

# ---- Linear seasonal plot ----
fig, ax = plt.subplots(figsize=(10.0, 5.4))

for year in sorted(year_to_months):
    months_dict = year_to_months[year]
    months = sorted(months_dict)
    vals = [months_dict[m] for m in months]
    ax.plot(months, vals, linewidth=0.8, alpha=0.28)

mean_cycle = [monthly_avg[m] for m in range(1, 13)]
ax.plot(
    range(1, 13),
    mean_cycle,
    linewidth=3.0,
    marker="o",
    label="Calendar-month mean",
)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_NAMES)
ax.set_title("Problem 6(b): Seasonal plot — linear form")
ax.set_xlabel("Calendar month")
ax.set_ylabel("Turnover (source units)")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "p6b_seasonal_linear.png", dpi=180)
plt.close(fig)

log("Saved p6b_seasonal_linear.png")

# ---- Cyclical / polar seasonal plot ----
fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111, projection="polar")

theta_months = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)

for year in sorted(year_to_months):
    months_dict = year_to_months[year]
    months = sorted(months_dict)
    theta = [theta_months[m - 1] for m in months]
    vals = [months_dict[m] for m in months]

    # Close only complete Jan-Dec years.
    if months == list(range(1, 13)):
        theta = theta + [theta[0]]
        vals = vals + [vals[0]]

    ax.plot(theta, vals, linewidth=0.8, alpha=0.22)

mean_theta = list(theta_months) + [theta_months[0]]
mean_vals = mean_cycle + [mean_cycle[0]]
ax.plot(
    mean_theta,
    mean_vals,
    linewidth=3.0,
    marker="o",
    label="Calendar-month mean",
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticks(theta_months)
ax.set_xticklabels(MONTH_NAMES)
ax.set_title("Problem 6(b): Seasonal plot — cyclical form", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

fig.tight_layout()
fig.savefig(OUT_DIR / "p6b_seasonal_cyclical.png", dpi=180, bbox_inches="tight")
plt.close(fig)

log("Saved p6b_seasonal_cyclical.png")


# =============================================================================
# PROBLEM 6(c): SAMPLE ACF
# =============================================================================

heading("Problem 6(c) — Sample autocorrelation function")

gamma_hat, rho_hat = sample_acf_exact(values, MAX_ACF_LAG)
actual_max_lag = len(rho_hat) - 1

acf_csv = OUT_DIR / "p6_acf_values.csv"
with acf_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["lag_months", "sample_autocovariance", "sample_acf"])
    for h in range(actual_max_lag + 1):
        writer.writerow([h, gamma_hat[h], rho_hat[h]])

# Print the lags most useful for writing the solution:
# every lag 0-24, then yearly lags through max.
selected_lags = list(range(0, min(24, actual_max_lag) + 1))
for h in (36, 48, 60):
    if h <= actual_max_lag and h not in selected_lags:
        selected_lags.append(h)

for h in selected_lags:
    log(
        f"lag {h:>2}: "
        f"gamma_hat={gamma_hat[h]:>12.4f}, "
        f"rho_hat={rho_hat[h]:>8.4f}"
    )

# ACF plot
lags = np.arange(actual_max_lag + 1)

fig, ax = plt.subplots(figsize=(10.0, 5.2))
ax.axhline(0.0, linewidth=0.8)
ax.vlines(lags, 0.0, rho_hat, linewidth=1.0)
ax.plot(lags, rho_hat, "o", markersize=3.5)

ax.set_title(
    f"Problem 6(c): Sample ACF of first NSW retail series (lags 0–{actual_max_lag})"
)
ax.set_xlabel("Lag h (months)")
ax.set_ylabel(r"Sample ACF $\hat{\rho}(h)$")
ax.set_xticks(np.arange(0, actual_max_lag + 1, 6))
ax.set_ylim(
    min(-0.1, float(rho_hat.min()) - 0.05),
    min(1.05, max(1.0, float(rho_hat.max()) + 0.05)),
)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT_DIR / "p6c_acf.png", dpi=180)
plt.close(fig)

log(f"Saved p6c_acf.png (lags 0 through {actual_max_lag})")
log(f"Saved {acf_csv.name}")


# =============================================================================
# WRITE SUMMARY
# =============================================================================

heading("DONE")
log(f"All outputs saved to: {OUT_DIR}")
log("The original retail.xlsx file was not modified.")
log("Use the figures/numerical summaries for 6(a)-6(c).")
log("Write 6(d)-6(e) as statistical reasoning based on the observed plots/data.")

summary_path = OUT_DIR / "p6_summary.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print()
print(f"Summary written to: {summary_path}")
