# sheet2_q5_script.py

"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\hocke\Desktop\quant_portfolio_scaffold\columbia\stat_inf_class")
DEFAULT_INPUT = BASE_DIR / "simu.txt"
DEFAULT_OUTPUT = BASE_DIR / "sheet2_q5_outputs"

MA_HALF_WIDTH = 5          # k=5: centered 11-point moving average.
GAUSSIAN_BANDWIDTH = 6.0  # b=6, in observation/time-index units.
PERIOD = 24
SPIKE_START = 6
N_BLOCKS = 5
DPI = 180

def load_data(file_path: Path) -> np.ndarray:
    """Read the one-column simu.txt, which has the column header 'x'."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, sep=r"\s+")
    if df.shape[1] != 1:
        raise ValueError(f"Expected one column; found {df.shape[1]} in {file_path}")
    x = pd.to_numeric(df.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    if x.size < 50 or not np.isfinite(x).all():
        raise ValueError("Data must contain at least 50 finite numerical observations.")
    if (x <= 0).any():
        raise ValueError("Log/square-root transformations require strictly positive data.")
    return x


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "Time (t)") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20)


def centered_weighted_smoother(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Centered smoothing; renormalize weights near the series boundaries."""
    numerator = np.convolve(y, weights, mode="same")
    denominator = np.convolve(np.ones_like(y), weights, mode="same")
    return numerator / denominator


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    weights = np.ones(2 * k + 1, dtype=float)
    return centered_weighted_smoother(y, weights)


def gaussian_kernel(y: np.ndarray, bandwidth: float) -> np.ndarray:
    # Gaussian kernel K((t-s)/b) with truncation at +/-4b.
    half_width = int(np.ceil(4 * bandwidth))
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    return centered_weighted_smoother(y, weights)


def design_matrix(t: np.ndarray, with_spikes: bool) -> tuple[np.ndarray, list[str]]:
    omega = 2.0 * np.pi * t / PERIOD
    terms = [np.ones(t.size), t, np.sin(omega), np.cos(omega)]
    names = ["a", "b", "beta1_sin", "beta2_cos"]
    if with_spikes:
        terms.append(((t - SPIKE_START) % PERIOD == 0).astype(float))
        names.append("beta3_spike")
    return np.column_stack(terms), names


def ols_fit(y: np.ndarray, t: np.ndarray, with_spikes: bool) -> dict:
    matrix, names = design_matrix(t, with_spikes)
    beta, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    fitted = matrix @ beta
    residual = y - fitted
    rss = float(np.dot(residual, residual))
    tss = float(np.dot(y - y.mean(), y - y.mean()))
    n, p = matrix.shape
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    return {
        "coefficients": dict(zip(names, beta)),
        "fitted": fitted,
        "residual": residual,
        "rmse": float(np.sqrt(rss / n)),
        "mae": float(np.mean(np.abs(residual))),
        "r_squared": float(1.0 - rss / tss),
        "aic_relative": float(n * np.log(rss / n) + 2 * p),
        "lag1_resid_corr": lag1,
        "num_coefficients": p,
    }


def block_variability(y: np.ndarray, t: np.ndarray, spike: np.ndarray) -> dict:
    """Assess variance stability using 1st differences, excluding spikes and their neighbors.

    Differences remove most slowly changing levels. Ignoring differences involving
    the known 24-period spikes prevents recurring jumps from overwhelming the test.
    Report standard deviations separately for five successive time blocks.
    """
    d = np.diff(y)
    eligible = ~(spike[1:] | spike[:-1])
    indices = np.array_split(np.arange(1, len(y)), N_BLOCKS)
    sds = []
    for idx in indices:
        values = d[idx - 1]
        valid = eligible[idx - 1]
        if valid.sum() < 2:
            raise ValueError("Not enough non-spike observations for variance comparison.")
        sds.append(float(np.std(values[valid], ddof=1)))
    return {
        "sd_by_block": sds,
        "cv_of_block_sds": float(np.std(sds, ddof=1) / np.mean(sds)),
        "max_sd_over_min_sd": float(max(sds) / min(sds)),
    }


def plot_fit(t: np.ndarray, y: np.ndarray, fitted: np.ndarray,
             title: str, destination: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, y, lw=0.95, alpha=0.75, label="Transformed data")
    ax.plot(t, fitted, lw=1.5, label="Estimated mean/trend")
    make_axis(ax, title, ylabel)
    ax.legend(loc="best")
    save_figure(fig, destination)


def plot_residual(t: np.ndarray, residual: np.ndarray,
                  title: str, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(t, residual, lw=0.9)
    ax.axhline(0, color="black", lw=0.9, ls="--")
    make_axis(ax, title, "Residual")
    save_figure(fig, destination)


def plot_regression(t: np.ndarray, y: np.ndarray, fit: dict,
                    title: str, destination: Path, ylabel: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    ax1.plot(t, y, lw=0.95, alpha=0.75, label="Transformed data")
    ax1.plot(t, fit["fitted"], lw=1.5, label="OLS fitted mean")
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.20)
    ax2.plot(t, fit["residual"], lw=0.9)
    ax2.axhline(0, color="black", lw=0.9, ls="--")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Time (t)")
    ax2.grid(alpha=0.20)
    fig.tight_layout()
    save_figure(fig, destination)


def run(data_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = load_data(data_file)
    n = len(x)
    t = np.arange(1, n + 1, dtype=int)  # Essential: 1-based indexing.
    spike = ((t - SPIKE_START) % PERIOD == 0)
    transforms = {"original": x, "sqrt": np.sqrt(x), "log": np.log(x)}

    # 5(a): Plot the data; compare transformations using time-block variability.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, x, lw=0.9)
    make_axis(ax, "5(a): Original simu.txt observations", "Original x(t)")
    save_figure(fig, out_dir / "01_original_series.png")

    diag = {name: block_variability(z, t, spike) for name, z in transforms.items()}
    # The assignment explicitly requires one of these two transformations.
    selected = min(("sqrt", "log"), key=lambda name: diag[name]["cv_of_block_sds"])
    y = transforms[selected]
    ylabel = f"{selected}(x(t))"

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, name in zip(axes, ("sqrt", "log")):
        ax.plot(t, transforms[name], lw=0.95)
        ax.set_title(f"5(a): {name} transformation" + (" (selected)" if name == selected else ""))
        ax.set_ylabel(f"{name}(x(t))")
        ax.grid(alpha=0.20)
    axes[-1].set_xlabel("Time (t)")
    fig.tight_layout()
    save_figure(fig, out_dir / "02_transformation_comparison.png")

    diagnostics_rows = []
    for name, d in diag.items():
        row = {"transformation": name, "cv_of_block_sds": d["cv_of_block_sds"],
               "max_sd_over_min_sd": d["max_sd_over_min_sd"]}
        row.update({f"block_{i}_difference_sd": val for i, val in enumerate(d["sd_by_block"], 1)})
        diagnostics_rows.append(row)
    pd.DataFrame(diagnostics_rows).to_csv(out_dir / "transformation_diagnostics.csv", index=False)

    # 5(b): Centered MA smoother, k=5, and Gaussian kernel smoother, b=6.
    ma_fit = moving_average(y, MA_HALF_WIDTH)
    kernel_fit = gaussian_kernel(y, GAUSSIAN_BANDWIDTH)
    ma_resid = y - ma_fit
    kernel_resid = y - kernel_fit
    plot_fit(t, y, ma_fit, "5(b): Centered moving-average fit (k=5, 11 observations)",
             out_dir / "03_moving_average_fit.png", ylabel)
    plot_residual(t, ma_resid, "5(b): Moving-average residuals",
                  out_dir / "04_moving_average_residual.png")
    plot_fit(t, y, kernel_fit, "5(b): Gaussian kernel fit (bandwidth b=6)",
             out_dir / "05_gaussian_kernel_fit.png", ylabel)
    plot_residual(t, kernel_resid, "5(b): Gaussian-kernel residuals",
                  out_dir / "06_gaussian_kernel_residual.png")

    # 5(c) and 5(d): Fit the exact regressions given in the question.
    harmonic = ols_fit(y, t, with_spikes=False)
    with_spikes = ols_fit(y, t, with_spikes=True)
    plot_regression(t, y, harmonic, "5(c): Linear trend + 24-period sine/cosine",
                    out_dir / "07_periodic_regression.png", ylabel)
    plot_regression(t, y, with_spikes, "5(d): Trend + seasonality + periodic spike",
                    out_dir / "08_spike_regression.png", ylabel)

    # Make a tidy CSV with all observations, fitted values, and residuals.
    pd.DataFrame({
        "t": t, "original_x": x, "sqrt_x": transforms["sqrt"],
        "log_x": transforms["log"], "transformed_x": y,
        "spike": spike.astype(int), "moving_average_fit": ma_fit,
        "moving_average_residual": ma_resid, "gaussian_fit": kernel_fit,
        "gaussian_residual": kernel_resid, "periodic_ols_fit": harmonic["fitted"],
        "periodic_ols_residual": harmonic["residual"],
        "spike_ols_fit": with_spikes["fitted"],
        "spike_ols_residual": with_spikes["residual"],
    }).to_csv(out_dir / "series_with_fits.csv", index=False)

    coeff_rows = []
    model_rows = []
    for label, result in (("5c_periodic", harmonic), ("5d_with_spike", with_spikes)):
        for name, estimate in result["coefficients"].items():
            coeff_rows.append({"model": label, "parameter": name, "estimate": estimate})
        model_rows.append({
            "model": label,
            "number_of_coefficients": result["num_coefficients"],
            "rmse": result["rmse"], "mae": result["mae"],
            "r_squared": result["r_squared"],
            "relative_aic": result["aic_relative"],
            "lag1_residual_correlation": result["lag1_resid_corr"],
            "rmse_at_spikes": float(np.sqrt(np.mean(result["residual"][spike] ** 2))),
        })
    pd.DataFrame(coeff_rows).to_csv(out_dir / "model_coefficients.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out_dir / "model_comparison.csv", index=False)

    pct_rmse_reduction = 100 * (1 - with_spikes["rmse"] / harmonic["rmse"])
    lines = [
        "STAT GR5263 / Homework 2 / Problem 5 -- automatic results",
        "=" * 66,
        f"Data file: {data_file}", f"Number of observations: {n}",
        f"Original data min/max: {x.min():.6f} / {x.max():.6f}",
        "Time index: t=1,...,n",
        "",
        "5(a): Original data and transformation comparison",
        "Blockwise SDs are computed from first differences, excluding differences",
        "touching the known 24-period spikes. Lower CV = more stable variation.",
    ]
    for name, d in diag.items():
        formatted_sds = ", ".join(f"{v:.6f}" for v in d["sd_by_block"])
        lines.append(f"  {name:>8}: CV={d['cv_of_block_sds']:.4f}; five block SDs=[{formatted_sds}]")
    lines.extend([
        f"Selected transformation, comparing sqrt vs log: {selected.upper()}",
        "Note: selection is relative to the two REQUIRED transformations; it does",
        "not imply that transforming improves upon the untransformed data.",
        "Interpret the original plot and transformed plots when discussing the",
        "additive model and possible stationarity of its residual component.",
        "",
        "5(b): Smoothers",
        f"Moving average: centered (2k+1) = {2 * MA_HALF_WIDTH + 1} observations; k={MA_HALF_WIDTH}.",
        f"Gaussian kernel: bandwidth b={GAUSSIAN_BANDWIDTH:g}, truncated at +/-4b.",
        "At series edges, the available smoothing weights are renormalized.",
        f"MA residual RMSE: {np.sqrt(np.mean(ma_resid**2)):.6f}",
        f"Gaussian residual RMSE: {np.sqrt(np.mean(kernel_resid**2)):.6f}",
    ])
    for part, fit in (("5(c): Periodic trend model", harmonic), ("5(d): Model with spike", with_spikes)):
        lines.extend(["", part])
        for name, val in fit["coefficients"].items():
            lines.append(f"  {name:>12} = {val: .8f}")
        lines.extend([
            f"  RMSE={fit['rmse']:.6f}",
            f"  R^2={fit['r_squared']:.6f}",
            f"  Lag-one residual correlation={fit['lag1_resid_corr']:.6f}",
            f"  RMSE at spike times={np.sqrt(np.mean(fit['residual'][spike]**2)):.6f}",
        ])
    lines.extend([
        "",
        f"5(d) vs. 5(c): reduction in residual RMSE = {pct_rmse_reduction:.2f}%",
        "IMPORTANT: Autocorrelation may remain in the residuals: it can be",
        "consistent with a stationary error process; zero ACF is not required.",
        "All fitted equations and residuals use the SELECTED transformed scale.",
        "",
        "Saved 8 PNG plots + 4 CSV files + this summary.",
    ])
    summary = "\n".join(lines) + "\n"
    (out_dir / "results_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Output folder: {out_dir.resolve()}")
    print("SUCCESS: Problem 5(a)-(d) analyses and figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Homework 2 Problem 5 results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Optional input override (defaults to the Windows simu.txt path).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Optional output override (defaults to sheet2_q5_outputs).")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()


