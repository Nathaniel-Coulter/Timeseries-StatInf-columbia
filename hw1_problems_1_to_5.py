#hw1_problems_1_to_5.py
"""
Disclaimer:
I had chatgpt go in and annotate / make minimal organizational changes to the slop that I created so you wouldn't see all the errors and my janky paths despite it still running. 

STAT 4263/5263 — Homework 1
Problems 1–5: simulation, plots, and numerical/symbolic checks

Purpose
-------
This script supports the written homework:
- Problem 1: generate the required simulations and time-series plots.
- Problems 2–5: print/save exact formulas and numerical sanity checks that
  can be used while writing the mathematical derivations in LaTeX.

AI disclosure note:
This script was developed with AI assistance. If submitted, disclose that use
in accordance with the course instructions.

Dependencies:
    numpy
    matplotlib

Run:
    python hw1_problems_1_to_5.py

Outputs:
    hw1_outputs_1_to_5/
        summary.txt
        p1_simulations.csv
        p1a_normal_rep1.png ... rep4.png
        p1b_chisquare_rep1.png ... rep4.png
        p1c_t5_rep1.png ... rep4.png
        p2d_lag1_covariance_check.csv
        p3_typical_path.png
        p4_probability_mc_check.txt
"""

from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 5263
N = 48
N_REPS = 4
MC_REPS = 500_000
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "hw1_outputs_1_to_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary_lines = []


def log(line=""):
    print(line)
    summary_lines.append(str(line))


def heading(title):
    bar = "=" * 72
    log()
    log(bar)
    log(title)
    log(bar)


def sample_acf(x, lag=1):
    """Sample ACF using the definition in the homework."""
    x = np.asarray(x, dtype=float)
    xbar = x.mean()
    centered = x - xbar
    numerator = np.dot(centered[lag:], centered[:-lag]) if lag > 0 else np.dot(centered, centered)
    denominator = np.dot(centered, centered)
    return float(numerator / denominator)


def save_series_plot(values, title, filename):
    t = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(t, values, marker="o", markersize=3, linewidth=1)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


# =====================================================================
# PROBLEM 1
# =====================================================================
heading("PROBLEM 1 — IID simulations and sample autocorrelation")
log("Using n = 48 observations per realization and 4 independent realizations")
log("for each distribution. For the Normal case we use N(0,1), a standard")
log("choice because location/scale do not affect the independence lesson.")

simulation_rows = []

heading("Problem 1(a) — Independent Normal series")
for rep in range(1, N_REPS + 1):
    x = rng.normal(loc=0.0, scale=1.0, size=N)
    rho1 = sample_acf(x, 1)
    save_series_plot(x, f"Problem 1(a): N(0,1) IID simulation — replicate {rep}", f"p1a_normal_rep{rep}.png")
    log(f"replicate {rep}: sample lag-1 ACF = {rho1:+.4f}")
    for t, value in enumerate(x, start=1):
        simulation_rows.append(["normal", rep, t, value])

heading("Problem 1(b) — Independent chi-square(df=2) series")
for rep in range(1, N_REPS + 1):
    x = rng.chisquare(df=2, size=N)
    rho1 = sample_acf(x, 1)
    save_series_plot(x, f"Problem 1(b): Chi-square(df=2) IID simulation — replicate {rep}", f"p1b_chisquare_rep{rep}.png")
    log(f"replicate {rep}: sample lag-1 ACF = {rho1:+.4f}")
    for t, value in enumerate(x, start=1):
        simulation_rows.append(["chi_square_df2", rep, t, value])

heading("Problem 1(c) — Independent t(df=5) series")
for rep in range(1, N_REPS + 1):
    x = rng.standard_t(df=5, size=N)
    rho1 = sample_acf(x, 1)
    save_series_plot(x, f"Problem 1(c): t(df=5) IID simulation — replicate {rep}", f"p1c_t5_rep{rep}.png")
    log(f"replicate {rep}: sample lag-1 ACF = {rho1:+.4f}")
    for t, value in enumerate(x, start=1):
        simulation_rows.append(["t_df5", rep, t, value])

with open(OUT_DIR / "p1_simulations.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["distribution", "replicate", "t", "value"])
    writer.writerows(simulation_rows)

heading("Problem 1(d) — Evaluate the supplied AI autocorrelation interpretation")
benchmark = 1.96 / math.sqrt(N)
given_acfs = {"Normal": 0.27, "Chi-square": -0.08, "t(df=5)": 0.04}
log(f"Approximate benchmark: ±1.96/sqrt(48) = ±{benchmark:.4f}")
for name, value in given_acfs.items():
    inside = abs(value) <= benchmark
    log(f"{name:10s}: rho_hat(1) = {value:+.2f}; |rho_hat| {'<=' if inside else '>'} {benchmark:.4f} -> {'inside' if inside else 'outside'} the approximate bounds.")
log("Key check: 0.27 is still inside the approximate ±1.96/sqrt(48) bounds.")
log("Therefore one observed sample ACF of 0.27 is not enough, by this benchmark,")
log("to conclude that the underlying observations are dependent.")
log("Repeating the simulation demonstrates sampling variability: IID series can")
log("produce noticeably nonzero sample autocorrelations by chance.")


# =====================================================================
# PROBLEM 2
# =====================================================================
heading("PROBLEM 2 — Weak stationarity, ACVF, ACF, and dependence")

heading("Problem 2(a) — X_t = Z_t Z_{t-2}")
log("Exact results:")
log("E[X_t] = 0")
log("gamma_X(0) = 1")
log("gamma_X(h) = 0 for every h != 0")
log("rho_X(0) = 1, rho_X(h) = 0 for every h != 0")
log("Conclusion: weakly stationary.")
log("Important: zero autocorrelation at nonzero lags does NOT make the process IID.")

heading("Problem 2(b) — X_t = a + b Z_t + c Z_{t-1}")
log("Exact results (assuming b^2 + c^2 > 0):")
log("E[X_t] = a")
log("gamma_X(0) = b^2 + c^2")
log("gamma_X(1) = gamma_X(-1) = b c")
log("gamma_X(h) = 0 for |h| > 1")
log("rho_X(1) = rho_X(-1) = b c / (b^2 + c^2)")
log("rho_X(h) = 0 for |h| > 1")
log("Conclusion: weakly stationary.")
log("Edge case: if b = c = 0, the process is constant a and its ACF is undefined because variance is zero.")

heading("Problem 2(c) — X_t = Z_t - Z_{t-3}")
log("Exact results:")
log("E[X_t] = 0")
log("gamma_X(0) = 2")
log("gamma_X(3) = gamma_X(-3) = -1")
log("gamma_X(h) = 0 otherwise")
log("rho_X(3) = rho_X(-3) = -1/2")
log("rho_X(h) = 0 otherwise for h != 0")
log("Conclusion: weakly stationary.")

heading("Problem 2(d) — X_t = Z_t cos(c t) + Z_{t-1} sin(c t)")
log("E[X_t] = 0 and Var(X_t) = 1 for every t.")
log("At lag 1:")
log("gamma_t(1) = cos(c t) sin(c(t+1))")
log("           = 1/2 [sin(c) + sin(2 c t + c)].")
log("For a generic constant c this depends on t, so the process is NOT weakly stationary.")
log("Special case: if c is an integer multiple of pi, sin(c t)=0 for integer t")
log("and X_t = (+/-) Z_t; then the process is IID N(0,1) and is stationary.")

c_demo = 0.7
rows_2d = []
for t in range(0, 12):
    gamma1_t = math.cos(c_demo * t) * math.sin(c_demo * (t + 1))
    rows_2d.append([t, c_demo, gamma1_t])
with open(OUT_DIR / "p2d_lag1_covariance_check.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t", "c_demo", "gamma_t_lag1"])
    writer.writerows(rows_2d)
log(f"Numerical illustration saved for c={c_demo}: p2d_lag1_covariance_check.csv")

heading("Problem 2(e) — Is process (a) IID?")
log("X_t^2 X_{t+2}^2 = Z_t^4 Z_{t-2}^2 Z_{t+2}^2")
log("E[X_t^2 X_{t+2}^2] = E[Z_t^4] E[Z_{t-2}^2] E[Z_{t+2}^2] = 3")
log("E[X_t^2] E[X_{t+2}^2] = 1 * 1 = 1")
log("Comparison: 3 != 1")
log("Therefore X_t and X_{t+2} are not independent, despite zero covariance.")

z = rng.normal(size=(MC_REPS, 3))
x_t = z[:, 1] * z[:, 0]
x_t2 = z[:, 2] * z[:, 1]
mc_joint = np.mean((x_t**2) * (x_t2**2))
mc_product = np.mean(x_t**2) * np.mean(x_t2**2)
log(f"Monte Carlo check ({MC_REPS:,} draws):")
log(f"  E[X_t^2 X_(t+2)^2] ≈ {mc_joint:.4f} (exact 3)")
log(f"  E[X_t^2]E[X_(t+2)^2] ≈ {mc_product:.4f} (exact 1)")


# =====================================================================
# PROBLEM 3
# =====================================================================
heading("PROBLEM 3 — Y_t = X for all t")

heading("Problem 3(a) — Strict and weak stationarity")
log("All coordinates are the same random variable X.")
log("Any shifted finite vector (Y_{t1+h},...,Y_{tk+h}) equals (X,...,X),")
log("the same as (Y_{t1},...,Y_{tk}); hence the process is strictly stationary.")
log("Also E[Y_t]=mu and Cov(Y_t,Y_{t+h})=sigma^2 for every h,")
log("so it is weakly stationary.")

heading("Problem 3(b) — Autocovariance")
log("gamma_Y(h) = sigma^2 for every lag h.")
log("If sigma^2 > 0, rho_Y(h) = 1 for every h.")

heading("Problem 3(c) — Typical time plot")
illustrative_x = 1.25
t = np.arange(1, 49)
y = np.full_like(t, illustrative_x, dtype=float)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(t, y, marker="o", markersize=3, linewidth=1)
ax.set_title("Problem 3(c): illustrative realization of Y_t = X")
ax.set_xlabel("Time t")
ax.set_ylabel("Realized value X(omega)")
ax.set_ylim(illustrative_x - 1.0, illustrative_x + 1.0)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "p3_typical_path.png", dpi=180)
plt.close(fig)
log("Saved p3_typical_path.png.")
log("The numerical height 1.25 is arbitrary; the mathematical point is that a single realization is a horizontal line because Y_t equals the same X at every t.")

heading("Problem 3(d) — Variance of the sample mean")
log("Ybar_n = (1/n) sum Y_t = (1/n)(n X) = X.")
log("Therefore Var(Ybar_n) = Var(X) = sigma^2, not sigma^2/n.")
log("Observing the process longer repeats the same random draw and provides no new independent information about mu.")


# =====================================================================
# PROBLEM 4
# =====================================================================
heading("PROBLEM 4 — X_t = sin(2 pi U t), U ~ Uniform(0,1)")

heading("Problem 4(a) — Mean and autocovariance")
log("For integer t >= 1:")
log("E[X_t] = integral_0^1 sin(2 pi u t) du = 0.")
log("E[X_t^2] = 1/2, so gamma(0) = 1/2.")
log("For h != 0, orthogonality of distinct integer-frequency sine functions gives E[X_t X_{t+h}] = 0.")
log("Thus gamma(h) = 1/2 if h=0 and 0 otherwise.")

heading("Problem 4(b) — Weak stationarity")
log("The mean is constant (0), the variance is constant (1/2), and the autocovariance depends only on the lag. Therefore X_t is weakly stationary.")

heading("Problem 4(c) — Not strictly stationary")
threshold = math.sin(math.pi / 3)
p23_exact = 1.0 / 36.0
log("Using the hint threshold sin(pi/3) = sqrt(3)/2:")
log("P(X_1 > sin(pi/3), X_2 > sin(pi/3)) = 0.")
log("P(X_2 > sin(pi/3), X_3 > sin(pi/3)) = 1/36.")
log("These joint probabilities differ under a one-step time shift, so the process is not strictly stationary.")

u = rng.random(MC_REPS)
X1 = np.sin(2 * np.pi * u)
X2 = np.sin(4 * np.pi * u)
X3 = np.sin(6 * np.pi * u)
p12_mc = np.mean((X1 > threshold) & (X2 > threshold))
p23_mc = np.mean((X2 > threshold) & (X3 > threshold))
prob_text = (
    f"Problem 4(c) Monte Carlo sanity check with {MC_REPS:,} U draws\n"
    f"P(X1 > sin(pi/3), X2 > sin(pi/3)) ≈ {p12_mc:.6f} (exact 0)\n"
    f"P(X2 > sin(pi/3), X3 > sin(pi/3)) ≈ {p23_mc:.6f} (exact {p23_exact:.6f})\n"
)
(OUT_DIR / "p4_probability_mc_check.txt").write_text(prob_text, encoding="utf-8")
log(f"Monte Carlo: first probability ≈ {p12_mc:.6f}; second ≈ {p23_mc:.6f}.")
log("Saved p4_probability_mc_check.txt.")

heading("Problem 4(d) — Sample mean variance versus IID-CLT claim")
log("Because gamma(h)=0 for h != 0 and Var(X_t)=1/2:")
log("Var(Xbar_n) = (1/n^2) * n * (1/2) = 1/(2n).")
log("That variance calculation IS justified by the autocovariances.")
log("However, zero autocorrelation does not imply independence.")
log("Therefore the usual IID central limit theorem cannot be invoked merely from the fact that all nonzero-lag autocorrelations are zero.")
log("The homework explicitly says the actual limiting distribution need not be derived.")


# =====================================================================
# PROBLEM 5
# =====================================================================
heading("PROBLEM 5 — Z_t = X_t + Y_t")

heading("Problem 5(a) — Stationarity and ACVF of the sum")
log("Assume X and Y are weakly stationary and X_s is uncorrelated with Y_t for every pair s,t.")
log("E[Z_t] = E[X_t] + E[Y_t] = mu_X + mu_Y, constant in t.")
log("gamma_Z(h) = Cov(X_t+Y_t, X_{t+h}+Y_{t+h})")
log("           = gamma_X(h) + gamma_Y(h),")
log("because both cross-covariance terms are zero.")
log("Hence Z_t is weakly stationary.")

heading("Problem 5(b) — Do +0.8 and -0.8 correlations cancel?")
rho_x, rho_y = 0.8, -0.8
var_x, var_y = 1.0, 4.0
gamma_x_1 = rho_x * var_x
gamma_y_1 = rho_y * var_y
gamma_z_0 = var_x + var_y
gamma_z_1 = gamma_x_1 + gamma_y_1
rho_z_1 = gamma_z_1 / gamma_z_0
log(f"gamma_X(1) = rho_X(1) Var(X_t) = {rho_x} * {var_x:g} = {gamma_x_1:g}")
log(f"gamma_Y(1) = rho_Y(1) Var(Y_t) = {rho_y} * {var_y:g} = {gamma_y_1:g}")
log(f"gamma_Z(0) = Var(X_t)+Var(Y_t) = {gamma_z_0:g}")
log(f"gamma_Z(1) = gamma_X(1)+gamma_Y(1) = {gamma_z_1:g}")
log(f"rho_Z(1) = gamma_Z(1)/gamma_Z(0) = {rho_z_1:g}")
log("So the two lag-1 correlations do NOT cancel here.")
log("More generally, if rho_X(1) = -rho_Y(1) != 0, cancellation requires Var(X_t) = Var(Y_t), equivalently gamma_X(1)+gamma_Y(1)=0.")

heading("DONE")
log(f"All outputs saved to: {OUT_DIR}")
log("Use the plots/results as support when writing the LaTeX derivations.")
log("Remember: the numerical checks for Problems 2–5 do not replace proofs.")

(OUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print()
print(f"Script complete. Summary: {OUT_DIR / 'summary.txt'}")
