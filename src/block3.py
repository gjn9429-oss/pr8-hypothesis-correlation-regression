import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import statsmodels.api as sm
from block1 import save_figure
import matplotlib.pyplot as plt

def run():
    rng = np.random.default_rng(43)
    n = 100
    cpu = rng.uniform(20, 90, n)
    requests = rng.uniform(5, 50, n)
    noise = rng.normal(0, 4, n)
    latency = 10 + .5*cpu + 1.2*requests + noise
    design = sm.add_constant(np.column_stack((cpu, requests)))
    model = sm.OLS(latency, design).fit()
    predicted = model.fittedvalues
    residuals = model.resid
    # ANOVA регресії: пояснена та залишкова суми квадратів.
    regression_ss = float(sum((predicted - np.mean(latency))**2))
    error_ss = float(sum(residuals**2))
    total_ss = float(sum((latency - np.mean(latency))**2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].scatter(predicted, latency)
    low, high = min(predicted.min(), latency.min()), max(predicted.max(), latency.max())
    axes[0].plot([low, high], [low, high], "r--")
    axes[0].set(xlabel="Прогноз Ŷ, мс", ylabel="Спостереження Y, мс", title="Y проти Ŷ")
    axes[1].scatter(predicted, residuals)
    axes[1].axhline(0, color="tab:red", linestyle="--")
    axes[1].set(xlabel="Прогноз Ŷ, мс", ylabel="Залишок, мс", title="Залишки моделі")
    for ax in axes: ax.grid(alpha=.25)
    save_figure(fig, "block3_diagnostics.png")
    result = {"cpu": cpu.tolist(), "requests": requests.tolist(), "latency": latency.tolist(),
              "coefficients": model.params.tolist(), "coefficient_p": model.pvalues.tolist(),
              "r_squared": float(model.rsquared), "adjusted_r_squared": float(model.rsquared_adj),
              "f": float(model.fvalue), "pvalue": float(model.f_pvalue),
              "regression_ss": regression_ss, "error_ss": error_ss, "total_ss": total_ss}
    print("БЛОК 3. Коефіцієнти:", model.params)
    print(f"R²={model.rsquared:.6f}; adjusted R²={model.rsquared_adj:.6f}")
    print(f"F={model.fvalue:.6f}; p={model.f_pvalue:.6e}")
    print("ANOVA SS: регресія, залишок, загальна:", regression_ss, error_ss, total_ss)
    return result

if __name__ == "__main__":
    run()
