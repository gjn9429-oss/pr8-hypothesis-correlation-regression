import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy import stats
from block1 import save_figure
import matplotlib.pyplot as plt

def compare_means(n1, mean1, std1, n2, mean2, std2):
    pooled_variance = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)
    pooled_std = pooled_variance**.5
    test = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2,
                                     equal_var=True, alternative="two-sided")
    welch = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2,
                                      equal_var=False, alternative="two-sided")
    difference = mean1 - mean2
    standard_error = (pooled_variance*(1/n1+1/n2))**.5
    df = n1+n2-2
    margin = stats.t.ppf(.975, df)*standard_error
    return {"difference": difference, "pooled_std": pooled_std, "df": df,
            "t": float(test.statistic), "pvalue": float(test.pvalue),
            "d": difference/pooled_std, "welch_p": float(welch.pvalue),
            "difference_ci95": [float(difference-margin), float(difference+margin)]}

def run():
    comparison = compare_means(50, 45.2, 8.1, 50, 38.6, 7.4)
    # Окремі навчальні дані для регресії; це не виміри A/B тесту.
    rng = np.random.default_rng(42)
    cpu = rng.uniform(20, 90, 50)
    latency = 10 + .5*cpu + rng.normal(0, 4, 50)
    correlation = stats.pearsonr(cpu, latency)
    fit = stats.linregress(cpu, latency)
    predicted = fit.intercept + fit.slope*cpu
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(cpu, latency, label="Модельні дані")
    order = np.argsort(cpu)
    ax.plot(cpu[order], predicted[order], color="tab:red", label="Лінійна регресія")
    ax.set(xlabel="Завантаження CPU, %", ylabel="Час відгуку, мс",
           title="Навчальна модель: CPU і час відгуку")
    ax.legend(); ax.grid(alpha=.25)
    save_figure(fig, "block2_regression.png")
    result = {"comparison": comparison, "cpu": cpu.tolist(), "latency": latency.tolist(),
              "r": float(correlation.statistic), "r_p": float(correlation.pvalue),
              "intercept": float(fit.intercept), "slope": float(fit.slope),
              "r_squared": float(fit.rvalue**2)}
    print("БЛОК 2. Redis A/B:", comparison)
    print("Парна регресія:", {k: result[k] for k in
          ["r", "r_p", "intercept", "slope", "r_squared"]})
    return result

if __name__ == "__main__":
    run()
