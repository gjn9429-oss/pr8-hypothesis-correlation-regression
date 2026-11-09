import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAPHICS = Path(__file__).resolve().parent.parent / "graphics"

def save_figure(fig, name):
    GRAPHICS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(GRAPHICS / name, dpi=200)
    plt.close(fig)

def run():
    observed = np.array([8, 22, 44, 52, 40, 20, 10, 4])
    n, mean, std = 200, 112.5, 21.4
    edges = np.arange(60, 181, 15)
    probability = np.diff(stats.norm.cdf(edges, loc=mean, scale=std))
    expected = n * probability
    literal_stat = float(sum((observed - expected)**2 / expected))
    # Вісім заданих інтервалів не охоплюють усю числову вісь.
    full_edges = np.array([-np.inf, 75, 90, 105, 120, 135, 150, 165, np.inf])
    full_expected = n * np.diff(stats.norm.cdf(full_edges, mean, std))
    # Останні дві групи об'єднано, щоб усі очікувані частоти були >= 5.
    merged_observed = np.append(observed[:6], sum(observed[6:]))
    merged_expected = np.append(full_expected[:6], sum(full_expected[6:]))
    df = len(merged_observed) - 1 - 2
    statistic = float(sum((merged_observed - merged_expected)**2 / merged_expected))
    pvalue = float(stats.chi2.sf(statistic, df))
    critical = float(stats.chi2.ppf(.95, df))
    print("БЛОК 1. Критерій Пірсона")
    print("За початковими межами: E =", expected)
    print("Сума E:", sum(expected), "chi2:", literal_stat,
          "df=5; p:", stats.chi2.sf(literal_stat, 5))
    print("Після врахування хвостів та об'єднання:")
    for o, e in zip(merged_observed, merged_expected):
        print(f"O={o:3d}; E={e:.6f}; внесок={(o-e)**2/e:.6f}")
    print(f"df={df}; chi2={statistic:.6f}; critical={critical:.6f}; p={pvalue:.6f}")
    print("H0 не відхиляємо" if pvalue >= .05 else "H0 відхиляємо")
    return {"observed": observed.tolist(), "probability": probability.tolist(),
            "expected": expected.tolist(), "literal_stat": literal_stat,
            "literal_p": float(stats.chi2.sf(literal_stat, 5)),
            "literal_critical": float(stats.chi2.ppf(.95, 5)),
            "full_expected": full_expected.tolist(),
            "merged_observed": merged_observed.tolist(),
            "merged_expected": merged_expected.tolist(), "df": df,
            "statistic": statistic, "critical": critical, "pvalue": pvalue}

if __name__ == "__main__":
    run()
