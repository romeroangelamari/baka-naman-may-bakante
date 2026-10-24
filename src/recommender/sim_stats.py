import pandas as pd
from scipy.stats import wilcoxon

r = pd.read_csv("results/simulation_runs.csv")
p = r.pivot_table(index=["day", "hour", "students", "seed"], columns="policy", values="overflow_rate")
pairs = [("soft_cautious", "greedy_cautious"), ("soft_plain", "greedy_plain"), ("soft_cautious", "soft_plain")]
rows = []
for n, g in p.groupby(level="students"):
    for a, b in pairs:
        d = g[a] - g[b]
        pval = wilcoxon(g[a], g[b]).pvalue if (d != 0).any() else 1.0
        rows.append({"students": n, "A": a, "B": b, "A_overflow_%": round(g[a].mean() * 100, 2),
                     "B_overflow_%": round(g[b].mean() * 100, 2), "runs_different": int((d != 0).sum()),
                     "runs": len(d), "p_value": round(pval, 4), "significant": pval < 0.05})
out = pd.DataFrame(rows)
out.to_csv("results/simulation_tests.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))