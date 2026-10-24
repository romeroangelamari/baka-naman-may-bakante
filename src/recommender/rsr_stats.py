import numpy as np
import pandas as pd
from scipy.stats import binomtest

r = pd.read_csv("results/rsr_requests.csv")
w = r.pivot(index="request", columns="policy", values="top1_correct").astype(bool)
n = len(w)

print("Top-1 RSR with 95% confidence interval:")
for p in w.columns:
    rate = w[p].mean()
    half = 1.96 * np.sqrt(rate * (1 - rate) / n)
    print(f"  {p:20s} {rate*100:5.1f}%  ({(rate-half)*100:.1f} to {(rate+half)*100:.1f})")

pairs = [("lgbm_det_cautious", "random"), ("lgbm_det_cautious", "static_least_used"),
         ("lgbm_det_plain", "lgbm_det_cautious"), ("lgbm_true_cautious", "lgbm_det_cautious")]
rows = []
for a, b in pairs:
    only_a = int((w[a] & ~w[b]).sum())       # A right, B wrong
    only_b = int((~w[a] & w[b]).sum())       # B right, A wrong
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    rows.append({"A": a, "B": b, "A right only": only_a, "B right only": only_b,
                 "p_value": round(p, 4), "significant": p < 0.05})
out = pd.DataFrame(rows)
out.to_csv("results/rsr_mcnemar.csv", index=False)
print("\nMcNemar tests (paired, same 377 requests):")
print(out.to_string(index=False))