# src/detection/make_splits.py
# Assigns every sequence (whole video) to train, val or test.
import pandas as pd

gt = pd.read_csv("data/processed/gt_counts.csv")
seqs = gt[["dataset", "sequence"]].drop_duplicates()

# CEPDOF low-light videos: used only for testing
LOW_LIGHT = ["off", "irfilter", "irill"]
# One normal-light CEPDOF video kept for validation (choosing settings)
VAL_VIDEO = "lunch3"

def assign(row):
    seq = str(row.sequence).lower()
    if row.dataset != "cepdof":
        return "test"                   # FRIDA, SCUT-HEAD, Library Seats, RHC, AIRC
    if any(word in seq for word in LOW_LIGHT):
        return "test"                   # low light: hardest case, test only
    if VAL_VIDEO in seq:
        return "val"
    return "train"

seqs["split"] = seqs.apply(assign, axis=1)
seqs.to_csv("data/processed/splits.csv", index=False)

print(seqs.sort_values(["dataset", "split", "sequence"]).to_string(index=False))
print(gt.merge(seqs).groupby(["dataset", "split"]).size())   # frames per group