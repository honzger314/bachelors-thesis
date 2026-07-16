import pandas as pd
import numpy as np
from scipy.stats import spearmanr


# Load feature sets
score = pd.read_parquet(
    "score_features.parquet"
)

temporal = pd.read_parquet(
    "temporal_gradient_features.parquet"
)


# Merge spatial + temporal gradient features
df = score.merge(
    temporal[
        [
            "seed",
            "alpha",
            "round",
            "client_id",
            "history_alignment",
            "global_history_alignment",
            "gradient_change"
        ]
    ],
    on=[
        "seed",
        "alpha",
        "round",
        "client_id"
    ],
    how="inner"
)


print("Merged dataset:", df.shape)


# Remove rows where temporal information does not exist
df = df.dropna(
    subset=[
        "loo_score",
        "cosine_to_global",
        "diversity",
        "redundancy",
        "history_alignment",
        "global_history_alignment",
        "gradient_change"
    ]
)


print("Evaluation dataset:", df.shape)


groups = df.groupby(
    ["seed", "alpha", "round"]
)


baseline_scores = []
optimized_scores = []


for _, g in groups:

    if len(g) < 3:
        continue


    loo = g["loo_score"]


    # Baseline
    baseline = g["cosine_to_global"]


    # Optimized score from within-round optimization
    score = (
        4.746 * g["cosine_to_global"]
        -1.431 * g["diversity"]
        +0.123 * g["history_alignment"]
        +0.401 * g["global_history_alignment"]
        -0.121 * g["gradient_change"]
        -1.502 * g["redundancy"]
    )


    b = spearmanr(
        baseline,
        loo
    ).statistic


    s = spearmanr(
        score,
        loo
    ).statistic


    if not np.isnan(b):
        baseline_scores.append(b)

    if not np.isnan(s):
        optimized_scores.append(s)



print()
print("Rounds evaluated:", len(optimized_scores))

print()

print("Baseline:")
print(np.mean(baseline_scores))

print()

print("Optimized:")
print(np.mean(optimized_scores))

print()

print("Improvement:")
print(
    np.mean(optimized_scores)
    -
    np.mean(baseline_scores)
)