import pandas as pd
import numpy as np

from scipy.stats import spearmanr
from scipy.optimize import differential_evolution


# -------------------------------------------------
# Load datasets
# -------------------------------------------------

score = pd.read_parquet(
    "score_features.parquet"
)

temporal = pd.read_parquet(
    "temporal_gradient_features.parquet"
)


# merge temporal features
df = score.merge(
    temporal[
        [
            "seed",
            "alpha",
            "round",
            "client_id",
            "history_alignment",
            "global_history_alignment",
            "gradient_change",
        ]
    ],
    on=[
        "seed",
        "alpha",
        "round",
        "client_id",
    ],
    how="inner"
)


print(df.shape)


# -------------------------------------------------
# Remove invalid rows
# -------------------------------------------------

features = [
    "cosine_to_global",
    "diversity",
    "history_alignment",
    "global_history_alignment",
    "gradient_change",
    "redundancy",
]


df = df.dropna(
    subset=features + ["loo_score"]
)


print(
    "Removed:",
    3000-len(df)
)


X = df[features].values
y = df["loo_score"].values



# -------------------------------------------------
# Standardize
# -------------------------------------------------

X = (
    X - X.mean(axis=0)
) / (
    X.std(axis=0)+1e-8
)



# -------------------------------------------------
# Score function
# -------------------------------------------------

def score_function(w):

    prediction = X @ w

    corr,_ = spearmanr(
        prediction,
        y
    )

    return -corr



# -------------------------------------------------
# Optimize weights
# -------------------------------------------------

result = differential_evolution(
    score_function,
    bounds=[
        (-5,5),
        (-5,5),
        (-5,5),
        (-5,5),
        (-5,5),
        (-5,5),
    ],
    seed=42,
)


print(result)


weights = result.x


print("\nWeights")

for f,w in zip(features,weights):
    print(
        f,
        w
    )


corr,_ = spearmanr(
    X@weights,
    y
)

print(
    "\nFinal Spearman:",
    corr
)