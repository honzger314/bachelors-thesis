import pandas as pd
import numpy as np

from scipy.optimize import differential_evolution


# ----------------------------
# Load data
# ----------------------------

score = pd.read_parquet(
    "score_features.parquet"
)

temporal = pd.read_parquet(
    "temporal_gradient_features.parquet"
)


df = score.merge(
    temporal[
        [
            "seed",
            "alpha",
            "round",
            "client_id",
            "history_alignment",
            "gradient_change",
            "global_history_alignment"
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


features = [
    "cosine_to_global",
    "diversity",
    "history_alignment",
    "global_history_alignment",
    "gradient_change",
    "redundancy"
]


df = df.dropna(
    subset=features + ["loo_score"]
)


print("Dataset:", df.shape)


# ----------------------------
# Standardize features
# ----------------------------

for f in features:
    df[f] = (
        df[f] - df[f].mean()
    ) / (
        df[f].std() + 1e-8
    )


# ----------------------------
# Prepare rounds
# ----------------------------

groups = list(
    df.groupby(
        [
            "seed",
            "alpha",
            "round"
        ]
    )
)


prepared = []

for _, g in groups:

    X = g[features].values.astype(
        np.float32
    )

    y = g["loo_score"].values.astype(
        np.float32
    )

    prepared.append(
        (X, y)
    )


print(
    "Rounds:",
    len(prepared)
)


# ----------------------------
# Rank helper
# ----------------------------

def rank(x):

    return np.argsort(
        np.argsort(x)
    ).astype(
        np.float32
    )



# Precompute LOO ranks

prepared_ranked = []

for X,y in prepared:

    prepared_ranked.append(
        (
            X,
            rank(y)
        )
    )



# ----------------------------
# Objective
# ----------------------------

def objective(w):

    correlations = []

    for X, y_rank in prepared_ranked:

        prediction = X @ w

        pred_rank = rank(
            prediction
        )


        c = np.corrcoef(
            pred_rank,
            y_rank
        )[0,1]


        if not np.isnan(c):
            correlations.append(c)


    return -np.mean(
        correlations
    )



# ----------------------------
# Optimize
# ----------------------------

result = differential_evolution(
    objective,
    bounds=[
        (-5,5)
    ] * len(features),
    seed=42,
    popsize=10,
    maxiter=20,
    polish=True
)


print(result)


print("\nWeights:")

for f,w in zip(features,result.x):
    print(
        f,
        w
    )


print(
    "\nMean within-round Spearman:",
    -result.fun
)