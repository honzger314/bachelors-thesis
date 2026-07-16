import pandas as pd


df = pd.read_parquet(
    "features_step1.parquet"
)


print("Dataset:")
print(df.shape)


# -------------------------------------------------
# Correlation with LOO
# -------------------------------------------------

features = [
    "update_norm",
    "grad_norm",
    "cosine_to_global",
    "cosine_to_prev",
    "cosine_grad_update",
    "mean_cosine",
    "max_cosine",
    "nearest_neighbor_cosine",
    "redundancy",
    "diversity"
]


print("\nCorrelation with LOO:")
print(
    df[features + ["loo_score"]]
    .corr()["loo_score"]
    .sort_values()
)


# -------------------------------------------------
# Compare top and bottom contributors
# -------------------------------------------------

print("\nHighest LOO clients:")
print(
    df.sort_values(
        "loo_score",
        ascending=False
    )
    [
        [
            "seed",
            "alpha",
            "round",
            "client_id",
            "loo_score",
            "redundancy",
            "diversity"
        ]
    ]
    .head(10)
)


print("\nLowest LOO clients:")
print(
    df.sort_values(
        "loo_score",
        ascending=True
    )
    [
        [
            "seed",
            "alpha",
            "round",
            "client_id",
            "loo_score",
            "redundancy",
            "diversity"
        ]
    ]
    .head(10)
)


# -------------------------------------------------
# By non-IID severity
# -------------------------------------------------

print("\nFeature averages by alpha:")
print(
    df.groupby("alpha")
    [
        [
            "loo_score",
            "redundancy",
            "diversity"
        ]
    ]
    .mean()
)