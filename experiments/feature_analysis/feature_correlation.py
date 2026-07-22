import pandas as pd

df = pd.read_parquet(
    "score_features.parquet"
)

temporal = pd.read_parquet(
    "temporal_gradient_features.parquet"
)

df = df.merge(
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
    ]
)


features = [
    "cosine_to_global",
    "diversity",
    "redundancy",
    "history_alignment",
    "global_history_alignment",
    "gradient_change",
    "update_norm",
    "grad_norm",
    "cosine_to_prev"
]


print(
    df[features + ["loo_score"]]
    .corr()["loo_score"]
    .sort_values()
)