import pandas as pd
from scipy.stats import spearmanr


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
            "gradient_change",
            "global_history_alignment"
        ]
    ],
    on=[
        "seed",
        "alpha",
        "round",
        "client_id"
    ]
)


groups = df.groupby(
    [
        "seed",
        "alpha",
        "round"
    ]
)


features = [
    "cosine_to_global",
    "diversity",
    "redundancy",
    "history_alignment",
    "gradient_change"
]


results = []


for name,g in groups:

    if len(g) < 3:
        continue

    for f in features:

        c,_ = spearmanr(
            g[f],
            g["loo_score"]
        )

        results.append(
            {
                "feature":f,
                "corr":c
            }
        )


res = pd.DataFrame(results)


print(
    res.groupby("feature")
    ["corr"]
    .agg(
        ["mean","std","count"]
    )
)