import pandas as pd
import numpy as np

from scipy.stats import spearmanr, kendalltau


# ============================
# Load data
# ============================

df = pd.read_parquet(
    "full_gradient_features.parquet"
)

score = pd.read_parquet(
    "score_features.parquet"
)

temporal = pd.read_parquet(
    "temporal_gradient_features.parquet"
)


keys = [
    "seed",
    "alpha",
    "round",
    "client_id"
]


# add diversity/redundancy/alignment
df = df.merge(
    score[
        keys +
        [
            "alignment",
            "diversity",
            "redundancy"
        ]
    ],
    on=keys,
    how="left"
)


# add temporal information
df = df.merge(
    temporal[
        keys +
        [
            "history_alignment",
            "global_history_alignment",
            "gradient_change"
        ]
    ],
    on=keys,
    how="left"
)


print("Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nNaN counts:")
print(df.isna().sum())

# ============================
# Define metrics
# ============================

# ============================
# Define metrics
# ============================

metrics = {

    # Existing baselines

    "cosine_to_global":
        lambda g: g["cosine_to_global"],


    "update_norm":
        lambda g: g["update_norm"],


    "grad_norm":
        lambda g: g["grad_norm"],


    # --------------------------------
    # New combined metrics
    # --------------------------------


    # Dot product:
    # magnitude + direction
    "dot_product":
        lambda g:
            g["update_norm"] *
            g["cosine_to_global"],


    # Explicit magnitude weighted alignment
    "magnitude_alignment":
        lambda g:
            g["update_norm"] *
            g["cosine_to_global"],


    # Diversity adjusted usefulness
    #
    # high if:
    #   - aligned with global direction
    #   - different from others
    #
    "diversity_alignment":
        lambda g:
            g["cosine_to_global"] *
            g["diversity"],


    # Temporal adjusted usefulness
    #
    # high if:
    #   - currently useful
    #   - historically consistent
    #
    "temporal_alignment":
        lambda g:
            g["cosine_to_global"] *
            g["history_alignment"],


    # Global history consistency
    "global_temporal_alignment":
        lambda g:
            g["cosine_to_global"] *
            g["global_history_alignment"],


}


# ============================
# Evaluate per round
# ============================

results = []


groups = df.groupby(
    [
        "seed",
        "alpha",
        "round"
    ]
)


for name, g in groups:

    # Need complete rounds
    if len(g) != 10:
        continue


    # Ground truth ranking
    loo_rank = g["loo_score"].rank(
        ascending=False
    )


    for metric_name, func in metrics.items():

        scores = func(g)


        # ignore unavailable metrics
        valid = scores.notna()


        if valid.sum() < 5:
            continue


        scores_valid = scores[valid]
        loo_valid = loo_rank[valid]


        # rank predicted contribution
        score_rank = scores_valid.rank(
            ascending=False
        )


        rho = spearmanr(
            score_rank,
            loo_valid
        ).statistic


        tau = kendalltau(
            score_rank,
            loo_valid
        ).statistic


        # top 3 overlap
        top_metric = set(
            g.loc[
                scores_valid.nlargest(3).index,
                "client_id"
            ]
        )


        top_loo = set(
            g.loc[
                loo_valid.nlargest(3).index,
                "client_id"
            ]
        )


        overlap = len(
            top_metric & top_loo
        ) / 3


        results.append(
            {
                "metric": metric_name,

                "spearman": rho,
                "kendall": tau,
                "top3_overlap": overlap,

                # metadata
                "alpha": name[1],
                "seed": name[0],
                "round": name[2]
            }
        )


# ============================
# Aggregate by alpha
# ============================

results = pd.DataFrame(results)

alpha_summary = (
    results
    .groupby(["alpha", "metric"])
    .agg(
        mean_spearman=("spearman", "mean"),
        mean_kendall=("kendall", "mean"),
        mean_top3=("top3_overlap", "mean"),
        rounds=("spearman", "count")
    )
    .reset_index()
)


print("\n====================")
print("Results by alpha")
print("====================")

print(
    alpha_summary
    .sort_values(
        ["alpha", "mean_spearman"],
        ascending=[True, False]
    )
)


alpha_summary.to_csv(
    "metric_ranking_by_alpha.csv",
    index=False
)


# ============================
# Pivot table for thesis
# ============================

pivot = (
    alpha_summary
    .pivot(
        index="metric",
        columns="alpha",
        values="mean_spearman"
    )
)


print("\n====================")
print("Spearman by alpha")
print("====================")

print(pivot)


pivot.to_csv(
    "metric_spearman_by_alpha.csv"
)