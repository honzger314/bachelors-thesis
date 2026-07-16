import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


# -----------------------------
# Load features
# -----------------------------

df = pd.read_parquet("full_gradient_features.parquet")

print(df.shape)


# -----------------------------
# Normalize features
# -----------------------------

feature_cols = [
    "cosine_to_global",
    "cosine_to_prev",
    "gradient_progress_cosine",
    "pca_0",
    "pca_1",
    "pca_2",
    "pca_3",
    "pca_4",
    "pca_5",
    "pca_6",
    "pca_7",
    "pca_8",
    "pca_9",
]


# -----------------------------
# Construct score components
# -----------------------------


# usefulness
df["alignment"] = df["cosine_to_global"]


# temporal consistency
df["temporal"] = df["gradient_progress_cosine"]


# diversity from PCA distance
pca_cols = [f"pca_{i}" for i in range(10)]

center = df[pca_cols].mean()

df["diversity"] = np.linalg.norm(
    df[pca_cols] - center.values,
    axis=1
)


# redundancy
# already computed from clustering
# redundancy = how many clients are close in gradient space
# using PCA coordinates

from sklearn.metrics.pairwise import cosine_similarity


pca_cols = [
    f"pca_{i}" for i in range(10)
]


redundancy = np.zeros(len(df))


for (seed, alpha, rnd), group in df.groupby(
    ["seed", "alpha", "round"]
):

    idx = group.index

    X = group[pca_cols].values

    sim = cosine_similarity(X)

    # ignore self similarity
    np.fill_diagonal(sim, 0)

    # closest other client
    redundancy[idx] = sim.max(axis=1)


df["redundancy"] = redundancy


# novelty
df["diversity"] = 1 - df["redundancy"]

# normalize everything

components = [
    "alignment",
    "diversity",
    "temporal",
    "redundancy"
]


for c in components:
    df[c] = (
        df[c]-df[c].mean()
    ) / (
        df[c].std()+1e-8
    )


df.to_parquet(
    "score_features.parquet"
)


print(df[components].describe())