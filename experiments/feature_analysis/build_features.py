from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# Location of experiment results
# -------------------------------------------------

ROOT = Path("results2") / "results"

print("Searching in:", ROOT.resolve())

parquet_files = list(ROOT.rglob("telemetry.parquet"))

print(f"Found {len(parquet_files)} parquet files")


# -------------------------------------------------
# Load telemetry
# -------------------------------------------------

dfs = []

for parquet_file in parquet_files:

    print("Loading:", parquet_file)

    df = pd.read_parquet(parquet_file)

    # Extract metadata
    parts = parquet_file.parent.name.split("_")

    seed = int(parts[3])
    alpha = float(parts[5])

    df["seed"] = seed
    df["alpha"] = alpha

    dfs.append(df)


if len(dfs) == 0:
    raise RuntimeError("No parquet files found")


df = pd.concat(dfs, ignore_index=True)


print("\nOriginal dataset:")
print(df.shape)


# -------------------------------------------------
# Feature computation
# -------------------------------------------------

def compute_round_features(group):

    gradients = np.stack(
        group["gradient"].values
    )

    # pairwise cosine matrix
    sim = cosine_similarity(gradients)


    features = []

    for i in range(len(group)):

        # remove self similarity
        others = np.delete(sim[i], i)


        mean_cos = np.mean(others)

        max_cos = np.max(others)

        nearest_cos = max_cos


        features.append({
            "mean_cosine": mean_cos,
            "max_cosine": max_cos,
            "nearest_neighbor_cosine": nearest_cos,

            # high = similar to everyone else
            "redundancy": mean_cos,

            # high = unique update
            "diversity": 1 - mean_cos
        })


    feature_df = pd.DataFrame(
        features,
        index=group.index
    )

    return feature_df



# -------------------------------------------------
# Apply per FL round
# -------------------------------------------------

print("\nComputing redundancy features...")


feature_parts = []

groups = df.groupby(
    [
        "seed",
        "alpha",
        "round"
    ]
)


total = len(groups)

for idx, (_, group) in enumerate(groups):

    if idx % 10 == 0:
        print(
            f"Processing round {idx}/{total}"
        )

    feature_parts.append(
        compute_round_features(group)
    )


new_features = pd.concat(feature_parts)


# attach features
df = pd.concat(
    [
        df,
        new_features
    ],
    axis=1
)


# -------------------------------------------------
# Save
# -------------------------------------------------

print("\nFinal dataset:")
print(df.shape)


print("\nNew columns:")
print(
    [
        c for c in df.columns
        if c not in [
            "gradient"
        ]
    ]
)

# Remove huge gradient column before saving feature table
df_save = df.drop(columns=["gradient"])

df_save.to_parquet(
    "features_step1.parquet"
)

print("\nSaved features_step1.parquet")
df = pd.read_parquet("features_step1.parquet")

print(df.shape)
print(df.columns)