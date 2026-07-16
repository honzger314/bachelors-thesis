from pathlib import Path
import pandas as pd

# -------------------------------------------------
# Location of the experiment results
# -------------------------------------------------
ROOT = Path("results2") / "results"

print("Searching in:", ROOT.resolve())

parquet_files = list(ROOT.rglob("telemetry.parquet"))

print(f"Found {len(parquet_files)} parquet files.\n")

for f in parquet_files:
    print(f)

# -------------------------------------------------
# Load all telemetry files
# -------------------------------------------------
dfs = []

for parquet_file in parquet_files:

    df = pd.read_parquet(parquet_file)

    # Folder name example:
    # run_1_seed_4_alpha_0.1
    parts = parquet_file.parent.name.split("_")

    seed = int(parts[3])
    alpha = float(parts[5])

    df["seed"] = seed
    df["alpha"] = alpha

    dfs.append(df)

if len(dfs) == 0:
    raise RuntimeError("No telemetry.parquet files were found.")

all_df = pd.concat(dfs, ignore_index=True)

# -------------------------------------------------
# Verify dataset
# -------------------------------------------------
print("\nDataset shape:")
print(all_df.shape)

print("\nColumns:")
print(all_df.columns.tolist())

print("\nFirst five rows:")
print(all_df.head())

print("\nRows per (seed, alpha):")
print(
    all_df.groupby(["seed", "alpha"]).size()
)
all_df.to_pickle("all_telemetry.pkl")
df = pd.read_pickle("all_telemetry.pkl")

print(df["loo_score"].describe())
features = [
    "update_norm",
    "grad_norm",
    "cosine_to_global",
    "cosine_to_prev",
    "cosine_grad_update"
]

print(
    df[features + ["loo_score"]]
    .corr()["loo_score"]
    .sort_values()
)

print(
    df.groupby("alpha")["loo_score"]
    .agg(["mean", "std", "min", "max"])
)
print(df["gradient"].iloc[0].shape)
print(type(df["gradient"].iloc[0]))
g = df["gradient"].iloc[0]
print(type(g))
print(len(g))