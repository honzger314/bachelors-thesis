import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load
# -----------------------------

from pathlib import Path

ROOT = Path("results2") / "results"

files = list(ROOT.rglob("telemetry.parquet"))

dfs = []

for f in files:

    tmp = pd.read_parquet(f)

    parts = f.parent.name.split("_")

    tmp["seed"] = int(parts[3])
    tmp["alpha"] = float(parts[5])

    dfs.append(tmp)


df = pd.concat(
    dfs,
    ignore_index=True
)

print(df.shape)

print(df.shape)


# -----------------------------
# Reconstruct gradient matrix
# -----------------------------

print("Building gradient matrix")

from sklearn.random_projection import GaussianRandomProjection


print("Building random projection")

# fit projector using one gradient size
dim = len(df["gradient"].iloc[0])

projector = GaussianRandomProjection(
    n_components=256,
    random_state=42
)

# sklearn needs data, so process in chunks

projected = []

batch_size = 200

for i in range(0, len(df), batch_size):

    print(
        f"Projecting {i}/{len(df)}"
    )

    batch = np.stack(
        df["gradient"].iloc[i:i+batch_size]
    ).astype(np.float32)

    if i == 0:
        projector.fit(batch)

    projected.append(
        projector.transform(batch)
    )


G = np.vstack(projected)

print("Projected gradients:")
print(G.shape)

print(G.shape)


# normalize gradients
G_norm = G / (
    np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
)


# -----------------------------
# Prepare storage
# -----------------------------

history_alignment = []
direction_change = []
global_history_alignment = []
client_gradient_norm_change = []


# -----------------------------
# Process each run separately
# -----------------------------

group_cols = [
    "seed",
    "alpha"
]


for run, run_df in df.groupby(group_cols):

    print("Processing", run)

    indices = run_df.index.to_numpy()

    # order by round/client
    run_df = run_df.sort_values(
        ["client_id", "round"]
    )

    run_indices = run_df.index.to_numpy()

    run_G = G_norm[run_indices]


    # global history
    global_mean_history = []


    for client, client_df in run_df.groupby("client_id"):

        client_indices = client_df.index.to_numpy()

        client_G = G_norm[client_indices]


        previous = []
        client_history = []


        for k, idx in enumerate(client_indices):

            current = client_G[k]


            # first round has no history
            if len(previous) == 0:

                history_alignment.append(np.nan)
                direction_change.append(np.nan)
                client_gradient_norm_change.append(np.nan)

            else:

                hist = np.mean(
                    previous,
                    axis=0
                )

                hist /= (
                    np.linalg.norm(hist)+1e-12
                )


                history_alignment.append(
                    np.dot(current,hist)
                )


                direction_change.append(
                    1 - np.dot(
                        current,
                        previous[-1]
                    )
                )


                client_gradient_norm_change.append(
                    np.linalg.norm(current-previous[-1])
                )


            previous.append(current)


        # global trajectory
        for idx in client_indices:

            current = G_norm[idx]

            if len(global_mean_history)==0:

                global_history_alignment.append(
                    np.nan
                )

            else:

                gh = np.mean(
                    global_mean_history,
                    axis=0
                )

                gh /= (
                    np.linalg.norm(gh)+1e-12
                )

                global_history_alignment.append(
                    np.dot(current,gh)
                )


            global_mean_history.append(
                G_norm[idx]
            )



# -----------------------------
# Add features
# -----------------------------

# Careful: order is client grouping order,
# so create temporary dataframe

temporal = pd.DataFrame(
    {
        "index": df.sort_values(
            ["seed","alpha","client_id","round"]
        ).index,

        "history_alignment":
            history_alignment,

        "direction_change":
            direction_change,

        "global_history_alignment":
            global_history_alignment,

        "gradient_change":
            client_gradient_norm_change
    }
)


temporal = temporal.set_index("index")


for col in temporal.columns:

    df[col] = temporal[col]


# -----------------------------
# Save
# -----------------------------

# Remove huge raw gradient vectors before saving
if "gradient" in df.columns:
    df = df.drop(columns=["gradient"])

print("Final dataset:")
print(df.shape)

print(df.columns)

print("Done")
print(df.columns)

output = "temporal_gradient_features.parquet"

df.to_parquet(
    output,
    index=False
)

print("Saved:", output)