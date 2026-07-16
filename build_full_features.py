import numpy as np
import pandas as pd
import joblib

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# -----------------------------
# LOAD
# -----------------------------

print("Loading telemetry...")

df = pd.read_pickle("all_telemetry.pkl")

print("Dataset:")
print(df.shape)


# -----------------------------
# BUILD GRADIENT MATRIX
# -----------------------------

print("\nBuilding gradient matrix...")

G = np.stack(df["gradient"].values)

print("Gradient matrix:")
print(G.shape)


# -----------------------------
# RANDOM PROJECTION
# -----------------------------

print("\nRandom projection...")


projector = GaussianRandomProjection(
    n_components=256,
    random_state=42
)

G_proj = projector.fit_transform(G)


print("Projected gradients:")
print(G_proj.shape)



# normalize for cosine operations

G_norm = normalize(G_proj)



# -----------------------------
# PCA ON PROJECTED GRADIENTS
# -----------------------------

print("\nRunning PCA...")


pca = PCA(
    n_components=10,
    random_state=42
)

P = pca.fit_transform(G_proj)


print("PCA shape:")
print(P.shape)

print(
"Explained variance:",
pca.explained_variance_ratio_.sum()
)



# -----------------------------
# ADD FEATURES
# -----------------------------


for i in range(P.shape[1]):
    df[f"pca_{i}"] = P[:,i]



# -----------------------------
# TEMPORAL FEATURES
# -----------------------------

print("\nComputing temporal gradient features...")


df = df.sort_values(
    ["seed","alpha","client_id","round"]
)


# cosine with previous gradient of same client

df["gradient_progress_cosine"] = np.nan

for _, group in df.groupby(
    ["seed","alpha","client_id"]
):

    idx = group.index

    vectors = G_norm[idx]

    if len(idx) > 1:

        sims = np.sum(
            vectors[1:] * vectors[:-1],
            axis=1
        )

        df.loc[idx[1:], "gradient_progress_cosine"] = sims



# -----------------------------
# CLUSTERING
# -----------------------------

print("\nRunning clustering...")


kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)


df["gradient_cluster"] = kmeans.fit_predict(
    G_proj
)



# -----------------------------
# SAVE
# -----------------------------


# Remove huge raw gradient vectors
df = df.drop(columns=["gradient"])

joblib.dump(
    projector,
    "gradient_projector.pkl"
)

joblib.dump(
    pca,
    "gradient_pca.pkl"
)

joblib.dump(
    kmeans,
    "gradient_kmeans.pkl"
)

df.to_parquet(
    "full_gradient_features.parquet"
)


print("\nDone!")
print(df.shape)

print(
df.columns.tolist()
)