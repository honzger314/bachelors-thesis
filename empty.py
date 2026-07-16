import pandas as pd

df = pd.read_parquet("temporal_gradient_features.parquet")

print(df.shape)
print(df.isna().sum())
print(df[["seed","alpha","round","client_id"]].head())