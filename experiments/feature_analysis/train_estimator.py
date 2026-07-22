import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# -------------------------
# LOAD
# -------------------------

df = pd.read_parquet(
    "full_gradient_features.parquet"
)


print(df.shape)


# -------------------------
# SPLIT BY SEED
# -------------------------

train = df[df.seed != 6]
test = df[df.seed == 6]


print("Train:", train.shape)
print("Test:", test.shape)


y_train = train["loo_score"]
y_test = test["loo_score"]



# -------------------------
# FEATURE GROUPS
# -------------------------


baseline = [
    "update_norm",
    "grad_norm",
    "cosine_to_global",
    "cosine_to_prev",
    "cosine_grad_update"
]


gradient_features = [
    f"pca_{i}" for i in range(10)
] + [
    "gradient_progress_cosine",
    "gradient_cluster"
]


all_features = baseline + gradient_features



experiments = {
    "baseline": baseline,
    "gradient": gradient_features,
    "combined": all_features
}



# -------------------------
# MODELS
# -------------------------
from sklearn.impute import SimpleImputer


models = {

    "ridge":
    make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        Ridge(alpha=1)
    ),


    "random_forest":
    make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
}


# -------------------------
# RUN
# -------------------------

for feat_name, features in experiments.items():

    print("\n================")
    print(feat_name)
    print("================")


    X_train = train[features]
    X_test = test[features]


    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )


        pred = model.predict(
            X_test
        )


        mse = mean_squared_error(
            y_test,
            pred
        )


        rho,_ = spearmanr(
            y_test,
            pred
        )


        print(
            model_name,
            "MSE:",
            mse,
            "Spearman:",
            rho
        )