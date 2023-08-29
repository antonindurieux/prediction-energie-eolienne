import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ml_pipelines(
    numerical_features, categorical_features, random_state
) -> dict[str, sklearn.pipeline.Pipeline]:  # type: ignore
    # Models
    lr = LinearRegression(
        n_jobs=-1,
    )
    knn = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)

    hgb = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=1000,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    # Column transformer
    ct = ColumnTransformer(
        [
            (
                "scaler",
                StandardScaler(),
                numerical_features,
            ),
            (
                "onehot",
                OneHotEncoder(),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    # Linear regression pipeline
    lr_pipeline = Pipeline(
        steps=[
            ("col_trans", ct),
            (
                "model",
                lr,
            ),
        ]
    )

    # KNN pipeline
    knn_pipeline = Pipeline(
        steps=[
            ("col_trans", ct),
            (
                "model",
                knn,
            ),
        ]
    )

    # Histogram gradient boosting pipeline
    hgb_pipeline = Pipeline(
        steps=[
            (
                "model",
                hgb,
            ),
        ]
    )

    return {
        "linear regression": lr_pipeline,
        "KNN": knn_pipeline,
        "histogram gradient boosting": hgb_pipeline,
    }


def train_and_evaluate_models(
    pipelines: dict[str, sklearn.pipeline.Pipeline],  # type: ignore
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.DataFrame,
) -> tuple[dict[str, sklearn.pipeline.Pipeline], pd.DataFrame, pd.DataFrame]:  # type: ignore
    metrics_df = pd.DataFrame()
    target = y_train.name
    y_pred = y_test.copy()

    for model_name, pipeline in pipelines.items():
        print(f"Training {model_name}...")

        pipeline.fit(X_train, y_train)

        y_pred[f"pred_eolien-normalisé_{model_name}"] = pipeline.predict(X_test)
        y_pred[f"pred_eolien_{model_name}"] = (
            y_pred[f"pred_eolien-normalisé_{model_name}"]
            * y_pred[f"Parc installé éolien (MW)"]
        )

        y_pred[f"residus_eolien-normalisé_{model_name}"] = (
            y_pred[target] - y_pred[f"pred_eolien-normalisé_{model_name}"]
        )
        y_pred[f"residus_eolien_{model_name}"] = (
            y_pred["Eolien (MW)"] - y_pred[f"pred_eolien_{model_name}"]
        )

        mae = metrics.mean_absolute_error(
            y_pred["Eolien (MW)"],
            y_pred[f"pred_eolien_{model_name}"],
        )
        r2 = metrics.r2_score(
            y_pred["Eolien (MW)"],
            y_pred[f"pred_eolien_{model_name}"],
        )

        metrics_df.loc[
            model_name,
            [
                "MAE",
                "R2",
            ],
        ] = [
            mae,
            r2,
        ]

    return pipelines, metrics_df, y_pred