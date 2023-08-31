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
    numerical_features: list[str], categorical_features: list[str], random_state: int
) -> dict[str, sklearn.pipeline.Pipeline]:  # type: ignore
    """
    Build pipelines of transforms with final estimators for linear regression, K-Nearest Neighbors and histogram gradient boosting.

    Args:
        numerical_features (list[str]): List of numerical features.
        categorical_features (list[str]): List of categorical features.
        random_state (int): Random seed.

    Returns:
        dict[str, sklearn.pipeline.Pipeline]: Dictionary of linear regression, K-Nearest Neighbors and histogram gradient boosting pipelines.
    """
    # Models
    lr = LinearRegression(
        n_jobs=-1,
    )
    knn = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)

    hgb = HistGradientBoostingRegressor(
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
    """
    Train pipelines and compute performance metrics.

    Args:
        pipelines (dict[str, sklearn.pipeline.Pipeline]): Dictionary of machine learning pipelines.
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Test data.
        y_train (pd.Series): Training labels.
        y_test (pd.DataFrame): Test labels.

    Returns:
        tuple[dict[str, sklearn.pipeline.Pipeline], pd.DataFrame, pd.DataFrame]: Dictionary of fitted pipelines, metrics, and predictions DataFrames.
    """
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

        rmse = metrics.mean_squared_error(
            y_pred["Eolien (MW)"], y_pred[f"pred_eolien_{model_name}"], squared=False
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
                "RMSE",
                "MAE",
                "R2",
            ],
        ] = [
            rmse,
            mae,
            r2,
        ]

    return pipelines, metrics_df, y_pred


def pred_prod_2023(
    meteo_agg_df: pd.DataFrame,
    region_list: list[str],
    model: sklearn.base.RegressorMixin,
    features: list[str],
    parc_regional_2022: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute wind turbin energy predictions for 2023.

    Args:
        meteo_agg_df (pd.DataFrame): Weather report data, aggregated by region and timestamp.
        region_list (list[str]): List of regions for which to compute predictions.
        model (sklearn.base.RegressorMixin): Trained regression model.
        features (list[str]): List of features to take as input fro predictions.
        parc_regional_2022 (pd.DataFrame): Installed wind power per region at the end of 2022, to convert normalized predictions into absolute values.

    Returns:
        pd.DataFrame: DataFrame of 2023 predictions.
    """
    meteo_agg_2023_df = meteo_agg_df[
        (meteo_agg_df["Date"] >= "2023-01-01")
        & (meteo_agg_df["region (name)"].isin(region_list))
    ].rename(columns={"region (code)": "Code INSEE région"})

    pred_2023_df = meteo_agg_2023_df[["Date", "region (name)"]].copy()

    pred_2023_df["Prédictions normalisées"] = model.predict(meteo_agg_2023_df[features])  # type: ignore

    pred_2023_df = pd.merge(
        pred_2023_df,
        parc_regional_2022,
        left_on="region (name)",
        right_on="Région",
    )

    pred_2023_df["Prédiction eolien (MW)"] = (
        pred_2023_df["Prédictions normalisées"]
        * pred_2023_df["Parc installé éolien (MW)"]
    )

    return pred_2023_df
