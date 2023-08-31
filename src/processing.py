from typing import Callable, Union

import numpy as np
import pandas as pd


def normalize_production(
    production_df: pd.DataFrame, parc_regional_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalize wind power production by the installed wind power across region and time.
    To do that we:
        - Compute a monthly interpolation of installed wind power capacities for each region, from yearly data.
        - Merge these monthly interpolations data into the production DataFrame,
          so as for each timestamp and region we get the nearest monthly estimated installed wind power.
        - Then we divide the production data by the installed wind power estimation,
          to normalize the production according to the estimated installed power capacities.

    Args:
        production_df (pd.DataFrame): DataFrame of power production.
        parc_regional_df (pd.DataFrame): DataFrame of installed wind power capacities.

    Returns:
        pd.DataFrame: DataFrame of power production with normalized production.
    """
    # Compute monthly interpolation of installed wind power capacities
    montly_interp_parc_regional_df = (
        parc_regional_df.set_index("Date")
        .groupby("Région")["Parc installé éolien (MW)"]
        .apply(lambda group: group.resample("M").interpolate(method="linear"))
        .reset_index()
        .sort_values(["Date", "Région"])
    )

    # Merge with production data on nearest monthly installed wind power
    production_df = pd.merge_asof(
        production_df,
        montly_interp_parc_regional_df,
        on="Date",
        by="Région",
        direction="nearest",
    )

    production_df["Eolien normalisé (%)"] = (
        production_df["Eolien (MW)"] / production_df["Parc installé éolien (MW)"]
    )

    return production_df


def compute_wind_direction_features(meteo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the sinus and cosinus of the wind direction, in order to take the cyclicity of angular data into account.

    Args:
        meteo_df (pd.DataFrame): Weather reports DataFrame.

    Returns:
        pd.DataFrame: Weather reports DataFrame.
    """
    meteo_df["Sinus direction du vent moyen 10 mn"] = np.sin(
        meteo_df["Direction du vent moyen 10 mn"] * np.pi / 180
    )
    meteo_df["Cosinus direction du vent moyen 10 mn"] = np.cos(
        meteo_df["Direction du vent moyen 10 mn"] * np.pi / 180
    )

    return meteo_df


def compute_weather_aggregations(
    meteo_df: pd.DataFrame,
    aggregation_dict: dict[str, Union[str, list[Union[str, Callable]], Callable]],
) -> pd.DataFrame:
    """
    Compute the aggregation defined in aggregation_dict, for each region and timestamp.

    Args:
        meteo_df (pd.DataFrame): Weather reports DataFrame.

    Returns:
        pd.DataFrame: Aggregated weather reports DataFrame.
    """
    aggregation_dict["region (code)"] = "first"
    meteo_agg_df = (
        meteo_df.groupby(["Date", "region (name)"]).agg(aggregation_dict).reset_index()
    )

    meteo_agg_df.columns = meteo_agg_df.columns.map("-".join)
    meteo_agg_df = meteo_agg_df.rename(
        columns={
            "Date-": "Date",
            "region (name)-": "region (name)",
            "region (code)-first": "region (code)",
        }
    )

    return meteo_agg_df


def handle_nan_values(
    production_meteo_df: pd.DataFrame,
    mean_features: list[str],
    std_features: list[str],
    categorical_features: list[str],
    target: str,
) -> pd.DataFrame:
    """
    Handle nan values: we suppress rows with any nan in the feature and target columns,
    except for standard deviation features where we encode them by -1 as they correspond to regions with a single weather station.

    Args:
        production_meteo_df (pd.DataFrame): Merged DataFrames of power production and weather reports.
        mean_features (list[str]): Mean average aggregation features.
        std_features (list[str]): Standard deviation aggregation features.
        categorical_features (list[str]): Categorical features.
        target (str): Target.

    Returns:
        pd.DataFrame: DataFrame without any nans for features and target.
    """
    init_len = len(production_meteo_df)
    production_meteo_df = production_meteo_df[
        production_meteo_df[mean_features + categorical_features + [target]]
        .notna()
        .all(axis=1)
    ].copy()
    filtered_len = len(production_meteo_df)
    diff_len = init_len - filtered_len
    print(
        f"Suppression des NaNs: {diff_len} lignes supprimées ({diff_len / init_len:.2f}%)"
    )

    production_meteo_df[std_features] = production_meteo_df[std_features].fillna(-1)

    return production_meteo_df
