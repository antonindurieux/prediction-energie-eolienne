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
        - Then we divide the production data by the installed wind power estimation, to normalize the production.
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
    """_summary_

    Args:
        meteo_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    meteo_df["Sinus direction du vent moyen 10 mn"] = np.sin(
        meteo_df["Direction du vent moyen 10 mn"] * np.pi / 180
    )
    meteo_df["Cosinus direction du vent moyen 10 mn"] = np.cos(
        meteo_df["Direction du vent moyen 10 mn"] * np.pi / 180
    )

    return meteo_df


def compute_weather_aggregations(meteo_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        meteo_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    meteo_agg_df = (
        meteo_df.groupby(["Date", "region (name)"])
        .agg(
            {
                "Pression au niveau mer": "mean",
                "Variation de pression en 3 heures": "mean",
                "Sinus direction du vent moyen 10 mn": "mean",
                "Cosinus direction du vent moyen 10 mn": "mean",
                "Vitesse du vent moyen 10 mn": "mean",
                "Température": "mean",
                "Point de rosée": "mean",
                "Humidité": "mean",
                "Pression station": "mean",
                "Rafales sur une période": "mean",
                "Rafale sur les 10 dernières minutes": "mean",
                "Précipitations dans la dernière heure": "mean",
            }
        )
        .reset_index()
    )

    return meteo_agg_df
