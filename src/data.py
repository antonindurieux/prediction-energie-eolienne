import os
from pathlib import Path

import pandas as pd
from odsclient import get_whole_dataset

DATA_PATH = Path("data/")
DATASETS = [
    "eco2mix-regional-cons-def",
    "parc-regional-annuel-prod-eolien-solaire",
    "donnees-synop-essentielles-omm",
]


def download_data() -> None:
    """
    Download datasets and save them in the data path.
    """
    DATA_PATH.mkdir(parents=False, exist_ok=True)

    for dataset in DATASETS:
        file_name = f"{dataset}.csv"
        file_path = DATA_PATH / file_name
        platform_id = "odre"
        if dataset == "donnees-synop-essentielles-omm":
            platform_id = "public"

        if file_path.is_file():
            if os.stat(file_path).st_size > 0:
                print(f"{dataset} is already downloaded.")
                continue

        print(f"Downloading {dataset}...")
        get_whole_dataset(dataset, platform_id=platform_id, tqdm=True, to_path=file_path)  # type: ignore


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and returns DataFrames of power production, installed wind power and weather reports.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames of power production, installed wind power and weather reports.
    """
    production_df = pd.read_csv(
        DATA_PATH / "eco2mix-regional-cons-def.csv",
        sep=";",
        parse_dates=[5],
        dtype={
            "Code INSEE région": "category",
            "Région": "category",
            "Nature": "category",
        },
    )
    production_df = production_df.dropna(axis=1, how="all")
    production_df = production_df.drop(columns=["Date", "Heure"])
    production_df = production_df.rename(columns={"Date - Heure": "Date"})
    production_df = production_df.sort_values(
        ["Date", "Code INSEE région"]
    ).reset_index()

    parc_regional_df = pd.read_csv(
        DATA_PATH / "parc-regional-annuel-prod-eolien-solaire.csv",
        sep=";",
        dtype={"Code INSEE région": "category", "Région": "category"},
    )
    parc_regional_df["Date"] = parc_regional_df["Année"].apply(
        lambda x: pd.Timestamp(f"12-31-{x}")
    )
    parc_regional_df = parc_regional_df.sort_values(["Date", "Code INSEE région"])

    meteo_df = pd.read_csv(
        "data/donnees-synop-essentielles-omm.csv",
        sep=";",
        parse_dates=[1],
        dtype={
            "region (name)": "category",
            "department (name)": "category",
            "region (code)": "category",
            "department (code)": "category",
            "communes (name)": "category",
            "communes (code)": "category",
            "ID OMM station": "category",
            "EPCI (name)": "category",
            "EPCI (code)": "category",
        },
    )
    meteo_df = meteo_df.sort_values(["Date", "ID OMM station"])

    return production_df, parc_regional_df, meteo_df


def filter_date_range(
    production_df: pd.DataFrame, parc_regional_df: pd.DataFrame, meteo_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter data according to the beginning of wind power production data.

    Args:
        production_df (pd.DataFrame): DataFrame of power production.
        parc_regional_df (pd.DataFrame): DataFrame of installed wind power.
        meteo_df (pd.DataFrame): DataFrame of weather reports.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames of power production, installed wind power and weather reports.
    """
    date_range_beg = production_df[production_df["Eolien (MW)"].notna()][
        "Date - Heure"
    ].min()
    production_df = production_df[production_df["Date - Heure"] >= date_range_beg]

    parc_regional_df = parc_regional_df[
        parc_regional_df["Année"] >= date_range_beg.year
    ]

    meteo_df[meteo_df["Date"] >= date_range_beg]["Date"]

    return production_df, parc_regional_df, meteo_df
