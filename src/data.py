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
        get_whole_dataset(dataset, platform_id=platform_id, to_path=file_path)  # type: ignore


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, save into pickles and return DataFrames of power production, installed wind power and weather reports.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames of power production, installed wind power and weather reports.
    """
    pickle_path = DATA_PATH / "pickles"
    pickle_path.mkdir(parents=False, exist_ok=True)

    # Production
    print("Loading production data...")

    if os.path.isfile(pickle_path / "production_df.pkl"):
        production_df = pd.read_pickle(pickle_path / "production_df.pkl")

    else:
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
        production_df = (
            production_df.dropna(axis=1, how="all")
            .drop(columns=["Date", "Heure"])
            .rename(columns={"Date - Heure": "Date"})
            .sort_values(["Date", "Région"])
            .reset_index(drop=True)
        )
        production_df.to_pickle(pickle_path / "production_df.pkl")

    # Installed wind power
    print("Loading installed wind power data...")
    parc_regional_df = pd.read_csv(
        DATA_PATH / "parc-regional-annuel-prod-eolien-solaire.csv",
        sep=";",
        dtype={"Code INSEE région": "category", "Région": "category"},
    )
    parc_regional_df["Date"] = parc_regional_df["Année"].apply(
        lambda x: pd.Timestamp(f"12-31-{x}")
    )
    parc_regional_df["Date"] = parc_regional_df["Date"].dt.tz_localize("UTC")
    parc_regional_df = parc_regional_df.sort_values(["Date", "Région"]).reset_index(
        drop=True
    )

    # Weather reports
    print("Loading weather reports data...")

    if os.path.isfile(pickle_path / "meteo_df.pkl"):
        meteo_df = pd.read_pickle(pickle_path / "meteo_df.pkl")

    else:
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
        meteo_df = meteo_df.sort_values(["Date", "ID OMM station"]).reset_index(
            drop=True
        )
        meteo_df.to_pickle(pickle_path / "meteo_df.pkl")

    return production_df, parc_regional_df, meteo_df
