import os
from pathlib import Path

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
