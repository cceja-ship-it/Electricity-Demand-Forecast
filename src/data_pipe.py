import os
import subprocess
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
DATASET_SLUG = "robikscube/hourly-energy-consumption"


def download_dataset(dest_dir: Path = RAW_DIR) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    if any(dest_dir.glob("*.csv")):
        print(f"Data already present in {dest_dir}, skipping download.")
        return

    print(f"Downloading {DATASET_SLUG} from Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(dest_dir),
            "--unzip",
        ],
        check=True,
    )
    print(f"Done. Files saved to {dest_dir}")


def load_region(region: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    filepath = raw_dir / f"{region}_hourly.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Run download_dataset() first, or check "
            f"the region name against the files in {raw_dir}."
        )

    df = pd.read_csv(filepath, parse_dates=["Datetime"])
    df = df.rename(columns={"Datetime": "datetime", df.columns[1]: "load_mw"})
    df = df.set_index("datetime").sort_index()

    # The raw files can contain duplicate timestamps, so exact duplicate rows are dropped here
    # and DST adjustment happens in preprocessing.
    df = df[~df.index.duplicated(keep="first")]

    return df


if __name__ == "__main__":
    download_dataset()
    df = load_region("DAYTON") 
    print(df.head())
    print(f"\nRows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
