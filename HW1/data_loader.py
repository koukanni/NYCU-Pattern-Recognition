import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

from ucimlrepo import fetch_ucirepo

import os
import joblib

CACHE_DIR = os.environ.get("UCIMLREPO_DIR", "./data_cache") 
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"Using data cache directory: {os.path.abspath(CACHE_DIR)}")

memory = joblib.Memory(CACHE_DIR, verbose=0)


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load dataset by name from the specified directory.

    Parameters:
    - name: Name of the dataset (without file extension).
    - data_dir: Directory where datasets are stored.

    Returns:
    - DataFrame containing the loaded dataset.
    """

    if name == "telescope":
        return load_telescope()
    elif name == "bankruptcy":
        return load_bankruptcy()
    elif name == "cover_type":
        return load_cover_type()
    elif name == "dry_bean":
        return load_dry_bean()
    else:
        logger.error(f"Unknown dataset name: {name}")
        raise ValueError(f"Unknown dataset name: {name}")


def _preprocess_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = X.values
    y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y.ravel())

    return X, y


@memory.cache
def load_telescope() -> pd.DataFrame:
    logger.info("Loading Telescope (id=159) from ucimlrepo...")

    telescope = fetch_ucirepo(id=159)

    X = telescope.data.features
    y = telescope.data.targets

    return _preprocess_data(X, y)


@memory.cache
def load_bankruptcy() -> pd.DataFrame:
    logger.info("Loading Bank Bankruptcy (id=572) from ucimlrepo...")

    bank_bankruptcy = fetch_ucirepo(id=572)

    X = bank_bankruptcy.data.features
    y = bank_bankruptcy.data.targets

    return _preprocess_data(X, y)


@memory.cache
def load_cover_type() -> pd.DataFrame:
    logger.info("Loading Cover Type (id=31) from ucimlrepo...")

    cover_type = fetch_ucirepo(id=31)

    X = cover_type.data.features
    y = cover_type.data.targets

    sample = np.random.choice(len(X), size=int(5E4), replace=False)
    X = X.iloc[sample]
    y = y.iloc[sample]

    return _preprocess_data(X, y)


@memory.cache
def load_dry_bean() -> pd.DataFrame:
    logger.info("Loading Dry Bean Classification (id=602) from ucimlrepo...")

    dry_bean = fetch_ucirepo(id=602)

    X = dry_bean.data.features
    y = dry_bean.data.targets

    return _preprocess_data(X, y)
