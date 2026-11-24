
"""Data loading, synthetic fallback, and leakage-safe preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET = "Class"

def make_synthetic_data(n_samples: int = 30000, fraud_rate: float = 0.0017,
                        random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a plausible imbalanced dataset with the Kaggle column schema.

    This is a development/demo substitute, not a statistically faithful copy
    of the Kaggle data and not suitable for real-world risk decisions.
    """
    n_fraud = max(20, int(n_samples * fraud_rate))
    n_normal = n_samples - n_fraud
    X, y = make_classification(
        n_samples=n_samples, n_features=28, n_informative=12,
        n_redundant=8, n_repeated=0, n_clusters_per_class=2,
        weights=[n_normal / n_samples, n_fraud / n_samples],
        class_sep=1.15, flip_y=0.002, random_state=random_state
    )
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    frame.insert(0, "Time", rng.uniform(0, 172800, n_samples))
    # Positive, right-skewed transaction amounts with a modest fraud shift.
    amounts = rng.lognormal(mean=3.0, sigma=1.25, size=n_samples)
    amounts[y == 1] *= rng.uniform(1.05, 1.8, size=(y == 1).sum())
    frame.insert(1, "Amount", amounts)
    frame["Class"] = y.astype(int)
    return frame[FEATURES + [TARGET]]

def load_data(data_path: str = "data/creditcard.csv",
              n_synthetic: int = 30000) -> tuple[pd.DataFrame, str]:
    path = Path(data_path)
    if path.exists():
        df = pd.read_csv(path)
        source = f"Kaggle CSV: {path}"
    else:
        df = make_synthetic_data(n_samples=n_synthetic)
        source = "Synthetic demo data (Kaggle file not found)"
    required = set(FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (legit) and 1 (fraud).")
    return df, source

def prepare_splits(df: pd.DataFrame, test_size: float = 0.2,
                   random_state: int = RANDOM_STATE):
    """Split first, then fit RobustScaler on train only; transform test.

    SMOTE is intentionally not performed here. It belongs only on X_train.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = RobustScaler()
    scale_cols = ["Time", "Amount"]
    X_train.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test.loc[:, scale_cols] = scaler.transform(X_test[scale_cols])
    return X_train, X_test, y_train, y_test, scaler
