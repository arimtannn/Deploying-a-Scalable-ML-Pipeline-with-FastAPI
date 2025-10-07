import pytest
import os
import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import (train_model, inference, compute_model_metrics)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

LABEL = "salary"


@pytest.fixture(scope="module")
def small_df():
    """Load a small, representative slice of the census data for quick tests."""
    root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root, "data", "census.csv")
    df = pd.read_csv(csv_path)
    # Clean minor whitespace in object columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    # Sample for speed but keep enough variety for categories
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42)
    return df


def test_process_data_outputs(small_df):
    """
    process_data returns arrays with correct shape and fitted encoder/lb
    """
    X, y, encoder, lb = process_data(
        small_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 1
    assert encoder is not None and lb is not None
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 10  # one-hot should expand features


def test_train_and_inference_shapes(small_df):
    """
    Model trains and predicts binary labels with correct length.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        small_df,
        test_size=0.2,
        random_state=42,
        stratify=small_df[LABEL]
    )
    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    """
    Deterministic check:
        y = [0, 1, 1, 0], preds = [0, 1, 0, 0]
        TP=1, FP=0, FN=1 => Precision=1.0, Recall=0.5, F1=2/3
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    p, r, f1 = compute_model_metrics(y, preds)
    assert pytest.approx(p, rel=1e-6) == 1.0
    assert pytest.approx(r, rel=1e-6) == 0.5
    assert pytest.approx(f1, rel=1e-6) == 2 / 3
