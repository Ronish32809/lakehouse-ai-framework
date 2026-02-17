# imports
import os
import json
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# -------------------------
# Basic Settings
# -------------------------
TARGET_COL = "Target"
MODEL_FILE = "models/model.pkl"
METRICS_FILE = "models/metrics.json"

# Only keep the important columns (must match your dataset column names)
FEATURE_WHITELIST = [
    "pH",
    "Iron",
    "Nitrate",
    "Chloride",
    "Lead",
    "Turbidity",
    "Sulfate",
    "Total Dissolved Solids",
]


# -------------------------
# Dataset Helpers
# -------------------------
def _find_dataset_file() -> str:
    """
    Looks for the first CSV in this folder that has the target column.
    This helps avoid hardcoding filenames.
    """
    csvs = sorted(glob.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No .csv file found in the current folder.")

    # Quick scan first (fast)
    for f in csvs:
        try:
            df_head = pd.read_csv(f, nrows=5)
            if TARGET_COL in df_head.columns:
                return f
        except Exception:
            pass

    # Full scan fallback (for messy CSVs)
    for f in csvs:
        try:
            df = pd.read_csv(f)
            if TARGET_COL in df.columns:
                return f
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not find a CSV containing '{TARGET_COL}'. Found: {csvs}"
    )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - replace weird placeholders
    - remove inf values
    - try turning numeric columns into numbers
    """
    df = df.replace("########", np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Try numeric conversion. If a column is truly text, it stays as-is.
    # (Pandas warns that errors='ignore' may change in future versions, but it's OK for now.)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def prepare_data():
    """
    Loads the dataset, selects valid features, and returns:
    X (features), y (target), dataset filename
    """
    data_file = _find_dataset_file()
    df = pd.read_csv(data_file)
    df = _clean_dataframe(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing '{TARGET_COL}' in {data_file}")

    # y is the label we want to predict (0/1)
    y = df[TARGET_COL].astype(int)

    # X is everything except the label
    X = df.drop(columns=[TARGET_COL, "Index"], errors="ignore")

    # Keep only expected columns that actually exist in this dataset
    available = [c for c in FEATURE_WHITELIST if c in X.columns]
    if len(available) < 2:
        raise ValueError(
            "Dataset columns do not match the expected features.\n"
            f"Expected (any of): {FEATURE_WHITELIST}\n"
            f"Found: {list(X.columns)}"
        )

    X = X[available]

    # Convert to numeric (models require numbers)
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, data_file


# -------------------------
# Model Setup
# -------------------------
def build_models():
    """
    A small set of common ML models.
    We will try each one and keep the best accuracy.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_and_save():
    """
    Trains all models, picks the best, then saves:
    - model.pkl (best trained pipeline)
    - metrics.json (accuracy, model name, features list, etc.)
    """
    X, y, data_file = prepare_data()

    # Split dataset: train (80%) / test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()

    best_model = None
    best_accuracy = -1.0
    best_name = None
    results = {}

    for name, model in models.items():
        # Pipeline: fill missing values -> scale -> model
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)

        # Evaluate on the test set
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results[name] = float(acc)

        if acc > best_accuracy:
            best_accuracy = float(acc)
            best_model = pipe
            best_name = name

    # Train best pipeline again using ALL data (better final model)
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_FILE)

    # Save metrics (these are what we show in API response)
    metrics = {
        "dataset_file": data_file,
        "best_model": best_name,
        "best_model_accuracy": float(best_accuracy),
        "all_accuracies": results,
        "features": list(X.columns),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# -------------------------
# Used by FastAPI
# -------------------------
def ensure_trained():
    """
    If model and metrics exist, load metrics.
    If not, train once and create them.
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)

    return train_and_save()


def _normalize_keys(features: dict) -> dict:
    """
    Makes input more tolerant (helps UI/n8n).
    Examples:
    - pH vs ph vs PH
    - Total Dissolved Solids vs total_dissolved_solids
    """
    out = {}
    for k, v in (features or {}).items():
        key = str(k).strip()

        variants = {
            key,
            key.lower(),
            key.upper(),
            key.replace(" ", "_"),
            key.replace("_", " "),
            key.replace(" ", "_").lower(),
            key.replace("_", " ").lower(),
        }

        for kk in variants:
            out[kk] = v

    return out


def predict_one(features_dict: dict):
    """
    Predicts Safe / Not Safe for ONE input row.
    Returns a JSON-friendly dictionary.
    """
    metrics = ensure_trained()
    model = joblib.load(MODEL_FILE)

    cols = metrics["features"]
    incoming = _normalize_keys(features_dict)

    # Build a single-row dataframe in the correct order
    row = {}
    for c in cols:
        v = None
        for k_try in [c, c.lower(), c.upper(), c.replace(" ", "_"), c.replace("_", " ")]:
            if k_try in incoming:
                v = incoming[k_try]
                break

        try:
            row[c] = float(v) if v is not None else np.nan
        except Exception:
            row[c] = np.nan

    X = pd.DataFrame([row], columns=cols)

    # Probability for class "1"
    if hasattr(model, "predict_proba"):
        proba_safe = float(model.predict_proba(X)[0][1])
    else:
        pred_label = int(model.predict(X)[0])
        proba_safe = 1.0 if pred_label == 1 else 0.0

    pred = int(proba_safe >= 0.5)

    # --- Load training accuracy info from metrics.json ---
    best_acc = metrics.get("best_model_accuracy")
    all_acc = metrics.get("all_accuracies")
    best_model_name = metrics.get("best_model")

    return {
        "prediction": pred,
        "safe": bool(pred == 1),
        "probability_safe": proba_safe,
        "probability_not_safe": float(1.0 - proba_safe),

        "model_used": best_model_name,
        "model_accuracy": float(best_acc) if best_acc is not None else None,
        "model_accuracy_percent": round(float(best_acc) * 100, 2) if best_acc is not None else None,

        "all_accuracies": all_acc,
        "best_model": best_model_name,
        "best_model_accuracy": float(best_acc) if best_acc is not None else None,

        "features_used": cols,
        "message": f"Prediction computed ({'SAFE' if pred == 1 else 'UNSAFE'}).",
    }


if __name__ == "__main__":
    m = train_and_save()
    print("Trained OK:", m["best_model"], m["best_model_accuracy"])
    print("Features:", m["features"])