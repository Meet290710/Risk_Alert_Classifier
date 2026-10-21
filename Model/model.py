"""
Risk Alert Classifier - model training and evaluation utilities.
Target column: risk_status (0 = Low Risk, 1 = High Risk).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

TARGET = "risk_status"
DROP_COLUMNS = ["customer_id"]

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and derive date features without using the target."""
    df = data.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "last_transaction_date" in df.columns:
        dt = pd.to_datetime(df["last_transaction_date"], errors="coerce")
        df["last_transaction_year"] = dt.dt.year
        df["last_transaction_month"] = dt.dt.month
        df["days_since_last_transaction"] = (
            pd.Timestamp.today().normalize() - dt.dt.normalize()
        ).dt.days
        df = df.drop(columns=["last_transaction_date"])
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    return df

def load_dataset(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV file."""
    return pd.read_csv(uploaded_file)

def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = SkPipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")

def _make_sampler(name: str):
    samplers = {
        "None": None,
        "Random Under-Sampling": RandomUnderSampler(random_state=42),
        "Random Over-Sampling": RandomOverSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42, k_neighbors=3),
        "ADASYN": ADASYN(random_state=42, n_neighbors=3),
    }
    if name not in samplers:
        raise ValueError(f"Unknown sampling method: {name}")
    return samplers[name]

def _make_estimator(model_name: str, random_state: int = 42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=None, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight=None),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=random_state, n_jobs=-1,
            class_weight=None, min_samples_leaf=2
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]

def train_and_evaluate(
    data: pd.DataFrame,
    model_name: str = "Random Forest",
    sampling_method: str = "SMOTE",
    tune: bool = False,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """
    Train/test split, preprocessing, optional resampling and optional randomized tuning.
    Returns the fitted pipeline and evaluation results.
    """
    df = data.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")
    df = df.dropna(subset=[TARGET])
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.isin([0, 1])
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise ValueError("The target column must contain both classes 0 and 1.")
    X = prepare_features(df.drop(columns=[TARGET]))
    if X.empty:
        raise ValueError("No usable feature columns were found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    preprocessor = _make_preprocessor(X_train)
    sampler = _make_sampler(sampling_method)
    estimator = _make_estimator(model_name)
    steps = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", estimator))
    pipeline = ImbPipeline(steps=steps)

    best_params = {}
    if tune:
        if model_name == "Decision Tree":
            params = {
                "classifier__max_depth": [None, 4, 6, 10, 16, 24],
                "classifier__min_samples_split": [2, 5, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__criterion": ["gini", "entropy", "log_loss"],
            }
        elif model_name == "Random Forest":
            params = {
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [None, 8, 14, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2"],
            }
        else:
            params = {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 50.0],
                "classifier__solver": ["lbfgs"],
                "classifier__class_weight": [None, "balanced"],
            }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            pipeline, param_distributions=params, n_iter=8,
            scoring="recall", cv=cv, random_state=42, n_jobs=-1,
            refit=True, error_score="raise"
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    try:
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        probabilities = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, probabilities)) if probabilities is not None else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Low Risk", "High Risk"], zero_division=0, output_dict=True
        ),
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "best_params": best_params,
        "train_class_counts": y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "model_name": model_name,
        "sampling_method": sampling_method,
        "feature_columns": X.columns.tolist(),
    }
    if probabilities is not None:
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        metrics["roc_curve"] = (fpr, tpr, thresholds)
    else:
        metrics["roc_curve"] = None
    return {"pipeline": pipeline, "metrics": metrics, "X_test": X_test}
