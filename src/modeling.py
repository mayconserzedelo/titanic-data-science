"""Model training and evaluation helpers for the Titanic project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    confusion: list[list[int]]
    report: str


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_models(X: pd.DataFrame, y: pd.Series) -> dict[str, Pipeline]:
    """Train a couple of baseline classification models."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    pipelines: dict[str, Pipeline] = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipelines[name].fit(X, y)

    return pipelines


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    predictions = model.predict(X_test)
    return ModelResult(
        model_name=type(model.named_steps["model"]).__name__,
        accuracy=accuracy_score(y_test, predictions),
        confusion=confusion_matrix(y_test, predictions).tolist(),
        report=classification_report(y_test, predictions),
    )


def split_data(df: pd.DataFrame, target_col: str = "survived", test_size: float = 0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
