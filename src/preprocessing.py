"""Data loading, cleaning, and feature engineering helpers for the Titanic project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a Titanic CSV file."""
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps.

    Expected steps:
    - standardize column names
    - fill missing values where appropriate
    - drop columns with very high missingness if needed
    """
    data = df.copy()
    data.columns = [col.strip().lower() for col in data.columns]
    return data


def extract_title(name: str) -> str:
    """Extract passenger title from the name column."""
    if "," in name and "." in name:
        return name.split(",", 1)[1].split(".", 1)[0].strip()
    return "Unknown"


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create a few simple features commonly used in Titanic notebooks."""
    data = df.copy()

    if "name" in data.columns:
        data["title"] = data["name"].astype(str).apply(extract_title)

    if {"sibsp", "parch"}.issubset(data.columns):
        data["family_size"] = data["sibsp"].fillna(0) + data["parch"].fillna(0) + 1

    if "age" in data.columns:
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 12, 18, 35, 60, 120],
            labels=["child", "teen", "young_adult", "adult", "senior"],
            include_lowest=True,
        )

    return data


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full preparation pipeline for the Titanic dataset."""
    return feature_engineering(basic_cleaning(df))
