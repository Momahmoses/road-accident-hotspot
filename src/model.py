"""
Road accident hotspot prediction and severity analysis model.
Binary classification of hotspot road segments.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

CAT_COLS = ["road_type", "crash_type", "weather_condition", "vehicle_type"]
NUM_FEATURES = [
    "estimated_speed_kmh", "road_quality_score", "lighting_quality",
    "n_vehicles_involved", "alcohol_suspected", "dist_to_nearest_hospital_km",
    "road_surface_wet", "visibility_m", "is_night", "speed_limit_kmh",
]
FEATURE_COLS: list[str] = []
TARGET_COL = "is_hotspot_segment"
MODEL_PATH = Path("assets/accident_model.pkl")
ENCODERS_PATH = Path("assets/encoders.pkl")
METRICS_PATH = Path("assets/metrics.json")


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoders: dict[str, LabelEncoder] = {}
    df = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return NUM_FEATURES + [f"{c}_enc" for c in CAT_COLS]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )),
    ])


def train(df: pd.DataFrame) -> tuple[Pipeline, dict[str, LabelEncoder], dict]:
    df_enc, encoders = encode_features(df)
    feat_cols = get_feature_cols(df_enc)
    X, y = df_enc[feat_cols], df_enc[TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_pipeline()
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    cv_auc = cross_val_score(pipeline, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    report = classification_report(y_te, y_pred, output_dict=True)
    metrics = {
        "roc_auc_test": round(float(roc_auc_score(y_te, y_prob)), 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "f1_hotspot": round(float(report["1"]["f1-score"]), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "feature_importance": dict(zip(feat_cols, pipeline.named_steps["clf"].feature_importances_.tolist())),
    }
    return pipeline, encoders, metrics


def save_model(pipeline: Pipeline, encoders: dict, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_model():
    return joblib.load(MODEL_PATH), joblib.load(ENCODERS_PATH)
