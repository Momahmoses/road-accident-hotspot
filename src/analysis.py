import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


FEATURES = [
    "road_type_code", "weather_code", "hour_of_day", "is_night",
    "is_peak_hour", "speed_limit_kmh", "estimated_speed_kmh",
    "vehicles_involved", "pedestrian_involved", "severity_score",
    "speeding", "visibility_m"
]


def compute_kde_hotspots(df, bandwidth=0.05):
    coords = df[["longitude", "latitude"]].values.T
    kde = gaussian_kde(coords, bw_method=bandwidth)
    df = df.copy()
    df["kde_density"] = kde(coords.T.T)
    df["is_hotspot"] = df["kde_density"] > df["kde_density"].quantile(0.80)
    return df


def cluster_hotspots(df, radius_km=1.0):
    hotspot_df = df[df["is_hotspot"]].copy()
    coords = np.radians(hotspot_df[["latitude", "longitude"]].values)
    tree = KDTree(coords)
    clusters = []
    visited = set()
    for i, (_, row) in enumerate(hotspot_df.iterrows()):
        if i in visited:
            continue
        neighbors = tree.query_ball_point(coords[i], r=radius_km / 6371.0)
        for j in neighbors:
            visited.add(j)
        cluster_rows = hotspot_df.iloc[neighbors]
        clusters.append({
            "cluster_id": f"HOT_{len(clusters):03d}",
            "center_lat": cluster_rows["latitude"].mean(),
            "center_lon": cluster_rows["longitude"].mean(),
            "accident_count": len(cluster_rows),
            "fatal_count": (cluster_rows["severity"] == "fatal").sum(),
            "avg_severity_score": cluster_rows["severity_score"].mean(),
            "dominant_road_type": cluster_rows["road_type"].mode()[0],
            "dominant_weather": cluster_rows["weather"].mode()[0],
            "state": cluster_rows["state"].mode()[0],
            "risk_score": cluster_rows["kde_density"].mean() * len(cluster_rows)
        })
    return pd.DataFrame(clusters).sort_values("risk_score", ascending=False)


def train_severity_model(df, config):
    X = df[FEATURES]
    y = (df["severity"] == "fatal").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"], stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"],
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return model, report


def save_model(model, path):
    joblib.dump(model, path)
