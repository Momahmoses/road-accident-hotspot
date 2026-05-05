import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_accident_dataset(n=5000, seed=42):
    np.random.seed(seed)
    states = ["Lagos", "Abuja", "Ogun", "Oyo", "Rivers", "Kano", "Kaduna"]
    road_types = ["highway", "trunk", "primary", "secondary", "urban_road"]
    weather = ["dry", "wet", "foggy"]
    severity = ["fatal", "serious", "minor"]
    sev_weights = [0.15, 0.30, 0.55]

    hour = np.random.choice(range(24), n)
    data = {
        "accident_id": [f"ACC_{i:05d}" for i in range(n)],
        "latitude": np.random.uniform(4.0, 13.5, n),
        "longitude": np.random.uniform(3.0, 14.5, n),
        "state": np.random.choice(states, n),
        "road_type": np.random.choice(road_types, n, p=[0.2, 0.15, 0.3, 0.25, 0.1]),
        "severity": np.random.choice(severity, n, p=sev_weights),
        "weather": np.random.choice(weather, n, p=[0.6, 0.35, 0.05]),
        "hour_of_day": hour,
        "is_night": (hour < 6) | (hour >= 20),
        "is_peak_hour": ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19)),
        "speed_limit_kmh": np.random.choice([60, 80, 100, 120], n, p=[0.3, 0.35, 0.25, 0.1]),
        "estimated_speed_kmh": np.random.normal(85, 25, n).clip(20, 160),
        "vehicles_involved": np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.45, 0.15, 0.1]),
        "pedestrian_involved": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "road_surface_condition": np.random.choice(["dry", "wet", "potholed"], n, p=[0.5, 0.3, 0.2]),
        "visibility_m": np.random.exponential(scale=800, size=n).clip(50, 2000),
        "fatalities": np.random.poisson(lam=0.2, size=n),
        "injuries": np.random.poisson(lam=1.5, size=n),
        "year": np.random.choice(range(2018, 2024), n),
        "month": np.random.choice(range(1, 13), n),
    }
    df = pd.DataFrame(data)
    df["road_type_code"] = pd.Categorical(df["road_type"]).codes
    df["weather_code"] = pd.Categorical(df["weather"]).codes
    df["severity_score"] = df["severity"].map({"fatal": 3, "serious": 2, "minor": 1})
    df["speeding"] = (df["estimated_speed_kmh"] > df["speed_limit_kmh"]).astype(int)
    return df


def to_geodataframe(df):
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )
