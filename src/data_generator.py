"""
Synthetic dataset generator for road accident hotspot analysis.
Each record is an accident event with GPS coordinates, road, weather, and vehicle features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_ACCIDENTS = 8000

STATES = [
    ("Lagos", 6.52, 3.38), ("Abuja", 9.07, 7.40), ("Ogun", 7.00, 3.35),
    ("Rivers", 4.82, 7.05), ("Kano", 12.00, 8.52), ("Kaduna", 10.52, 7.44),
    ("Enugu", 6.45, 7.51), ("Borno", 11.83, 13.15), ("Ondo", 7.10, 5.05),
    ("Oyo", 8.00, 4.00),
]

ROAD_TYPES = ["Expressway", "Federal Highway", "State Road", "Urban Arterial", "Rural Road"]
CRASH_TYPES = ["Head-On", "Rear-End", "Sideswipe", "Rollover", "Pedestrian", "Hit-and-Run"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Harmattan Dust", "Night-Dry", "Foggy"]
VEHICLE_TYPES = ["Passenger Car", "Commercial Bus", "Articulated Truck", "Motorcycle (Okada)", "Tricycle (Keke)"]


def generate_accident_dataset(n_accidents: int = N_ACCIDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for acc_id in range(n_accidents):
        state, lat, lon = STATES[rng.integers(0, len(STATES))]
        road_type = ROAD_TYPES[rng.integers(0, len(ROAD_TYPES))]
        crash_type = CRASH_TYPES[rng.integers(0, len(CRASH_TYPES))]
        weather = WEATHER_CONDITIONS[rng.integers(0, len(WEATHER_CONDITIONS))]
        vehicle_type = VEHICLE_TYPES[rng.integers(0, len(VEHICLE_TYPES))]

        hour = int(rng.integers(0, 24))
        is_night = int(hour < 6 or hour >= 20)
        speed_limit_kmh = float(rng.choice([40, 60, 80, 100, 120]))
        estimated_speed_kmh = float(np.clip(speed_limit_kmh * rng.normal(1.1, 0.2), 20, 180))
        road_quality_score = float(np.clip(rng.beta(3, 2), 0.1, 1.0))
        lighting_quality = float(np.clip(rng.beta(2, 3) if is_night else rng.beta(5, 1), 0.1, 1.0))
        n_vehicles_involved = int(rng.integers(1, 6))
        casualties = int(rng.poisson(1.5 * (1 + is_night * 0.5)))
        fatalities = int(rng.binomial(casualties, 0.25))
        alcohol_suspected = int(rng.random() < 0.15)
        dist_to_nearest_hospital_km = max(0.1, float(rng.exponential(8)))
        road_surface_wet = int(weather in ("Rain", "Foggy"))
        visibility_m = float(rng.normal(150 if weather == "Harmattan Dust" else 800 if is_night else 5000, 200))
        visibility_m = max(20, visibility_m)

        severity_score = (
            (estimated_speed_kmh / 180) * 0.25
            + is_night * 0.15
            + alcohol_suspected * 0.15
            + (1 - road_quality_score) * 0.15
            + (1 - lighting_quality) * 0.10
            + road_surface_wet * 0.10
            + (dist_to_nearest_hospital_km / 50) * 0.10
        )
        is_hotspot_segment = int(rng.random() < (0.3 + 0.4 * (1 - road_quality_score)))

        records.append({
            "accident_id": acc_id, "state": state, "road_type": road_type,
            "latitude": round(lat + rng.normal(0, 0.25), 6),
            "longitude": round(lon + rng.normal(0, 0.25), 6),
            "hour_of_day": hour, "is_night": is_night,
            "crash_type": crash_type, "weather_condition": weather,
            "vehicle_type": vehicle_type,
            "speed_limit_kmh": speed_limit_kmh,
            "estimated_speed_kmh": round(estimated_speed_kmh, 1),
            "road_quality_score": round(road_quality_score, 4),
            "lighting_quality": round(lighting_quality, 4),
            "n_vehicles_involved": n_vehicles_involved,
            "casualties": casualties, "fatalities": fatalities,
            "alcohol_suspected": alcohol_suspected,
            "dist_to_nearest_hospital_km": round(dist_to_nearest_hospital_km, 2),
            "road_surface_wet": road_surface_wet,
            "visibility_m": round(visibility_m, 0),
            "severity_score": round(severity_score, 4),
            "is_hotspot_segment": is_hotspot_segment,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_accident_dataset()
    path = output_dir / "accident_data.csv"
    df.to_csv(path, index=False)
    return path
