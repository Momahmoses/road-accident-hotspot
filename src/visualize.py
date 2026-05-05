import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap, MarkerCluster


SEVERITY_COLORS = {"fatal": "#d73027", "serious": "#fc8d59", "minor": "#fee08b"}


def plot_hotspot_map(gdf, hotspot_clusters, output_path):
    center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    heat_data = [[row["latitude"], row["longitude"], row["kde_density"]]
                 for _, row in gdf.iterrows()]
    HeatMap(heat_data, radius=15, blur=12, name="Accident Density").add_to(m)

    cluster_layer = MarkerCluster(name="Hotspot Clusters").add_to(m)
    for _, row in hotspot_clusters.head(50).iterrows():
        folium.CircleMarker(
            location=[row["center_lat"], row["center_lon"]],
            radius=max(8, min(25, row["accident_count"] / 3)),
            color="#d73027", fill=True, fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>Hotspot {row['cluster_id']}</b><br>"
                f"Accidents: {row['accident_count']}<br>"
                f"Fatalities: {row['fatal_count']}<br>"
                f"Road: {row['dominant_road_type']}<br>"
                f"State: {row['state']}<br>"
                f"Risk Score: {row['risk_score']:.4f}",
                max_width=220
            )
        ).add_to(cluster_layer)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Hotspot map saved: {output_path}")


def plot_risk_factors(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    severity_hour = df.groupby("hour_of_day")["severity_score"].mean()
    axes[0, 0].bar(severity_hour.index, severity_hour.values, color="#d73027", alpha=0.8)
    axes[0, 0].set_title("Average Severity by Hour of Day")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Severity Score")

    weather_counts = df.groupby(["weather", "severity"]).size().unstack(fill_value=0)
    weather_counts.plot(kind="bar", stacked=True,
                        color=["#d73027", "#fee08b", "#1a9850"], ax=axes[0, 1])
    axes[0, 1].set_title("Accidents by Weather Condition")
    axes[0, 1].tick_params(axis="x", rotation=0)

    road_fatal = df[df["severity"] == "fatal"].groupby("road_type").size().sort_values(ascending=False)
    road_fatal.plot(kind="barh", color="#fc8d59", ax=axes[1, 0])
    axes[1, 0].set_title("Fatalities by Road Type")

    sns.histplot(df["estimated_speed_kmh"], bins=40, kde=True, color="#d73027", ax=axes[1, 1])
    axes[1, 1].axvline(df["speed_limit_kmh"].mean(), color="black", linestyle="--", label="Avg Speed Limit")
    axes[1, 1].set_title("Estimated Speed Distribution")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
