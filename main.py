import os
from src.data_loader import load_config, generate_accident_dataset, to_geodataframe
from src.analysis import (
    compute_kde_hotspots, cluster_hotspots,
    train_severity_model, save_model
)
from src.visualize import plot_hotspot_map, plot_risk_factors


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = load_config("config.yaml")
    print("[1/5] Config loaded — FRSC Road Safety Analysis")

    df = generate_accident_dataset(n=5000)
    print(f"[2/5] {len(df)} accident records loaded — {(df['severity'] == 'fatal').sum()} fatal cases")

    df = compute_kde_hotspots(df)
    hotspots = cluster_hotspots(df, radius_km=config["hotspot_radius_km"])
    hotspots.to_csv(config["output"]["hotspot_report"], index=False)
    print(f"[3/5] {len(hotspots)} hotspot clusters identified")
    print(f"      Top 5 hotspots:")
    print(hotspots[["cluster_id", "state", "accident_count", "fatal_count"]].head(5).to_string(index=False))

    model, report = train_severity_model(df, config)
    print(f"[4/5] Severity model trained:\n{report}")
    save_model(model, "models/severity_model.pkl")

    gdf = to_geodataframe(df)
    plot_hotspot_map(gdf, hotspots, config["output"]["hotspot_map"])
    plot_risk_factors(df, config["output"]["risk_chart"])
    print("[5/5] All outputs saved to /outputs/")
    print("\nDone.")


if __name__ == "__main__":
    main()
