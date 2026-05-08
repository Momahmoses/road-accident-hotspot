"""
Road Accident Hotspot Analysis & Traffic Safety
================================================
ML-powered platform identifying high-risk road corridors across Nigeria.
Integrates GPS accident reports, road geometry, weather, and speed data
to generate hotspot maps and safety recommendations for FRSC and state governments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_accident_dataset
from model import (
    CAT_COLS, NUM_FEATURES, METRICS_PATH, MODEL_PATH, ENCODERS_PATH,
    encode_features, get_feature_cols, load_model, save_model, train,
)

st.set_page_config(page_title="Road Accident Hotspot | Nigeria", page_icon="🚗", layout="wide")

SEVERITY_SCALE = px.colors.sequential.Reds


@st.cache_resource(show_spinner="Analyzing accident patterns…")
def get_model_and_data():
    df = generate_accident_dataset()
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        pipeline, encoders = load_model()
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        pipeline, encoders, metrics = train(df)
        save_model(pipeline, encoders, metrics)
    df_enc, _ = encode_features(df)
    feat_cols = get_feature_cols(df_enc)
    df["hotspot_probability"] = pipeline.predict_proba(df_enc[feat_cols])[:, 1]
    return pipeline, encoders, metrics, df


pipeline, encoders, metrics, df = get_model_and_data()

with st.sidebar:
    st.title("🚗 Accident Hotspot")
    st.caption("Nigeria Traffic Safety Intelligence")
    st.divider()
    page = st.radio("Navigation", ["Overview", "Hotspot Map", "Risk Analysis", "Model Performance"], label_visibility="collapsed")
    st.divider()
    state_filter = st.multiselect("Filter by State", sorted(df["state"].unique()), default=sorted(df["state"].unique()))

df_f = df[df["state"].isin(state_filter)]

if page == "Overview":
    st.title("Road Accident Hotspot Analysis & Traffic Safety")
    st.markdown("Identifying high-risk road corridors across Nigeria to support FRSC interventions, speed enforcement, and infrastructure improvement.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidents Analyzed", f"{len(df_f):,}")
    c2.metric("Total Fatalities", f"{df_f['fatalities'].sum():,}")
    c3.metric("Night Accidents", f"{df_f['is_night'].sum():,}", f"{df_f['is_night'].mean():.1%} of total", delta_color="inverse")
    c4.metric("Model ROC-AUC", f"{metrics['roc_auc_test']:.4f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        state_fat = df_f.groupby("state")[["fatalities", "casualties"]].sum().reset_index().sort_values("fatalities", ascending=False)
        fig = px.bar(state_fat, x="state", y=["fatalities", "casualties"],
                     barmode="group", title="Fatalities & Casualties by State",
                     color_discrete_sequence=["#E74C3C", "#F39C12"], height=420)
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        hour_counts = df_f.groupby("hour_of_day").size().reset_index(name="accidents")
        fig2 = px.bar(hour_counts, x="hour_of_day", y="accidents", color="accidents",
                      color_continuous_scale="Reds", title="Accident Frequency by Hour of Day", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        crash_dist = df_f["crash_type"].value_counts().reset_index()
        fig3 = px.pie(crash_dist, values="count", names="crash_type",
                      title="Crash Type Distribution", height=380)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        weather_sev = df_f.groupby("weather_condition")["severity_score"].mean().reset_index().sort_values("severity_score", ascending=False)
        fig4 = px.bar(weather_sev, x="weather_condition", y="severity_score",
                      color="severity_score", color_continuous_scale="Reds",
                      title="Mean Accident Severity by Weather Condition", height=380)
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Hotspot Map":
    st.title("Accident Hotspot Geographic Map")
    sample = df_f.sample(min(2000, len(df_f)), random_state=42)
    fig = px.density_mapbox(
        sample, lat="latitude", lon="longitude",
        z="hotspot_probability", radius=15,
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-darkmatter", zoom=5,
        center={"lat": 7.5, "lon": 5.5},
        title="Accident Hotspot Density Map — Nigeria", height=580,
    )
    fig.update_layout(coloraxis_colorbar_title="Hotspot Risk")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Risk Analysis":
    st.title("Accident Severity & Risk Factor Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_f.sample(min(2000, len(df_f)), random_state=42),
                         x="estimated_speed_kmh", y="severity_score",
                         color="road_type", opacity=0.5,
                         title="Speed vs Accident Severity by Road Type",
                         labels={"estimated_speed_kmh": "Estimated Speed (km/h)", "severity_score": "Severity Score"},
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        road_risk = df_f.groupby("road_type")["is_hotspot_segment"].mean().reset_index().sort_values("is_hotspot_segment", ascending=False)
        fig2 = px.bar(road_risk, x="is_hotspot_segment", y="road_type", orientation="h",
                      color="is_hotspot_segment", color_continuous_scale="Reds",
                      title="Hotspot Rate by Road Type", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    fi = metrics["feature_importance"]
    fi_s = sorted(fi.items(), key=lambda x: x[1])
    fig3 = go.Figure(go.Bar(x=[v for _, v in fi_s], y=[k for k, _ in fi_s], orientation="h", marker_color="#E74C3C"))
    fig3.update_layout(title="Feature Importance", height=440, margin={"l": 200, "r": 60})
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance — Gradient Boosting Hotspot Classifier")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{metrics['roc_auc_test']:.4f}")
    m2.metric("CV AUC", f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    m3.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m4.metric("Hotspot F1", f"{metrics['f1_hotspot']:.4f}")
    cm = metrics["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=np.array(cm), x=["Normal", "Hotspot"], y=["Normal", "Hotspot"],
        colorscale="Reds", showscale=False,
        text=[[str(v) for v in row] for row in cm], texttemplate="%{text}", textfont={"size": 18},
    ))
    fig.update_layout(title="Confusion Matrix", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full Metrics"):
        st.json(metrics)
