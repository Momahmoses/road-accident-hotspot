# Road Accident Hotspot Analysis & Traffic Safety

ML-powered road safety intelligence platform identifying high-risk corridors across Nigeria. Integrates GPS accident reports, road quality, weather conditions, and speed data to generate density heatmaps and actionable insights for the Federal Road Safety Corps (FRSC) and state governments.

---

## Features

- Gradient Boosting hotspot classifier with categorical encoding
- Density heatmap using Plotly Mapbox
- Hour-of-day, crash-type, and weather-condition analysis
- Severity scoring model across 10 numerical and 4 categorical features
- State-by-state fatality and casualty breakdown

---

## Setup & Run

```bash
git clone https://github.com/Momahmoses/road-accident-hotspot.git
cd road-accident-hotspot
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Tech Stack

`Python` · `scikit-learn` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`

---

## Author

**Momah Moses** — [github.com/Momahmoses](https://github.com/Momahmoses)
