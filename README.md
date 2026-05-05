# Road Accident Hotspot Analysis & Traffic Safety

GPS accident reports + road geometry + traffic counts + weather data to map high-risk corridors and support Federal Road Safety Corps (FRSC) and state governments in reducing road fatalities across Nigeria.

## Features
- Kernel Density Estimation (KDE) for spatial hotspot detection
- Hotspot clustering with accident count, fatality, and risk scoring
- Random Forest fatality severity predictor
- Hour-of-day, weather, and road-type risk factor analysis
- Interactive heatmap with annotated hotspot clusters (Folium)

## Project Structure
```
road-accident-hotspot/
├── src/
│   ├── data_loader.py    # Accident record generation
│   ├── analysis.py       # KDE hotspot detection, clustering, severity model
│   └── visualize.py      # Hotspot maps and risk factor charts
├── data/raw/             # FRSC accident logs, road network, weather data
├── models/               # Saved severity classifier
├── outputs/              # Maps, reports
├── config.yaml
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Data Sources
| Layer | Source |
|-------|--------|
| Accident records | FRSC Annual Reports |
| Road network | OpenStreetMap via OSMnx |
| Traffic counts | State traffic management agencies |
| Weather | Nigeria Met Service |

## Output
- `outputs/accident_hotspot_map.html` — interactive accident density and hotspot map
- `outputs/hotspot_report.csv` — ranked hotspot clusters
- `outputs/risk_factors_chart.png` — hour, weather, road-type analysis

## Author
MOMAH MOSES .C.
