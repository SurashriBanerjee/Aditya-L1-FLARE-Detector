<div align="center">

# ☀️ S.F.E.W.S
### Aditya-L1 Solar Flare Early Warning System

**A real-time, AI-driven mission control dashboard predicting catastrophic solar flares using live telemetry from ISRO's Aditya-L1 spacecraft.**

*Built for the Bhartiya Antariksh Hackathon 2024*

</div>

---

## 🌌 Overview

When a massive **X-Class solar flare** hits Earth, it can trigger global radio blackouts, cripple GPS systems, and damage power grids. By analyzing X-ray emissions from the sun in real time, S.F.E.W.S gives us the crucial early-warning window needed to brace for impact.

> ⚡ **TL;DR** — Aditya-L1 sensor data → physics-informed feature engineering → Random Forest prediction → immersive 3D mission-control dashboard.

---

## 🚀 What Does It Do?

| Capability | Description |
|---|---|
| 🛰️ **Ingests Space Data** | Processes raw ISRO PRADAN data from two Aditya-L1 instruments — **SoLEXS** (soft X-rays / thermal buildup) and **HEL1OS** (hard X-rays / non-thermal explosions) |
| 🧠 **Thinks Like an Astrophysicist** | Grounded in real solar physics (the **Neupert Effect**) — evaluates rate-of-change and soft/hard X-ray ratios, not just raw values |
| 🔮 **Predicts the Future** | A **Random Forest** model estimates the probability of a flare occurring within the **next 30 minutes** |
| 🌍 **Visualizes the Threat** | An immersive "Mission Control" UI with a live **3D magnetosphere simulation** and Geiger-counter-style audio sonification |

---

## 🧠 How the AI Works

```
   Quiet Sun Baseline          Extreme Event Anchor
  ┌─────────────────┐        ┌──────────────────────┐
  │  Noise Floor     │        │  Rapid ΔSoft X-ray    │
  │  (normal solar   │  +     │  + Hard X-ray burst   │
  │   background)    │        │  = Neupert signature  │
  └─────────────────┘        └──────────────────────┘
              │                         │
              └───────────┬─────────────┘
                           ▼
                 🌲 Random Forest Model
                           ▼
                 🚨 X-Class Flare Alert
```

The model is trained on thousands of minutes of baseline telemetry to learn what "quiet sun" looks like, then anchored with extreme out-of-distribution events. When soft X-ray counts spike rapidly **and** hard X-rays burst simultaneously, the model flags an impending X-Class flare and triggers system lockdown.

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technologies |
|---|---|
| **Backend / ML Engine** | Python · FastAPI · Scikit-Learn · Pandas · NumPy |
| **Frontend** | HTML5 · Tailwind CSS · Vanilla JavaScript |
| **Visualizations** | Three.js (3D Magnetosphere) · Observable Plot / D3.js (Telemetry) |
| **Data Source** | ISRO PRADAN — Aditya-L1 Mission Data |

</div>

---

## 💻 Getting Started

### 1️⃣ Prerequisites

Requires **Python 3.x**. Install dependencies:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic astropy
```

### 2️⃣ Train the Model *(optional but recommended)*

Generates the "brain" of the system — `flare_predictor_v1.pkl`:

```bash
python ml_engine/train_model.py
```

> This parses the dataset, applies the noise floor, injects the extreme-event calibration anchor, and saves the compiled model.

### 3️⃣ Start the Backend API

```bash
python backend/main.py
```

The API will be live at **`http://localhost:8000`**.

### 4️⃣ Launch Mission Control (Frontend)

No web server needed — just double-click **`index.html`** to open it in Chrome, Firefox, or Safari. 🎉

---

## 🎮 Demo Mode — "Inject Anomaly"

Since the sun is quiet 99% of the time, a nominal dashboard doesn't make for the most thrilling demo. Click **Inject Anomaly** to safely simulate an apocalyptic thermal buildup:

```
[ Inject Anomaly ]
        │
        ▼
Translates graph data → raw ISRO sensor counts
        │
        ▼
Applies low-pass filter (removes UI jitter)
        │
        ▼
Sends clamped simulated data to Python backend
        │
        ▼
AI detects Neupert Effect signature
        │
        ▼
🚨 CRITICAL state → audio alarms → magnetosphere compresses
```

Click **Reset** to return to normal operations.

---

## 📂 Project Structure

```
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── routes.py           # API endpoints
│   └── config.py           # Environment configurations
│
├── data_pipeline/
│   ├── fetch_pradan.py     # ISRO data downloader & parser
│   └── preprocess.py       # Cleans and aligns SoLEXS/HEL1OS data
│
├── ml_engine/
│   ├── train_model.py      # Random forest training script
│   ├── inference.py        # Connects the model to the API
│   ├── physics_rules.py    # Astrophysics math (Neupert effect)
│   └── explainability.py   # XAI (Explainable AI) logic
│
├── index.html              # The Mission Control Dashboard
└── requirements.txt        # Python dependencies
```

---

<div align="center">

**Built with ❤️ for the Bhartiya Antariksh Hackathon 2026**

☀️ 🛰️ 🌍 📡 🚨

</div>
