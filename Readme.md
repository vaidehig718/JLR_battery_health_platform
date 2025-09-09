# JLR Battery Health Platform

The JLR Battery Health Platform is a real-time monitoring and predictive analytics solution for electric vehicle batteries. It leverages machine learning to predict key battery health metrics, providing actionable insights through an interactive dashboard.

## Features

### Dashboard & UI Layer

The dashboard, built using Streamlit, provides a real-time interface for monitoring battery health. Key features include:

*   **Live metrics:** SOC, SOH, temperature, current
*   **Interactive sensor simulation sliders**
*   **Real-time alerts with severity levels**
*   **Risk breakdown (thermal, voltage, aging)**
*   **Historical trends for SOC and SOH**
*   **Vehicle-specific insights and metadata**

### Smart Assistant EVA

EVA (Electric Vehicle Assistant) is a contextual tip engine embedded in the dashboard. Based on real-time sensor data and ML predictions, EVA provides:

*   **Charging recommendations**
*   **Cooling suggestions during high temperature**
*   **Maintenance alerts for aging batteries**
*   **Personalized messages based on usage patterns**

### Sensor Simulation

This model emulates real-world sensor behavior through interactive dashboard sliders, allowing users to test and visualize system responses before transitioning to live sensors data.

*   **Simulates real-world sensor input using dashboard sliders**
*   **Parameters:** Temperature, Pack Current, Cell Voltage, Battery Age, Car Model.
*   **Simulated values are processed by the ML model to predict:**
    *   SoC
    *   SoH
    *   Risk Levels (thermal, voltage, aging)
    *   Real-Time Alerts

## Machine Learning Pipeline

We implemented a robust ML pipeline using Random Forest models to predict key battery health metrics:

*   **State of Charge (SOC)**
*   **State of Health (SOH)**
*   **Risk Level Classification**
*   **Critical Alert Detection**

The models were trained on over 1,000 real-world battery readings across 5 vehicle models. Feature engineering included parameters such as:

*   `Avg_Cell_Voltage`
*   `Pack_Current`
*   `Temperature_Gradient`
*   `Internal_Resistance`
*   `Cycle_Count`

SOC and SOH models achieved a Mean Absolute Error (MAE) of **±2.1%** and **±2.3%** respectively. Risk and alert classifiers achieved over **90%** accuracy.

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/jlr-battery-health-platform.git
    cd jlr-battery-health-platform
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r jlr-battery-health-platform/requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit dashboard:**

    ```bash
    streamlit run jlr-battery-health-platform/dashboard/app.py
    ```

2.  **Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).**

## Project Structure

```
jlr-battery-health-platform/
├── dashboard/
│   └── app.py              # Streamlit dashboard application
├── data/
│   ├── processed_dataset_with_targets.csv
│   └── raw_battery_dataset_1000.csv
├── models/
│   ├── alert_model.pkl
│   ├── feature_importance.csv
│   ├── risk_model.pkl
│   ├── scaler.pkl
│   ├── soc_model.pkl
│   ├── soh_model.pkl
│   ├── training_battery_models.py
│   └── encoders/
│       ├── charging_state_encoder.pkl
│       ├── location_encoder.pkl
│       └── vehicle_model_encoder.pkl
├── debug_ml.py
├── generator.py
├── installed_packages.txt
├── requirements.txt
└── test_models.py
```

## Dependencies

The main dependencies are listed in `jlr-battery-health-platform/requirements.txt` and include:

*   `streamlit`
*   `pandas`
*   `numpy`
*   `plotly`
*   `scikit-learn`
*   `joblib`
