# dashboard/app.py
# =================
# JLR Battery Health Platform - Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime, timedelta
import time

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure page
st.set_page_config(
    page_title="AmpSphere",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for JLR styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
    padding: 1rem 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #059669;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}

.alert-critical {
    background: #FEE2E2;
    border-left: 4px solid #DC2626;
    padding: 1rem;
    color: black;
    font-weight: 600;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.alert-warning {
    background: #FEF3C7;
    border-left: 4px solid #F59E0B;
    padding: 1rem;
    color: black;
    font-weight: 600;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.alert-normal {
    background: #D1FAE5;
    border-left: 4px solid #059669;
    padding: 1rem;
    color: black;
    font-weight: 600;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.risk-low {
    background-color: #D1FAE5;
    color: #065F46;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: bold;
}

.risk-medium {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: bold;
}

.risk-high {
    background-color: #FED7AA;
    color: #C2410C;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: bold;
}

.risk-critical {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: bold;
}

.sidebar .sidebar-content {
    background: #F8FAFC;
}
</style>
""", unsafe_allow_html=True)


# Load ML models
@st.cache_resource
def load_ml_models():
    """Load trained ML models"""
    models = {}
    try:
        models['soc'] = joblib.load('models/soc_model.pkl')
        models['soh'] = joblib.load('models/soh_model.pkl')
        models['risk'] = joblib.load('models/risk_model.pkl')
        models['alert'] = joblib.load('models/alert_model.pkl')
        models['scaler'] = joblib.load('models/scaler.pkl')
        
        # Load encoders
        models['vehicle_encoder'] = joblib.load('models/encoders/vehicle_model_encoder.pkl')
        models['charging_encoder'] = joblib.load('models/encoders/charging_state_encoder.pkl')
        models['location_encoder'] = joblib.load('models/encoders/location_encoder.pkl')
        
        return models
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please make sure you've trained the ML models first!")
        return None


# Load processed data
@st.cache_data
def load_processed_data():
    """Load processed battery data"""
    try:
        df = pd.read_csv('data/processed_dataset_with_targets.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        st.error("Processed dataset not found!")
        return None


def predict_battery_health(models, sensor_data):
    """Make ML predictions using trained models"""
    if models is None:
        return None
        
    try:
        # Prepare feature vector (simplified - matching your model training)
        features = np.array([[
            sensor_data.get('Battery_Age_Months', 24),
            sensor_data.get('Avg_Cell_Voltage', 3.6),
            sensor_data.get('Max_Cell_Voltage', 3.8),
            sensor_data.get('Min_Cell_Voltage', 3.4),
            sensor_data.get('Pack_Current', 0),
            sensor_data.get('Avg_Temperature', 25),
            sensor_data.get('Max_Temperature', 30),
            sensor_data.get('Min_Temperature', 20),
            sensor_data.get('Internal_Resistance', 3.0),
            sensor_data.get('Pressure_Level', 1.2),
            sensor_data.get('Coolant_Flow_Rate', 2.5),
            sensor_data.get('Cycle_Count', 300),
            sensor_data.get('Voltage_Imbalance', 0.05),
            sensor_data.get('Temperature_Gradient', 5),
            sensor_data.get('Power_kW', 10),
            sensor_data.get('Humidity_Percent', 60),
            sensor_data.get('Air_Pressure_kPa', 101),
            sensor_data.get('Hour', 12),
            sensor_data.get('Day_of_Week', 1),
            sensor_data.get('Month', 6),
            sensor_data.get('Vehicle_Model_Encoded', 0),
            sensor_data.get('Charging_State_Encoded', 1),
            sensor_data.get('Location_City_Encoded', 0)
        ]])
        
        # Scale features
        features_scaled = models['scaler'].transform(features)
        
        # Make predictions
        soc_pred = models['soc'].predict(features_scaled)[0]
        soh_pred = models['soh'].predict(features_scaled)[0]
        risk_pred = models['risk'].predict(features_scaled)[0]
        alert_pred = models['alert'].predict(features_scaled)[0]
        
        return {
            'soc': max(5, min(95, soc_pred)),
            'soh': max(40, min(100, soh_pred)),
            'risk_level': risk_pred,
            # 'critical_alert': bool(alert_pred),
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def create_battery_gauge(value, title, color="blue"):
    """Create a battery gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 85], 'color': "lightgreen"},
                {'range': [85, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_time_series_chart(df, metrics=['SOC', 'SOH']):
    """Create time series chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('State of Charge (%)', 'State of Health (%)'),
        vertical_spacing=0.3
    )
    
    # Add SOC trace
    fig.add_trace(
        go.Scatter(x=df['Timestamp'], y=df['SOC'], name='SOC (%)', line=dict(color='#059669')),
        row=1, col=1
    )
    
    # Add SOH trace
    fig.add_trace(
        go.Scatter(x=df['Timestamp'], y=df['SOH'], name='SOH (%)', line=dict(color='#DC2626')),
        row=2, col=1
    )
    
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Percentage", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    
    return fig


def create_risk_breakdown_chart(thermal_risk, voltage_risk, aging_risk):
    """Create risk breakdown chart"""
    categories = ['Thermal Risk', 'Voltage Risk', 'Aging Risk']
    values = [thermal_risk, voltage_risk, aging_risk]
    colors = ['#DC2626' if v > 60 else '#F59E0B' if v > 45 else "#0A7C58" for v in values]
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Risk Assessment Breakdown",
        yaxis_title="Risk Score (0-100)",
        height=400
    )
    
    return fig


# Main dashboard
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔋 AmpSphere - Battery intelligence, simplified </h1>
        <p>Real-time monitoring and predictive analytics for electric vehicle batteries</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and data
    models = load_ml_models()
    df = load_processed_data()
    
    if models is None or df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🚗 Vehicle Selection")
    
    # Vehicle selection
    vehicles = df['Vehicle_ID'].unique()
    selected_vehicle = st.sidebar.selectbox("Select Vehicle", vehicles)
    
    # Get vehicle data
    vehicle_data = df[df['Vehicle_ID'] == selected_vehicle].iloc[-1] if len(df) > 0 else None
    
    if vehicle_data is None:
        st.error("No data found for selected vehicle")
        return
    
    # Simulate real-time sensor data (in reality, this would come from IoT sensors)
    with st.sidebar:
        st.subheader("🔧 Sensor Simulation")
        
        # Allow manual adjustment of sensor values for demo
        voltage = st.slider("Cell Voltage (V)", 3.0, 4.2, float(vehicle_data['Avg_Cell_Voltage']))
        current = st.slider("Pack Current (A)", -100.0, 100.0, float(vehicle_data['Pack_Current']))
        temperature = st.slider("Temperature (°C)", 15.0, 60.0, float(vehicle_data['Avg_Temperature']))
        age_months = st.slider("Battery Age (months)", 1, 60, int(vehicle_data['Battery_Age_Months']))
    
    # Prepare sensor data for prediction
    sensor_data = {
        'Battery_Age_Months': age_months,
        'Avg_Cell_Voltage': voltage,
        'Max_Cell_Voltage': voltage + 0.1,
        'Min_Cell_Voltage': voltage - 0.1,
        'Pack_Current': current,
        'Avg_Temperature': temperature,
        'Max_Temperature': temperature + 5,
        'Min_Temperature': temperature - 3,
        'Internal_Resistance': vehicle_data['Internal_Resistance'],
        'Pressure_Level': vehicle_data['Pressure_Level'],
        'Coolant_Flow_Rate': vehicle_data['Coolant_Flow_Rate'],
        'Cycle_Count': vehicle_data['Cycle_Count'],
        'Voltage_Imbalance': abs(voltage + 0.1 - (voltage - 0.1)),
        'Temperature_Gradient': 5,
        'Power_kW': abs(current * voltage) / 1000,
        'Humidity_Percent': vehicle_data['Humidity_Percent'],
        'Air_Pressure_kPa': vehicle_data['Air_Pressure_kPa'],
        'Hour': datetime.now().hour,
        'Day_of_Week': datetime.now().weekday(),
        'Month': datetime.now().month,
        'Vehicle_Model_Encoded': 0,
        'Charging_State_Encoded': 1 if current < 0 else 0,
        'Location_City_Encoded': 0
    }
    
    # Make predictions
    predictions = predict_battery_health(models, sensor_data)
    
    if predictions is None:
        st.error("Unable to make predictions")
        return
    
    st.write("")
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        st.metric("🔋 State of Charge", f"{predictions['soc']:.1f}%", delta=f"{predictions['soc'] - vehicle_data['SOC']:.1f}%")
    
    with col2:
        st.metric("💚 State of Health", f"{predictions['soh']:.1f}%", delta=f"{predictions['soh'] - vehicle_data['SOH']:.1f}%")
    
    with col3:
        charging_state = "Charging" if current < 0 else "Discharging" if current > 5 else "Idle"
        st.metric("⚡ Current", f"{current:.1f}A", charging_state)
    
    with col4:
        st.metric("🌡 Temperature", f"{temperature:.1f}°C", delta=f"{temperature - 25:.1f}°C" if temperature != 25 else None)
    
    st.write("")
    st.write("")
    
    # Alert system
    st.subheader("🚨 Alert System")
    
    # Generate alerts based on predictions and thresholds
    alerts = []
    
    if predictions['soc'] < 20:
        alerts.append(("Critical", "Low battery - charge immediately"))
    
    if temperature > 40:
        alerts.append(("Warning", "High temperature detected"))
    
    if predictions['soh'] < 50:
        alerts.append(("Warning", "Battery health degraded"))
    
    # if predictions['critical_alert']:
    #     alerts.append(("Critical", "Critical system alert"))
    
    if not alerts:
        alerts.append(("Normal", "All systems operating normally"))
    
    for alert_type, message in alerts:
        alert_class = f"alert-{alert_type.lower()}"
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{alert_type}:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Charts section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔋 Battery Metrics")
        soc_gauge = create_battery_gauge(predictions['soc'], "State of Charge", "blue")
        st.plotly_chart(soc_gauge, use_container_width=True)
    
    with col2:
        st.write("")
        st.subheader("💚 Health Metrics")
        soh_gauge = create_battery_gauge(predictions['soh'], "State of Health", "blue")
        st.plotly_chart(soh_gauge, use_container_width=True)
    
    with col3:
        # Ambient Info + Smart Tips
        st.subheader("🌤 Environment & Smart Tips")
        col1, col2 = st.columns([1, 2], gap="medium")
        
        with col1:
            st.info(f"*Ambient Temp:* {vehicle_data['Ambient_Temperature']:.1f}°C")
            st.info(f"*Humidity:* {vehicle_data['Humidity_Percent']}%")
            st.info(f"*Location:* {vehicle_data['Location_City']}")
            st.info(f"*Altitude:* {vehicle_data['Altitude_m']} m")
        
        with col2:
            st.markdown("#### Hi, I am EVA! 🌱")
            # st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            tips = []
            
            # Friendly greetings
            # Dynamic suggestions
            if age_months > 50:
                tips.append("📉 <b>Battery age is high.</b><br>Time for a quick health check")
                tips.append("<br>🔋 <b>Older battery.</b><br>May affect range and charging.")
                tips.append("<br>🛠 <b>Check-up due?</b><br>Diagnostics can help.")
                tips.append("<br>⏳ <b>Near end of life.</b><br>Plan for servicing.")
            else:
                if predictions['soc'] < 20:
                    if temperature > 40:  # battery temp shoots up in general
                        tips.append("⚠ <b>Low battery & high temp.</b><br>Let's cool down first:")
                        tips.append("<br>🛑 <b>Stop the EV</b> - Let it rest.")
                        tips.append("<br>🌳 <b>Find shade</b> - Cool the battery.")
                        tips.append("<br>📟 <b>Check alerts</b> - Watch for warnings.")
                    else:
                        tips.append("🔌 <b>Battery is low.</b><br>Time to charge:")
                        tips.append("<br>📍 <b>Find a charger</b> nearby.")
                        tips.append("<br>⚡ <b>Save power</b> - Turn off extras.")
                        tips.append("<br>🗺 <b>Use maps</b> to plan route.")
                
                if predictions['soh'] < 50:  # battery health needs to be checked
                    tips.append("🛠 <b>Battery health is low.</b><br>Check-up recommended:")
                    tips.append("<br>🔧 <b>Time for a check?</b><br>Stay ahead of issues.")
                    tips.append("<br>📅 <b>Book service</b> if needed.")
                    tips.append("<br>⚙ <b>Performance drop?</b><br>Get it inspected.")
                
                if temperature > 40 and current > 0 and predictions['soc'] > 20:  # temperature shoots up while charging
                    tips.append("🌡 <b>It's getting hot in here!</b><br>Maybe take a short break to let the system cool down:")
                    tips.append("<br>🛑 <b>High temperature detected.</b><br>Consider pausing your drive or charging session.")
                    tips.append("<br>🧊 <b>Cooling down helps preserve battery health.</b><br>Find a shaded spot or ventilated area.")
                    tips.append("<br>🚗 <b>Reduce load on the battery.</b><br>Turn off non-essential systems like AC or infotainment.")
                
                if current < 0 and temperature < 40:
                    tips.append("🔋 <b>Charging now.</b><br>Keep an eye on temp:")
                    tips.append("<br>🌡 <b>Monitor heat</b> during charge.")
                    tips.append("<br>🧊 <b>Cool if needed</b> - Pause briefly.")
                
                if current < 0 and temperature > 40:
                    tips.append("🌡 <b>Hot while charging.</b><br>Take a break:")
                    tips.append("<br>🛑 <b>Pause charging</b> if needed.")
                    tips.append("<br>🧊 <b>Cool the battery</b> in shade.")
                    tips.append("<br>⚠ <b>Watch temp</b> - Avoid damage.")
            
            if not tips:
                tips.append("👋 Hey there! Hope you're having a smooth ride today.")
            
            for tip in tips:
                st.markdown(f"<div class='chat-bubble'>{tip}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    # Risk assessment
    st.markdown("### ⚠ Real-time Risk Assessment")
    risk_level = predictions['risk_level']
    risk_class = f"risk-{risk_level.lower()}" if isinstance(risk_level, str) else "risk-medium"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="{risk_class}">
            Risk Level: {risk_level}
        </div>
        """, unsafe_allow_html=True)
        
        # st.write("")
        # st.markdown("#### 🎯 ML Model Accuracy")
        # st.success("SOC Prediction: 99.1% accurate")
        # st.success("SOH Estimation: 99.2% accurate")
        # st.success("Risk Classification: 95.1% accurate")
    
    with col2:
        # Generate risk breakdown (simplified)
        thermal_risk = max(0, (temperature - 25) * 2)
        voltage_risk = abs(voltage - 3.7) * 100
        aging_risk = age_months * 1.5
        
        risk_fig = create_risk_breakdown_chart(thermal_risk, voltage_risk, aging_risk)
        st.plotly_chart(risk_fig, use_container_width=True)
    
    # Historical trends
    st.subheader("📈 Historical Trends")
    
    # Filter data for selected vehicle
    vehicle_history = df[df['Vehicle_ID'] == selected_vehicle].tail(50)  # Last 50 readings
    
    if len(vehicle_history) > 0:
        trend_chart = create_time_series_chart(vehicle_history)
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("No historical data available for this vehicle")
    
    # Vehicle information
    st.subheader("🚗 Vehicle Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"*Vehicle ID:* {selected_vehicle}")
    
    with col2:
        st.info(f"*Model:* {vehicle_data['Vehicle_Model']}")
    
    with col3:
        st.info(f"*Location:* {vehicle_data['Location_City']}")
    
    with col4:
        st.info(f"*Battery Age:* {age_months} months")
    
    # Auto-refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748B;'>
        <p>🔋 JLR Battery Health Platform | Real-time ML Predictions | Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()