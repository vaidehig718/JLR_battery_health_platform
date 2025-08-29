# train_battery_models.py
# =======================
# Complete ML training script for JLR Battery Health Platform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔋 JLR Battery Health ML Training")
print("=" * 50)

# ============================================================================
# STEP 1: Load and Explore Data
# ============================================================================

print("📊 Loading dataset...")
try:
    df = pd.read_csv('data/raw_battery_dataset_1000.csv')
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    print("❌ Dataset not found! Make sure raw_battery_dataset_1000.csv is in data/ folder")
    exit()

print("\n📋 Dataset info:")
print(f"  - Columns: {list(df.columns)}")
print(f"  - Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"  - Vehicles: {df['Vehicle_ID'].nunique()}")
print(f"  - Models: {df['Vehicle_Model'].unique()}")

# ============================================================================
# STEP 2: Create ML Target Variables (SOC, SOH, Risk Scores)
# ============================================================================

print("\n🎯 Creating ML target variables...")

def calculate_soc(row):
    """Calculate State of Charge based on voltage and charging state"""
    voltage = row['Avg_Cell_Voltage']
    charging_state = row['Charging_State']
    
    # Voltage to SOC mapping
    if voltage < 3.3:
        base_soc = 10
    elif voltage < 3.5:
        base_soc = 25
    elif voltage < 3.7:
        base_soc = 50
    elif voltage < 3.9:
        base_soc = 75
    else:
        base_soc = 90
    
    # Adjust based on charging state
    if charging_state == 'Charging':
        base_soc = min(95, base_soc + 10)
    elif charging_state == 'Discharging':
        base_soc = max(5, base_soc - 10)
    
    return np.clip(base_soc + np.random.uniform(-5, 5), 5, 95)

def calculate_soh(row):
    """Calculate State of Health based on age, cycles, and performance"""
    age = row['Battery_Age_Months']
    cycles = row['Cycle_Count']
    resistance = row['Internal_Resistance']
    
    # Degradation factors
    age_degradation = age * 0.5
    cycle_degradation = cycles * 0.008
    resistance_degradation = max(0, (resistance - 2.8) * 8)
    
    soh = 100 - age_degradation - cycle_degradation - resistance_degradation
    return max(40, soh + np.random.uniform(-3, 3))

# Create target variables
df['SOC'] = df.apply(calculate_soc, axis=1)
df['SOH'] = df.apply(calculate_soh, axis=1)

# Feature engineering
df['Voltage_Imbalance'] = df['Max_Cell_Voltage'] - df['Min_Cell_Voltage']
df['Temperature_Gradient'] = df['Max_Temperature'] - df['Min_Temperature']
df['Power_kW'] = abs(df['Pack_Current']) * df['Avg_Cell_Voltage'] / 1000

# Risk scores
df['Thermal_Risk_Score'] = np.clip(
    (df['Avg_Temperature'] - 25) * 2 + np.random.uniform(0, 10, len(df)), 0, 100
)
df['Voltage_Risk_Score'] = np.clip(
    df['Voltage_Imbalance'] * 200 + np.random.uniform(0, 15, len(df)), 0, 100
)
df['Aging_Risk_Score'] = np.clip(
    df['Battery_Age_Months'] * 1.5 + np.random.uniform(0, 15, len(df)), 0, 100
)
df['Overall_Risk_Score'] = (df['Thermal_Risk_Score'] + df['Voltage_Risk_Score'] + df['Aging_Risk_Score']) / 3

# Alert generation
def generate_alerts(row):
    alerts = []
    if row['SOC'] < 15:
        alerts.append('Low Battery')
    if row['Avg_Temperature'] > 45:
        alerts.append('High Temperature')
    if row['Voltage_Imbalance'] > 0.15:
        alerts.append('Cell Imbalance')
    if row['SOH'] < 45:
        alerts.append('Battery Degradation')
    if row['Internal_Resistance'] > 4:
        alerts.append('High Resistance')
    
    return ';'.join(alerts) if alerts else 'Normal'

df['Alert_Status'] = df.apply(generate_alerts, axis=1)
df['Critical_Alert'] = (df['Alert_Status'] != 'Normal').astype(int)

print("✅ Target variables created:")
print(f"  - SOC range: {df['SOC'].min():.1f}% to {df['SOC'].max():.1f}%")
print(f"  - SOH range: {df['SOH'].min():.1f}% to {df['SOH'].max():.1f}%")
print(f"  - Alerts: {df['Alert_Status'].value_counts().to_dict()}")

# ============================================================================
# STEP 3: Prepare Features for ML
# ============================================================================

print("\n⚙️ Preparing features...")

# Time features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# Select feature columns
feature_cols = [
    'Battery_Age_Months', 'Avg_Cell_Voltage', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
    'Pack_Current', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature',
    'Internal_Resistance', 'Pressure_Level', 'Coolant_Flow_Rate', 'Cycle_Count',
    'Voltage_Imbalance', 'Temperature_Gradient', 'Power_kW',
    'Humidity_Percent', 'Air_Pressure_kPa', 'Hour', 'Day_of_Week', 'Month'
]

# Encode categorical variables
encoders = {}
encoders['vehicle_model'] = LabelEncoder()
df['Vehicle_Model_Encoded'] = encoders['vehicle_model'].fit_transform(df['Vehicle_Model'])

encoders['charging_state'] = LabelEncoder()
df['Charging_State_Encoded'] = encoders['charging_state'].fit_transform(df['Charging_State'])

encoders['location'] = LabelEncoder()
df['Location_City_Encoded'] = encoders['location'].fit_transform(df['Location_City'])

# Add encoded features
feature_cols.extend(['Vehicle_Model_Encoded', 'Charging_State_Encoded', 'Location_City_Encoded'])

# Prepare feature matrix
X = df[feature_cols].fillna(0)
print(f"✅ Features prepared: {len(feature_cols)} features")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# STEP 4: Train ML Models
# ============================================================================

print("\n🤖 Training ML models...")

models = {}
results = {}

# 4.1 SOC Prediction Model
print("  🔋 Training SOC prediction model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['SOC'], test_size=0.2, random_state=42)

soc_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
soc_model.fit(X_train, y_train)
soc_pred = soc_model.predict(X_test)
soc_mae = mean_absolute_error(y_test, soc_pred)

models['soc'] = soc_model
results['SOC_MAE'] = soc_mae
print(f"     ✅ SOC Model MAE: {soc_mae:.2f}%")

# 4.2 SOH Estimation Model
print("  🔋 Training SOH estimation model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['SOH'], test_size=0.2, random_state=42)

soh_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
soh_model.fit(X_train, y_train)
soh_pred = soh_model.predict(X_test)
soh_mae = mean_absolute_error(y_test, soh_pred)

models['soh'] = soh_model
results['SOH_MAE'] = soh_mae
print(f"     ✅ SOH Model MAE: {soh_mae:.2f}%")

# 4.3 Risk Classification Model
print("  ⚠️ Training risk classification model...")
# Create risk categories
df['Risk_Category'] = pd.cut(df['Overall_Risk_Score'], 
                           bins=[0, 25, 50, 75, 100], 
                           labels=['Low', 'Medium', 'High', 'Critical'])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Risk_Category'], test_size=0.2, random_state=42)

risk_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
risk_model.fit(X_train, y_train)
risk_pred = risk_model.predict(X_test)
risk_accuracy = accuracy_score(y_test, risk_pred)

models['risk'] = risk_model
results['Risk_Accuracy'] = risk_accuracy
print(f"     ✅ Risk Model Accuracy: {risk_accuracy:.3f}")

# 4.4 Alert Classification Model
print("  🚨 Training alert classification model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Critical_Alert'], test_size=0.2, random_state=42)

alert_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
alert_model.fit(X_train, y_train)
alert_pred = alert_model.predict(X_test)
alert_accuracy = accuracy_score(y_test, alert_pred)

models['alert'] = alert_model
results['Alert_Accuracy'] = alert_accuracy
print(f"     ✅ Alert Model Accuracy: {alert_accuracy:.3f}")

# ============================================================================
# STEP 5: Feature Importance Analysis
# ============================================================================

print("\n📊 Analyzing feature importance...")

# Get feature importance from SOC model
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': models['soc'].feature_importances_
}).sort_values('importance', ascending=False)

print("🎯 Top 10 most important features for SOC prediction:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# ============================================================================
# STEP 6: Save Models and Results
# ============================================================================

print("\n💾 Saving models...")

# Save models
for name, model in models.items():
    joblib.dump(model, f'models/{name}_model.pkl')
    print(f"  ✅ Saved {name}_model.pkl")

# Save scaler and encoders
joblib.dump(scaler, 'models/scaler.pkl')
print(f"  ✅ Saved scaler.pkl")

for name, encoder in encoders.items():
    joblib.dump(encoder, f'models/{name}_encoder.pkl')
    print(f"  ✅ Saved {name}_encoder.pkl")

# Save processed dataset
df.to_csv('data/processed_dataset_with_targets.csv', index=False)
print(f"  ✅ Saved processed dataset")

# Save feature importance
feature_importance.to_csv('models/feature_importance.csv', index=False)
print(f"  ✅ Saved feature importance")

# ============================================================================
# STEP 7: Create Simple Prediction Function
# ============================================================================

print("\n🔮 Creating prediction function...")

def predict_battery_health(vehicle_data):
    """
    Predict battery health metrics for new data
    
    vehicle_data: dict with sensor readings
    Returns: dict with predictions
    """
    # This is a template - would need actual data preprocessing
    soc_pred = models['soc'].predict([vehicle_data])[0]
    soh_pred = models['soh'].predict([vehicle_data])[0]
    risk_pred = models['risk'].predict([vehicle_data])[0]
    
    return {
        'SOC': soc_pred,
        'SOH': soh_pred,
        'Risk_Level': risk_pred,
        'Timestamp': pd.Timestamp.now().isoformat()
    }

# Test prediction with sample data
sample_data = X_scaled[0].reshape(1, -1)
sample_pred = {
    'SOC': models['soc'].predict(sample_data)[0],
    'SOH': models['soh'].predict(sample_data)[0],
    'Risk': models['risk'].predict(sample_data)[0]
}

print("🧪 Sample prediction:")
print(f"  SOC: {sample_pred['SOC']:.1f}%")
print(f"  SOH: {sample_pred['SOH']:.1f}%")
print(f"  Risk: {sample_pred['Risk']}")

# ============================================================================
# STEP 8: Final Results Summary
# ============================================================================

print("\n" + "="*50)
print("🎯 ML TRAINING COMPLETE!")
print("="*50)

print("📊 Model Performance:")
for metric, value in results.items():
    if 'MAE' in metric:
        print(f"  {metric}: {value:.2f}%")
    else:
        print(f"  {metric}: {value:.3f}")

print(f"\n📁 Saved Files:")
print(f"  - 4 trained models in models/")
print(f"  - Feature scaler and encoders")
print(f"  - Processed dataset with ML targets")
print(f"  - Feature importance analysis")

print(f"\n🎯 Key Insights:")
print(f"  - SOC prediction accuracy: ±{results['SOC_MAE']:.1f}%")
print(f"  - SOH estimation accuracy: ±{results['SOH_MAE']:.1f}%")
print(f"  - Risk classification: {results['Risk_Accuracy']*100:.1f}% accurate")
print(f"  - Alert detection: {results['Alert_Accuracy']*100:.1f}% accurate")

print(f"\n🚀 Ready for dashboard integration!")
print(f"📋 Next steps:")
print(f"  1. Load models with: joblib.load('models/soc_model.pkl')")
print(f"  2. Make predictions on new sensor data")
print(f"  3. Integrate with real-time dashboard")

print("\n✅ ML pipeline completed successfully! 🎉")