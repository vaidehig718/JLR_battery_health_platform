import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_raw_battery_dataset(num_rows=1000):
    """
    Generate raw battery sensor dataset for ML model training
    """
    
    # Configuration
    vehicle_models = ['I-PACE', 'F-PACE_EV', 'RANGE_ROVER_EV', 'XF_EV', 'DISCOVERY_EV']
    locations = [
        {'city': 'Mumbai', 'lat': 19.076, 'lon': 72.8777, 'alt': 14, 'base_humidity': 70, 'base_pressure': 101.2},
        {'city': 'Delhi', 'lat': 28.6139, 'lon': 77.209, 'alt': 216, 'base_humidity': 60, 'base_pressure': 99.7},
        {'city': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946, 'alt': 920, 'base_humidity': 72, 'base_pressure': 92.0},
        {'city': 'Chennai', 'lat': 13.0827, 'lon': 80.2707, 'alt': 6, 'base_humidity': 75, 'base_pressure': 101.6},
        {'city': 'Pune', 'lat': 18.5204, 'lon': 73.8567, 'alt': 560, 'base_humidity': 65, 'base_pressure': 94.0}
    ]
    
    charging_states = ['Charging', 'Discharging', 'Idle']
    charging_probs = [0.15, 0.7, 0.15]  # Probabilities for each state
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(num_rows):
        # Time progression (every 6 hours)
        timestamp = start_date + timedelta(hours=6*i)
        
        # Vehicle assignment (50 readings per vehicle)
        vehicle_num = (i // 50) + 1001
        vehicle_id = f"JLR-EV-{vehicle_num:04d}"
        vehicle_model = vehicle_models[(i // 200) % len(vehicle_models)]
        
        # Battery age (1-60 months)
        battery_age_months = random.randint(1, 60)
        
        # Location selection
        location = random.choice(locations)
        
        # Environmental conditions with seasonal variation
        day_of_year = timestamp.timetuple().tm_yday
        seasonal_factor = np.sin((day_of_year * 2 * np.pi) / 365)
        
        ambient_temp = 20 + 15 * seasonal_factor + random.uniform(-5, 5)
        humidity = max(30, min(90, location['base_humidity'] + seasonal_factor * 10 + random.uniform(-10, 10)))
        air_pressure = location['base_pressure'] + random.uniform(-1, 1)
        
        # Charging state selection
        charging_state = np.random.choice(charging_states, p=charging_probs)
        
        # Current based on charging state
        if charging_state == 'Charging':
            pack_current = -(20 + random.uniform(0, 80))  # Negative for charging
        elif charging_state == 'Discharging':
            pack_current = 15 + random.uniform(0, 90)  # Positive for discharging
        else:  # Idle
            pack_current = random.uniform(-2, 2)  # Near zero
        
        # Temperature calculations (affected by current and ambient)
        current_heat = abs(pack_current) * 0.1
        avg_temperature = ambient_temp + current_heat + random.uniform(-4, 4)
        max_temperature = avg_temperature + 3 + random.uniform(0, 12)
        min_temperature = avg_temperature - 2 - random.uniform(0, 6)
        
        # Voltage calculations (affected by age and load)
        nominal_voltage = 3.6
        voltage_variation = random.uniform(-0.2, 0.2)
        age_effect = -battery_age_months * 0.001  # Slight degradation
        
        avg_cell_voltage = nominal_voltage + voltage_variation + age_effect
        max_cell_voltage = avg_cell_voltage + 0.1 + random.uniform(0, 0.3)
        min_cell_voltage = avg_cell_voltage - 0.1 - random.uniform(0, 0.3)
        
        # Internal resistance (increases with age and temperature)
        base_resistance = 2.8
        age_resistance = battery_age_months * 0.01
        temp_resistance = max(0, avg_temperature - 25) * 0.02
        internal_resistance = base_resistance + age_resistance + temp_resistance + random.uniform(-0.25, 0.25)
        
        # Usage metrics
        cycle_count = int(battery_age_months * 12 + random.uniform(0, 150))
        charge_throughput = cycle_count * 85 * random.uniform(0.8, 1.2)  # kWh
        
        # Depth of discharge (only when discharging)
        depth_of_discharge = random.uniform(20, 90) if charging_state == 'Discharging' else 0
        
        # Energy metrics
        power = abs(pack_current) * avg_cell_voltage / 1000  # kW
        energy_consumed = power * 6 if charging_state == 'Discharging' else 0  # kWh over 6 hours
        energy_regenerated = power * 6 if charging_state == 'Charging' else 0  # kWh over 6 hours
        
        # Driving metrics (only when discharging)
        if charging_state == 'Discharging':
            distance_traveled = random.uniform(0, 200)  # km
            speed = random.uniform(0, 120)  # km/h
            acceleration = random.uniform(-2.5, 2.5)  # m/s²
        else:
            distance_traveled = 0
            speed = 0
            acceleration = 0
        
        # Physical parameters
        pressure_level = 1.0 + random.uniform(0, 0.8)
        coolant_flow_rate = 2.0 + random.uniform(0, 2.0)
        insulation_resistance = random.uniform(1000000, 1800000)
        
        # Create row data
        row = {
            'Timestamp': timestamp.isoformat(),
            'Vehicle_ID': vehicle_id,
            'Vehicle_Model': vehicle_model,
            'Battery_Age_Months': battery_age_months,
            
            # Voltage measurements
            'Avg_Cell_Voltage': round(avg_cell_voltage, 3),
            'Max_Cell_Voltage': round(max_cell_voltage, 3),
            'Min_Cell_Voltage': round(min_cell_voltage, 3),
            
            # Current and charging
            'Pack_Current': round(pack_current, 2),
            'Charging_State': charging_state,
            'Charge_Throughput': round(charge_throughput, 2),
            
            # Temperature measurements
            'Avg_Temperature': round(avg_temperature, 2),
            'Max_Temperature': round(max_temperature, 2),
            'Min_Temperature': round(min_temperature, 2),
            'Ambient_Temperature': round(ambient_temp, 2),
            
            # Physical parameters
            'Internal_Resistance': round(internal_resistance, 3),
            'Pressure_Level': round(pressure_level, 3),
            'Coolant_Flow_Rate': round(coolant_flow_rate, 3),
            'Insulation_Resistance': int(insulation_resistance),
            
            # Usage metrics
            'Cycle_Count': cycle_count,
            'Depth_of_Discharge': round(depth_of_discharge, 1),
            
            # Energy metrics
            'Energy_Consumed_kWh': round(energy_consumed, 2),
            'Energy_Regenerated_kWh': round(energy_regenerated, 2),
            
            # Driving data
            'Distance_Traveled_km': round(distance_traveled, 1),
            'Speed_kmh': round(speed, 1),
            'Acceleration_ms2': round(acceleration, 1),
            
            # Location and environmental
            'Location_City': location['city'],
            'Location_Latitude': location['lat'],
            'Location_Longitude': location['lon'],
            'Altitude_m': location['alt'],
            'Humidity_Percent': int(humidity),
            'Air_Pressure_kPa': round(air_pressure, 1)
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

# Generate the dataset
print("Generating raw battery sensor dataset...")
df = generate_raw_battery_dataset(1000)

print(f"✅ Dataset generated successfully!")
print(f"📊 Shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")

# Save to CSV
df.to_csv('raw_battery_dataset_1000.csv', index=False)
print("💾 Saved as 'raw_battery_dataset_1000.csv'")

# Show sample data
print("\n🔍 Sample data:")
print(df.head())

# Basic statistics
print("\n📈 Dataset statistics:")
print("Charging state distribution:")
print(df['Charging_State'].value_counts())

print("\nVehicle model distribution:")
print(df['Vehicle_Model'].value_counts())

print("\nLocation distribution:")
print(df['Location_City'].value_counts())

print("\nTemperature statistics:")
print(df[['Avg_Temperature', 'Max_Temperature', 'Min_Temperature']].describe())

print("\nVoltage statistics:")
print(df[['Avg_Cell_Voltage', 'Max_Cell_Voltage', 'Min_Cell_Voltage']].describe())

print("\n🎯 Ready for ML model development!")
print("Next: Create target variables (SOC, SOH, risk scores) and train models")