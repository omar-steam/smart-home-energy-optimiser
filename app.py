# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from datetime import datetime, time

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('energy_optimizer_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

def main():
    st.title('Smart Home Energy Optimizer')
    st.write('Optimize your home energy usage with AI-driven insights')

    # Sidebar for input parameters
    st.sidebar.header('Home Environment Parameters')
    
    temperature = st.sidebar.slider('Temperature (°C)', 10.0, 35.0, 22.0)
    humidity = st.sidebar.slider('Humidity (%)', 30.0, 80.0, 50.0)
    time_of_day = st.sidebar.time_input('Time of Day', time(12, 0))
    day_of_week = st.sidebar.selectbox('Day of Week', 
                                      ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                       'Friday', 'Saturday', 'Sunday'])
    occupancy = st.sidebar.number_input('Number of Occupants', 0, 10, 2)
    appliance_usage = st.sidebar.slider('Appliance Usage Level', 0.0, 10.0, 5.0)

    # Convert inputs to model features
    day_mapping = {day: i for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 
                                                  'Thursday', 'Friday', 'Saturday', 'Sunday'])}
    
    features = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'time_of_day': [time_of_day.hour + time_of_day.minute/60],
        'day_of_week': [day_mapping[day_of_week]],
        'occupancy': [occupancy],
        'appliance_usage': [appliance_usage]
    })

    # Load model and make prediction
    model, scaler = load_model()
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Display results
    st.header('Energy Consumption Prediction')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Predicted Energy Consumption', f'{prediction:.2f} kWh')
    
    with col2:
        if prediction > 7:
            st.error('High Energy Usage Alert!')
            st.write('Consider reducing appliance usage or adjusting temperature.')
        elif prediction > 4:
            st.warning('Moderate Energy Usage')
            st.write('Your energy usage is within normal range.')
        else:
            st.success('Efficient Energy Usage')
            st.write('Great job maintaining low energy consumption!')

    # Energy optimization recommendations
    st.header('Optimization Recommendations')
    recommendations = []
    
    if temperature < 20 or temperature > 24:
        recommendations.append("Adjust temperature to between 20-24°C for optimal efficiency.")
    if appliance_usage > 7:
        recommendations.append("Consider spreading out appliance usage throughout the day.")
    if occupancy == 0 and appliance_usage > 2:
        recommendations.append("Reduce standby power consumption when home is unoccupied.")

    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("Your current settings are optimized for energy efficiency!")

    # Historical Usage Graph
    st.header('Simulated Daily Energy Usage Pattern')
    hours = np.arange(24)
    usage_pattern = np.sin(hours * np.pi / 12) * 3 + 5 + np.random.normal(0, 0.5, 24)
    
    chart_data = pd.DataFrame({
        'Hour': hours,
        'Energy Usage (kWh)': usage_pattern
    })
    
    st.line_chart(chart_data.set_index('Hour'))

if __name__ == '__main__':
    main()