import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pytz

@st.cache_resource
def load_models():
    """Load all models"""
    try:
        models = {
            'solar': pickle.load(open('solar_model.pkl', 'rb')),
            'wind': pickle.load(open('wind_model.pkl', 'rb')),
            'demand': pickle.load(open('demand_model.pkl', 'rb'))
        }
        st.success("✅ Models loaded successfully")
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_data(ttl="1h")
def get_weather_data(start_date, _lat=42.3601, _lon=-71.0589):
    """Get weather data"""
    try:
        start = pd.to_datetime(start_date)
        end = start + timedelta(days=1)
        location = Point(_lat, _lon)
        data = Hourly(location, start, end)
        data = data.fetch()
        
        if data.empty:
            return None
            
        # Rename columns
        data = data.rename(columns={
            'temp': 'temperature',
            'dwpt': 'dwpt',
            'rhum': 'humidity',
            'prcp': 'precipitation',
            'wdir': 'wdir',
            'wspd': 'windspeed',
            'pres': 'pres',
            'coco': 'cloudcover'
        })
        
        data = data.reset_index()
        data = data.rename(columns={'time': 'datetime'})
        return data
    except Exception as e:
        st.error(f"Error getting weather data: {str(e)}")
        return None

@st.cache_data
def prepare_features(weather_data):
    """Prepare features for prediction"""
    features = weather_data[['temperature', 'dwpt', 'humidity', 'precipitation',
                           'wdir', 'windspeed', 'pres', 'cloudcover']]

    data = weather_data.copy()
    data['hour'] = data['datetime'].dt.hour
    data['month'] = data['datetime'].dt.month
    data['season'] = np.where(data['datetime'].dt.month.isin([12, 1, 2]), 1,
                    np.where(data['datetime'].dt.month.isin([3, 4, 5]), 2,
                    np.where(data['datetime'].dt.month.isin([6, 7, 8]), 3, 4)))
    data['time_of_day'] = np.where(data['datetime'].dt.hour < 6, 1,
                          np.where(data['datetime'].dt.hour < 12, 2,
                          np.where(data['datetime'].dt.hour < 18, 3, 4)))

    return pd.concat([features, data[['hour', 'month', 'season', 'time_of_day']]], axis=1)

def main():
    st.set_page_config(page_title="Energy Forecast", layout="wide")
    st.title("⚡ Energy Generation Forecast")

    # Load models
    models = load_models()
    if not models:
        st.stop()

    # Sidebar
    st.sidebar.header("Settings")
    
    # Date/time selection
    date = st.sidebar.date_input("Select date", datetime.now())
    time = st.sidebar.time_input("Select time", datetime.strptime('00:00', '%H:%M').time())
    start_datetime = datetime.combine(date, time)

    # Get data and make predictions
    with st.spinner('Getting forecast...'):
        weather_data = get_weather_data(start_datetime)
        if weather_data is None:
            st.error("Could not get weather data")
            st.stop()
            
        X_pred = prepare_features(weather_data)
        predictions = {'datetime': weather_data['datetime']}
        
        for name, model in models.items():
            predictions[name] = model.predict(X_pred)
            
        predictions = pd.DataFrame(predictions)

    # Display predictions
    st.header("Forecasts")
    
    # Create plot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=predictions['datetime'], y=predictions['solar'],
                            name='Solar', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=predictions['datetime'], y=predictions['wind'],
                            name='Wind', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions['datetime'], y=predictions['demand'],
                            name='Demand', line=dict(color='red')))

    fig.update_layout(
        title="Energy Generation and Demand Forecast",
        xaxis_title="Time",
        yaxis_title="MWh",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show raw data
    if st.checkbox("Show raw data"):
        st.dataframe(predictions)

if __name__ == "__main__":
    main()