import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="Climate Trend Forecasting",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title {
        color: #1f77b4;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🌡️ Climate Trend Forecasting")
st.markdown("""
    This application analyzes historical temperature data to detect climate change trends over time.
    Explore temperature trends, seasonal patterns, and future forecasts.
    """)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Function to load and preprocess data
@st.cache_data
def load_data():
    # For now, we'll use a sample dataset
    # In a real application, you would load your actual dataset here
    dates = pd.date_range(start='1900-01-01', end='2023-12-31', freq='M')
    np.random.seed(42)
    base_temp = np.linspace(13, 15, len(dates))  # Simulated warming trend
    seasonal = np.sin(np.linspace(0, 2*np.pi*20, len(dates))) * 5  # Seasonal variation
    noise = np.random.normal(0, 0.5, len(dates))  # Random noise
    temperature = base_temp + seasonal + noise
    
    df = pd.DataFrame({
        'ds': dates,
        'y': temperature
    })
    return df

# Load data
df = load_data()

# Sidebar for user controls
st.sidebar.title("Controls")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Trend Analysis", "Seasonal Decomposition", "Forecast"]
)

# Main content area
if analysis_type == "Trend Analysis":
    st.header("Temperature Trend Analysis")
    
    # Plot raw data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Temperature'))
    
    # Add moving average
    window = st.slider("Moving Average Window (months)", 1, 24, 12)
    df['moving_avg'] = df['y'].rolling(window=window).mean()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['moving_avg'], name=f'{window}-month Moving Average'))
    
    fig.update_layout(
        title="Historical Temperature Data",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Seasonal Decomposition":
    st.header("Seasonal Decomposition")
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df['y'], period=12)
    
    # Create subplots
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    
    # Add traces
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['ds'], y=decomposition.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['ds'], y=decomposition.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['ds'], y=decomposition.resid, name="Residual"), row=4, col=1)
    
    fig.update_layout(height=1000, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

else:  # Forecast
    st.header("Temperature Forecast")
    
    # Fit Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    
    # Create future dataframe
    future_years = st.slider("Years to forecast", 1, 10, 5)
    future = model.make_future_dataframe(periods=future_years*12, freq='M')
    
    # Make predictions
    forecast = model.predict(future)
    
    # Plot forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='Upper Bound'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='Lower Bound'
    ))
    
    fig.update_layout(
        title="Temperature Forecast",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Climate Trend Forecasting App | Data Source: Simulated Dataset") 