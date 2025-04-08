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
import datetime
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Climate Trend Forecasting",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS for modern weather app styling
st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: #7B9AE0;
        background: linear-gradient(180deg, #7B9AE0 0%, #A8C4FF 100%);
        color: white;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        padding: 0;
        margin: 0;
    }
    
    .stApp {
        margin: 0 auto;
    }
    
    /* Header styling */
    .weather-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
    }
    
    .app-title {
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .current-time {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .current-date {
        font-size: 1rem;
        text-align: right;
    }
    
    /* Main weather display */
    .weather-status {
        font-size: 3.5rem;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 0;
    }
    
    .weather-desc {
        font-size: 1.8rem;
        font-weight: 500;
        margin-top: 0;
        margin-bottom: 20px;
    }
    
    .weather-info {
        font-size: 0.9rem;
        opacity: 0.9;
        max-width: 350px;
        line-height: 1.4;
    }
    
    /* Temperature styling */
    .temperature {
        font-size: 8rem;
        font-weight: 700;
        text-align: right;
        margin-right: 20px;
    }
    
    .temp-details {
        font-size: 0.9rem;
        text-align: right;
    }
    
    /* City comparison */
    .city-compare {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin-top: 20px;
    }
    
    .city-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Temperature graph */
    .temp-graph {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        margin-top: 30px;
    }
    
    /* Remove default streamlit padding and margins */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 5px 15px;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #7B9AE0;
    }

    /* Column styling */
    [data-testid="column"] {
        padding: 0 !important;
    }
    
    /* Card styling */
    .temp-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px 15px;
        text-align: center;
        margin: 5px;
    }
    
    /* Overwrite some streamlit element styles */
    .stSlider > div > div > div {
        background-color: white;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    </style>
    """, unsafe_allow_html=True)

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

# Function to create temperature trends by city
def create_city_data():
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
    cities = {
        'New York': {'base': 15, 'amplitude': 15, 'trend': 0.5, 'noise': 0.8},
        'Pennsylvania': {'base': 14, 'amplitude': 16, 'trend': 0.4, 'noise': 0.7},
        'Massachusetts': {'base': 13, 'amplitude': 17, 'trend': 0.6, 'noise': 0.9},
        'North Carolina': {'base': 18, 'amplitude': 12, 'trend': 0.3, 'noise': 0.6}
    }
    
    city_data = {}
    for city, params in cities.items():
        np.random.seed(42 + list(cities.keys()).index(city))
        base_temp = np.linspace(params['base'], params['base'] + params['trend'], len(dates))
        seasonal = np.sin(np.linspace(0, 2*np.pi*10, len(dates))) * params['amplitude']
        noise = np.random.normal(0, params['noise'], len(dates))
        city_data[city] = base_temp + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        **{city: temps for city, temps in city_data.items()}
    })

# Load data
df = load_data()
city_df = create_city_data()

# Define current date and time
now = datetime.datetime.now()
current_time = now.strftime("%H:%M")
current_date = now.strftime("%d %b")
current_day = now.strftime("%a")

# Weather conditions
weather_status = "Stormy"
weather_desc = "with Heavy Rain"
weather_info = "Variable clouds with snow showers. High 11F. Winds E at 10 to 20 mph. Chance of snow 50%. Snow accumulations less than one inch."
current_temp = 27
humidity = 93.8
uv_index = "0 OF 10"
wind_speed = "WSW 6 MPH"
location = "NEW YORK CITY"

# Header layout with logo, time and date
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="weather-header"><div class="app-title">☁️ Weather Forecast</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="current-time">{current_time}<span style="font-size:0.8rem;opacity:0.7;"> {now.strftime("%p")}</span></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="current-date">{current_date}<br><span style="opacity:0.7;">{current_day}</span></div>', unsafe_allow_html=True)

# Main tab navigation
tab1, tab2, tab3 = st.tabs(["📊 Forecast", "📈 Trends", "🔍 Analysis"])

with tab1:
    # Main weather display
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f'<h1 class="weather-status">{weather_status}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="weather-desc">{weather_desc}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p class="weather-info">{weather_info}</p>', unsafe_allow_html=True)
        
        # Temperature cards for next days
        cols = st.columns(3)
        temps = [12, 17, 14]
        labels = ["Today", "Tomorrow", "After"]
        for i, col in enumerate(cols):
            with col:
                st.markdown(f'''
                <div class="temp-card">
                    <div style="font-size:1.8rem;font-weight:600;">{temps[i]}°</div>
                    <div style="opacity:0.8;font-size:0.9rem;">{labels[i]}</div>
                </div>
                ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="temperature">{current_temp}°</div>', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="temp-details">
            <div style="margin-bottom:10px;">WIND: {wind_speed}</div>
            <div style="margin-bottom:10px;">UV INDEX: {uv_index}</div>
            <div style="margin-bottom:10px;">HUMIDITY: {humidity}%</div>
            <div style="margin-bottom:10px;">{location}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # City comparison section
    st.markdown('<div class="city-compare">', unsafe_allow_html=True)
    
    # Most recent temperatures for each city
    last_date = city_df['date'].iloc[-1]
    for city in ['Pennsylvania', 'Massachusetts', 'New York', 'North Carolina']:
        temp = int(city_df[city].iloc[-1])
        st.markdown(f'''
        <div class="city-row">
            <div>{city} [{city[:2].upper()}]</div>
            <div>{temp}°</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Temperature graph
    st.markdown('<div class="temp-graph">', unsafe_allow_html=True)
    
    # Get last week of data for the temperature curve
    recent_df = city_df.iloc[-30:].copy()
    fig = go.Figure()
    city_name = 'New York'
    fig.add_trace(go.Scatter(
        x=recent_df['date'], 
        y=recent_df[city_name],
        mode='lines',
        line=dict(color='white', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 255, 255, 0.1)'
    ))
    
    # Add markers for high and low points
    high_point = recent_df[city_name].max()
    high_idx = recent_df[city_name].idxmax()
    low_point = recent_df[city_name].min()
    low_idx = recent_df[city_name].idxmin()
    
    fig.add_trace(go.Scatter(
        x=[recent_df['date'].iloc[high_idx - recent_df.index[0]]],
        y=[high_point],
        mode='markers+text',
        marker=dict(color='white', size=10),
        text=[f"HIGH {high_point:.1f}°C"],
        textposition="top center"
    ))
    
    fig.add_trace(go.Scatter(
        x=[recent_df['date'].iloc[low_idx - recent_df.index[0]]],
        y=[low_point],
        mode='markers+text',
        marker=dict(color='white', size=10),
        text=[f"LOW {low_point:.1f}°C"],
        textposition="bottom center"
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        showlegend=False,
        height=200
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("Temperature Trend Analysis")
    
    # City selector
    selected_city = st.selectbox(
        "Select City",
        ["New York", "Pennsylvania", "Massachusetts", "North Carolina"]
    )
    
    # Plot trend data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df['date'], y=city_df[selected_city], name='Temperature'))
    
    # Add moving average
    window = st.slider("Moving Average Window (months)", 1, 24, 12)
    city_df[f'{selected_city}_moving_avg'] = city_df[selected_city].rolling(window=window).mean()
    fig.add_trace(go.Scatter(
        x=city_df['date'], 
        y=city_df[f'{selected_city}_moving_avg'], 
        name=f'{window}-month Moving Average',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=f"{selected_city} Historical Temperature Data",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        paper_bgcolor='rgba(255,255,255,0.1)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Climate Analysis")
    
    # Tabs for different analyses
    analysis_tab1, analysis_tab2 = st.tabs(["Seasonal Decomposition", "Forecast"])
    
    with analysis_tab1:
        selected_city_decomp = st.selectbox(
            "Select City for Decomposition",
            ["New York", "Pennsylvania", "Massachusetts", "North Carolina"],
            key="decomp_city"
        )
        
        # Perform seasonal decomposition
        city_series = pd.Series(city_df[selected_city_decomp].values, index=city_df['date'])
        decomposition = seasonal_decompose(city_series, period=12)
        
        # Create subplots
        fig = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        
        # Add traces
        fig.add_trace(go.Scatter(x=city_df['date'], y=city_series, name="Observed"), row=1, col=1)
        fig.add_trace(go.Scatter(x=city_df['date'], y=decomposition.trend, name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=city_df['date'], y=decomposition.seasonal, name="Seasonal"), row=3, col=1)
        fig.add_trace(go.Scatter(x=city_df['date'], y=decomposition.resid, name="Residual"), row=4, col=1)
        
        fig.update_layout(
            height=800, 
            template="plotly_white",
            paper_bgcolor='rgba(255,255,255,0.1)',
            plot_bgcolor='rgba(255,255,255,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with analysis_tab2:
        selected_city_forecast = st.selectbox(
            "Select City for Forecast",
            ["New York", "Pennsylvania", "Massachusetts", "North Carolina"],
            key="forecast_city"
        )
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': city_df['date'],
            'y': city_df[selected_city_forecast]
        })
        
        # Fit Prophet model
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_df)
        
        # Create future dataframe
        future_years = st.slider("Years to forecast", 1, 10, 5, key="forecast_slider")
        future = model.make_future_dataframe(periods=future_years*12, freq='M')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'], 
            y=prophet_df['y'], 
            name='Historical',
            line=dict(color='#A8C4FF')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'], 
            name='Forecast',
            line=dict(color='white', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(255,255,255,0.2)',
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,255,255,0.2)',
            name='Lower Bound'
        ))
        
        fig.update_layout(
            title=f"{selected_city_forecast} Temperature Forecast",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            paper_bgcolor='rgba(255,255,255,0.1)',
            plot_bgcolor='rgba(255,255,255,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('<div style="text-align:center;opacity:0.7;margin-top:20px;">Climate Trend Forecasting | Interactive Weather Analytics | v1.0</div>', unsafe_allow_html=True) 