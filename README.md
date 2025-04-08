# Climate Trend Forecasting

A Python-based application that analyzes historical temperature data to detect climate change trends over time. The project includes data cleaning, time series analysis, visualization, and an interactive user interface.

## Features

- Historical temperature data analysis (1900-present)
- Time series decomposition (trend, seasonality, noise)
- Temperature forecasting
- Interactive visualizations
- Modern, weather-app-like user interface

## Architecture

The application follows a multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Presentation Layer                        │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Forecast   │    │    Trends   │    │      Analysis       │  │
│  │    Tab      │    │     Tab     │    │        Tab          │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        Analysis Layer                           │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Prophet    │    │ Time Series │    │    Statistical      │  │
│  │ Forecasting │    │Decomposition│    │     Analysis        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                         Data Layer                              │
│                                                                 │
│  ┌────────────────────┐                                         │
│  │ global_temperature │                                         │
│  │       .csv         │                                         │
│  └────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Observed Climate Trends

Analysis of our temperature dataset (1900-present) reveals several key climate patterns:

### 1. Long-term Temperature Evolution

- **Century-Scale Warming**: Temperature records show a progressive increase from baseline temperatures of ~13-14°C in the early 1900s
- **Warming Acceleration**: The rate of temperature increase has accelerated significantly since the 1970s compared to early 20th century rates

### 2. Seasonal Pattern Changes

- **Winter Warming**: Winter months (Dec-Feb) show more pronounced temperature increases than summer months
- **Seasonal Shift**: Timing of seasonal transitions (spring and fall) shows detectable shifts over the analyzed period

### 3. Anomaly Detection

- **Extreme Events**: The application identifies statistically significant temperature anomalies that exceed historical variability
- **Clustering of Extremes**: Analysis shows increasing frequency of temperature extremes in recent decades

### 4. Cyclical Patterns

- **Multi-Year Cycles**: The application detects and visualizes cyclical temperature patterns beyond annual seasonality
- **ENSO Effects**: El Niño and La Niña cycles and their impact on temperature patterns are highlighted

## Time Series Analysis Methodology

### Data Preprocessing

- Monthly temperature data starting from 1900
- Missing data imputation using linear interpolation
- Outlier detection and removal using statistical methods
- Resampling to ensure consistent monthly data points

### Decomposition Techniques

The application decomposes the 120+ years of temperature data into:

1. **Trend Component**: The long-term progression showing the century-scale warming pattern
2. **Seasonal Component**: Monthly temperature cycles revealing changing seasonal patterns
3. **Residual Component**: Random variations and anomalies after removing trend and seasonality

### Forecasting Models

Prophet forecasting models are configured with:

- Yearly seasonality to capture annual temperature cycles
- Change-point detection to identify significant shifts in trends
- Uncertainty intervals to quantify prediction confidence
- Historical fit evaluation using cross-validation techniques

## Setup

1. Clone this repository
2. Run the setup script:
   ```bash
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application file with UI components and analysis logic
- `data/`: Directory containing temperature datasets
  - `global_temperature.csv`: Monthly temperature records from 1900 to present
- `requirements.txt`: Project dependencies
- `setup.sh`: Setup script for environment configuration
- `README.md`: Project documentation

## Data Description

Our primary dataset (`global_temperature.csv`) contains:
- Monthly temperature readings starting from January 1900
- Simple two-column format with date and temperature values
- Temperature values represent global or regional averages in Celsius
- Continuous monthly data allowing for comprehensive time series analysis

## Usage

1. Launch the application using the command above
2. Navigate through the different sections:
   - **Forecast**: View current weather and projected temperatures
   - **Trends**: 
     - Historical temperature visualization from 1900
     - Adjustable moving averages (5-year, 10-year, 30-year)
     - Decade-by-decade comparison
   - **Analysis**: 
     - Seasonal Decomposition: Explore trend, seasonal and residual components
     - Anomaly Detection: Identify statistically significant temperature events
     - Forecast: Generate future temperature projections with confidence intervals

## Requirements

- Python 3.8 or higher
- Virtual environment (created automatically by setup.sh)
- Dependencies listed in requirements.txt