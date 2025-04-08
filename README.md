# Climate Trend Forecasting

A Python-based application that analyzes historical temperature data to detect climate change trends over time. The project includes data cleaning, time series analysis, visualization, and an interactive user interface.

## Features

- Historical temperature data analysis
- Time series decomposition (trend, seasonality, noise)
- Temperature forecasting
- Interactive visualizations
- Modern, weather-app-like user interface

## Setup

1. Clone this repository
2. Run the setup script:
   ```bash
   ./setup.sh
   ```
   This will:
   - Create a virtual environment
   - Install all required dependencies with compatible versions
   - Set up the environment for running the application

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application file
- `data/`: Directory containing temperature datasets
- `requirements.txt`: Project dependencies
- `setup.sh`: Setup script for environment configuration
- `README.md`: Project documentation

## Data Source

The application uses historical temperature data from NOAA's Global Historical Climatology Network (GHCN). The dataset includes monthly average temperatures spanning several decades.

## Usage

1. Launch the application using the command above
2. Navigate through the different sections:
   - Trends: View historical temperature trends
   - Decomposition: Analyze trend, seasonality, and noise components
   - Forecast: View temperature predictions for the coming years

## Requirements

- Python 3.8 or higher
- Virtual environment (created automatically by setup.sh)
- Dependencies listed in requirements.txt 