#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Uninstall existing numpy and scipy to avoid conflicts
pip uninstall -y numpy scipy

# Install requirements
pip install -r requirements.txt

echo "Setup complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo "Then run the application with:"
echo "streamlit run app.py" 