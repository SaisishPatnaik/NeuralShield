# NeuralShield

**AI-Powered Network Intrusion Detection and Monitoring Platform**

NeuralShield is a hybrid AI-powered network security platform designed to detect malicious network activity, identify potential attack categories, and monitor network traffic through an interactive dashboard.

## Features

- Binary network intrusion classification using XGBoost.
- Multiclass attack classification.
- Autoencoder-based anomaly detection.
- Hybrid threat detection.
- Risk scoring and alert generation.
- FastAPI backend.
- Streamlit security dashboard.
- MongoDB Atlas integration.
- Network monitoring using Scapy.

## Technology Stack

- Python
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- FastAPI
- Streamlit
- MongoDB Atlas
- Plotly
- Scapy

## Project Structure

```text
NeuralShield/
├── backend/
├── capture_engine/
├── dashboard/
├── data/
├── ml_engine/
├── models/
├── notebooks/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Clone the repository:

   `git clone https://github.com/SaisishPatnaik/NeuralShield.git`

2. Enter the project directory:

   `cd NeuralShield`

3. Create and activate a virtual environment:

   `python -m venv venv`

   Windows PowerShell:

   `.\venv\Scripts\Activate.ps1`

4. Install dependencies:

   `pip install -r requirements.txt`

5. Configure environment variables using `.env.example` as a template.

6. Start the backend:

   `uvicorn backend.main:app --reload`

7. Start the dashboard in another terminal:

   `streamlit run dashboard/app.py`

## API Documentation

Once the backend is running, visit:

`http://127.0.0.1:8000/docs`

## Dataset

The project uses the UNSW-NB15 network intrusion detection dataset for model development and evaluation.

## Security Notice

Use live network monitoring only on networks and devices for which you have authorization. NeuralShield is designed for passive monitoring and detection.

## Project Status

Under development.
