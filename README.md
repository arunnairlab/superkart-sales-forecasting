# SuperKart Sales Forecasting

Predicts `Product_Store_Sales_Total` for SuperKart retail products using a tuned
Random Forest model, served via a Flask backend and a Streamlit frontend.

## Project layout

```
backend/            Flask REST API (/v1/predict, /v1/predictbatch)
frontend/            Streamlit UI that calls the backend
train.py             Trains the model and writes backend/superkart_model.joblib
SuperKart.csv         Training data
```

## Running in GitHub Codespaces

1. Open a Codespace on this repo (Code -> Codespaces -> Create codespace on main).
2. Train the model (the `.joblib` file is not committed to git — it's generated locally):
   ```
   pip install -r backend/requirements.txt
   python train.py
   ```
3. Build and run both services on a shared Docker network so the frontend can reach the backend by name:
   ```
   docker network create superkart-net

   docker build -t superkart-backend ./backend
   docker run -d --name backend --network superkart-net -p 7860:7860 superkart-backend

   docker build -t superkart-frontend ./frontend
   docker run -d --name frontend --network superkart-net -p 8501:8501 superkart-frontend
   ```
4. In the Codespace **Ports** tab, forward ports **7860** and **8501**, set visibility to **Public**, and open the 8501 URL to use the app.

## Local development (outside Codespaces)

Same steps as above, just run them on your machine with Docker installed.
