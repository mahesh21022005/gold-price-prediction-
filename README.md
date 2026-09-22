# Gold Price Prediction using LSTM

A beginner-friendly machine learning project that predicts gold prices using an LSTM (Long Short-Term Memory) neural network.

## Project Overview
This project:
- Downloads historical gold futures price data
- Preprocesses the closing price
- Scales the data using Min-Max Scaling
- Creates time-series sequences
- Trains an LSTM model
- Predicts gold prices
- Calculates MSE and RMSE
- Plots actual vs predicted prices

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- yfinance

## How to Run

```bash
pip install -r requirements.txt
python gold_price_prediction.py
```

The program downloads the data automatically, so a CSV dataset is not required.

## Model
The model uses previous gold-price observations to learn temporal patterns and predict the next value.

## Evaluation
The project reports:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Project Structure

```text
gold-price-prediction/
├── gold_price_prediction.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Note
This is an educational machine-learning project. Predictions are not financial advice.
