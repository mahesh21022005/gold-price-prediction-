import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Download historical gold futures data
ticker = "GC=F"
df = yf.download(ticker, start="2015-01-01", end=None, auto_adjust=False)

if df.empty:
    raise RuntimeError("No data was downloaded. Check your internet connection and try again.")

# Handle yfinance column format
if isinstance(df.columns, pd.MultiIndex):
    close = df[("Close", ticker)]
else:
    close = df["Close"]

close = pd.Series(close).dropna().values.reshape(-1, 1)

# Scale prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close)

# Create sequences
look_back = 60
X, y = [], []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Train-test split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Reshape for LSTM: samples, time steps, features
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Train
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predict
predicted = model.predict(X_test, verbose=0)

# Convert back to original price scale
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)

print("\nGold Price Prediction Results")
print("-----------------------------")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Gold Price")
plt.plot(predicted_prices, label="Predicted Gold Price")
plt.title("Gold Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Gold Price")
plt.legend()
plt.tight_layout()
plt.show()
