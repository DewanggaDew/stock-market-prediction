import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# Load the Bitcoin data
btc = yf.Ticker("BTC-USD")
btc = btc.history(period="max")
btc = btc.loc["2010-07-17":].copy()  # Bitcoin's first trading date

# Preprocess the data
btc = btc.dropna()
btc["Tomorrow"] = btc["Close"].shift(-1)
btc["Target"] = (btc["Tomorrow"] > btc["Close"]).astype(int)

# Calculate technical indicators
btc["RSI"] = calculate_rsi(btc["Close"], window=14)
btc["SMA_20"] = btc["Close"].rolling(window=20).mean()
btc["SMA_50"] = btc["Close"].rolling(window=50).mean()
btc["Volatility"] = btc["Close"].rolling(window=20).std()

# Create lag features
for lag in [1, 3, 5, 7, 14]:
    btc[f"Lag_{lag}"] = btc["Close"].shift(lag)

# Drop rows with NaN values
btc = btc.dropna()

# Select features
features = ["RSI", "SMA_20", "SMA_50", "Volatility"] + [
    f"Lag_{lag}" for lag in [1, 3, 5, 7, 14]
]
X = btc[features]
y = btc["Target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Down", "Up"])
plt.yticks(tick_marks, ["Down", "Up"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(
    btc.index[-len(y_test) :],
    btc["Close"][-len(y_test) :],
    label="Actual Price",
    alpha=0.7,
)
plt.scatter(
    btc.index[-len(y_test) :][y_pred == 1],
    btc["Close"][-len(y_test) :][y_pred == 1],
    color="green",
    label="Predicted Up",
    alpha=0.5,
)
plt.scatter(
    btc.index[-len(y_test) :][y_pred == 0],
    btc["Close"][-len(y_test) :][y_pred == 0],
    color="red",
    label="Predicted Down",
    alpha=0.5,
)
plt.title("Bitcoin Price with Predictions")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=False
)
plt.figure(figsize=(10, 6))
feature_importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
