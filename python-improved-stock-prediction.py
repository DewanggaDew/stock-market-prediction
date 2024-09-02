import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings

warnings.filterwarnings("ignore")


def fetch_data(symbol="BTC-USD", period="5y"):
    print(f"Fetching data for {symbol}...")
    btc = yf.Ticker(symbol)
    data = btc.history(period=period)
    return data


def add_features(data):
    print("Adding technical analysis features...")
    data = add_all_ta_features(
        data,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )

    # Add custom features
    data["MA_50_200"] = (
        data["Close"].rolling(50).mean() / data["Close"].rolling(200).mean()
    )

    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()

    rsi = RSIIndicator(data["Close"])
    data["RSI"] = rsi.rsi()

    bb = BollingerBands(data["Close"])
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()

    # Add price change percentage
    data["Price_Change"] = data["Close"].pct_change()

    # Add rolling volatility
    data["Volatility"] = data["Close"].rolling(window=20).std()

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    return data.dropna()


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length][-1])  # Target is the last column
    return np.array(X), np.array(y)


def create_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_ensemble_model():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    return [rf, xgb]


def train_models(X, y, seq_length):
    # Prepare data for LSTM
    X_seq, y_seq = create_sequences(X, seq_length)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    # LSTM
    lstm_model = create_lstm_model((seq_length, X.shape[1]))
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    lstm_model.fit(
        X_train_seq,
        y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Ensemble
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    ensemble_models = create_ensemble_model()
    for model in ensemble_models:
        model.fit(X_train, y_train)

    return lstm_model, ensemble_models


def hybrid_predict(lstm_model, ensemble_models, X, seq_length):
    X_seq = X[-seq_length:].reshape(1, seq_length, -1)
    lstm_prob = lstm_model.predict(X_seq, verbose=0)[0][0]

    ensemble_probs = [
        model.predict_proba(X[-1].reshape(1, -1))[0][1] for model in ensemble_models
    ]
    ensemble_prob = np.mean(ensemble_probs)

    # Weighted average, giving more weight to LSTM
    final_prob = 0.7 * lstm_prob + 0.3 * ensemble_prob
    return final_prob


def advanced_backtest(
    lstm_model,
    ensemble_models,
    data,
    seq_length,
    initial_balance=10000,
    risk_per_trade=0.02,
):
    portfolio_value = []
    btc_holdings = 0

    features = data.drop(
        ["Open", "High", "Low", "Close", "Volume", "Target"], axis=1
    ).values

    for i in range(seq_length, len(data)):
        current_price = data["Close"].iloc[i]

        # Calculate indicators
        ma_trend = data["MA_50_200"].iloc[i] > 1
        macd_trend = data["MACD"].iloc[i] > data["MACD_Signal"].iloc[i]
        rsi_oversold = data["RSI"].iloc[i] < 30
        rsi_overbought = data["RSI"].iloc[i] > 70
        bb_low = data["Close"].iloc[i] < data["BB_Low"].iloc[i]
        bb_high = data["Close"].iloc[i] > data["BB_High"].iloc[i]

        # Get hybrid prediction
        prediction = hybrid_predict(
            lstm_model, ensemble_models, features[: i + 1], seq_length
        )

        # Trading logic
        if prediction > 0.6 and ma_trend and macd_trend and (rsi_oversold or bb_low):
            # Strong buy signal
            trade_amount = min(
                initial_balance if not portfolio_value else portfolio_value[-1],
                (initial_balance if not portfolio_value else portfolio_value[-1])
                * risk_per_trade,
            )
            btc_to_buy = trade_amount / current_price
            btc_holdings += btc_to_buy
            new_value = (
                initial_balance if not portfolio_value else portfolio_value[-1]
            ) - trade_amount
        elif (
            prediction < 0.4
            and (not ma_trend or not macd_trend)
            and (rsi_overbought or bb_high)
        ):
            # Strong sell signal
            if btc_holdings > 0:
                sell_amount = btc_holdings * current_price
                new_value = (
                    initial_balance if not portfolio_value else portfolio_value[-1]
                ) + sell_amount
                btc_holdings = 0
            else:
                new_value = (
                    initial_balance if not portfolio_value else portfolio_value[-1]
                )
        else:
            # Hold
            new_value = (
                initial_balance if not portfolio_value else portfolio_value[-1]
            ) + btc_holdings * (current_price - data["Close"].iloc[i - 1])

        portfolio_value.append(new_value)

    # Add final BTC value to portfolio
    final_btc_value = btc_holdings * data["Close"].iloc[-1]
    portfolio_value[-1] += final_btc_value

    return portfolio_value


def plot_results(data, portfolio_value):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[60:], portfolio_value, label="Strategy Performance")
    plt.plot(
        data.index[60:],
        data["Close"][60:] * (portfolio_value[0] / data["Close"].iloc[60]),
        label="Buy and Hold",
    )
    plt.title("Backtesting Results: Optimized LSTM Strategy vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.yscale("log")
    plt.show()


# Main execution
data = fetch_data()
data = add_features(data)

X = data.drop(["Open", "High", "Low", "Close", "Volume", "Target"], axis=1)
y = data["Target"]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
seq_length = 60
lstm_model, ensemble_models = train_models(X_scaled, y, seq_length)


# Calculate performance metrics
portfolio_value = advanced_backtest(lstm_model, ensemble_models, data, seq_length)

# Calculate performance metrics
initial_balance = 10000
final_portfolio_value = portfolio_value[-1]
buy_and_hold_value = data["Close"].iloc[-1] * (initial_balance / data["Close"].iloc[60])

print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Buy and Hold Value: ${buy_and_hold_value:.2f}")
print(
    f"Strategy Performance: {((final_portfolio_value / buy_and_hold_value) - 1) * 100:.2f}%"
)

# Plot results
plot_results(data, portfolio_value)

# Make prediction for next day
last_data = X_scaled[-seq_length:]
prediction = hybrid_predict(lstm_model, ensemble_models, last_data, seq_length)
print(f"\nProbability of price increase for next day: {prediction:.4f}")
