import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
GRANGER_CSV = "granger_all_tickers.csv"
INPUT_CSV = "stock_data_final_transformed.csv"
RESULT_CSV = "lstm_results.csv"
ARTIFACT_FOLDER = "artifacts"
PLOT_PNG = "lstm_all_tickers.png"
LOOKBACK = 30
EPOCHS = 50
BATCH_SIZE = 16

# ---------- PATH HELPERS ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename):
    return os.path.join(get_project_root(), DATA_FOLDER, filename)

def get_artifact_path(filename):
    os.makedirs(os.path.join(get_project_root(), ARTIFACT_FOLDER), exist_ok=True)
    return os.path.join(get_project_root(), ARTIFACT_FOLDER, filename)

# ---------- DATA LOADING ----------
def load_and_prepare_data():
    df = pd.read_csv(get_data_path(INPUT_CSV), parse_dates=["Date"])
    granger_df = pd.read_csv(get_data_path(GRANGER_CSV))

    df = df.sort_values(["Ticker", "Date"])
    df["row"] = df.groupby("Ticker").cumcount()
    df = df[df["row"] >= 30].drop(columns="row")

    top_10_tickers = granger_df["Ticker"].unique()[:10]
    df = df[df["Ticker"].isin(top_10_tickers)]
    granger_df = granger_df[granger_df["Ticker"].isin(top_10_tickers)]

    ticker_features_map = {
        ticker: [
            f.replace("_stationary", "") + "_scaled"
            for f in granger_df[granger_df["Ticker"] == ticker]["Factor"].tolist()
            if f.replace("_stationary", "") + "_scaled" in df.columns
        ]
        for ticker in top_10_tickers
    }

    return df, top_10_tickers, ticker_features_map

# ---------- LSTM UTILS ----------
def prepare_lstm_data(df_ticker, features, target="Close_scaled", lookback=30):
    X, y = [], []
    for i in range(lookback, len(df_ticker)):
        X.append(df_ticker[features].iloc[i - lookback:i].values)
        y.append(df_ticker[target].iloc[i])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS,
              batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])
    return model

# ---------- FORECAST + COLLECT ----------
def forecast_and_collect(df, top_10_tickers, ticker_features_map):
    forecast_results = []

    for ticker in top_10_tickers:
        features = ticker_features_map.get(ticker, [])
        if not features:
            forecast_results.append({'Ticker': ticker, 'Error': 'No features'})
            continue

        df_ticker = df[df["Ticker"] == ticker].copy()
        df_ticker = df_ticker[["Date", "Close_scaled"] + features].dropna()

        if df_ticker.shape[0] < LOOKBACK + 10:
            forecast_results.append({'Ticker': ticker, 'Error': 'Insufficient data'})
            continue

        X, y = prepare_lstm_data(df_ticker, features, lookback=LOOKBACK)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        try:
            model = train_lstm_model(X_train, y_train, (X_train.shape[1], X_train.shape[2]))
            y_pred = model.predict(X_test).flatten()

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            forecast_results.append({
                "Ticker": ticker,
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
                "y_test": y_test,
                "y_pred": y_pred
            })
        except Exception as e:
            forecast_results.append({'Ticker': ticker, 'Error': str(e)})

    return forecast_results

# ---------- PLOTTING ----------
def plot_all_forecasts(forecast_results):
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    axs = axs.flatten()

    for idx, result in enumerate(forecast_results):
        if "Error" in result:
            continue
        ax = axs[idx]
        ax.plot(result["y_test"], label="Actual", color="black")
        ax.plot(result["y_pred"], label="Predicted", color="green")
        ax.set_title(result["Ticker"])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(get_artifact_path(PLOT_PNG))
    plt.close()
    print(f"📊 Combined LSTM forecast saved to: {get_artifact_path(PLOT_PNG)}")

# ---------- MAIN ----------
def main():
    warnings.filterwarnings("ignore")

    df, top_10_tickers, ticker_features_map = load_and_prepare_data()
    results = forecast_and_collect(df, top_10_tickers, ticker_features_map)

    # Save metrics only
    result_df = pd.DataFrame([
        {k: v for k, v in r.items() if k in ["Ticker", "RMSE", "MAE", "R2", "Error"]}
        for r in results
    ])
    result_df.to_csv(get_data_path(RESULT_CSV), index=False)
    print(f"✅ LSTM results saved to {get_data_path(RESULT_CSV)}")
    print(result_df)

    # Save combined plot
    plot_all_forecasts(results)

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()

