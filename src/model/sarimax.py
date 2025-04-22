import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
GRANGER_CSV = "granger_all_tickers.csv"
INPUT_CSV = "stock_data_final_transformed.csv"
RESULT_CSV = "sarimax_results.csv"
ARTIFACT_FOLDER = "artifacts"
PLOT_PNG = "sarimax_all_tickers.png"

# ---------- PATH HELPERS ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename):
    return os.path.join(get_project_root(), DATA_FOLDER, filename)

def get_artifact_path(filename):
    os.makedirs(os.path.join(get_project_root(), ARTIFACT_FOLDER), exist_ok=True)
    return os.path.join(get_project_root(), ARTIFACT_FOLDER, filename)

# ---------- DATA LOADING ----------
def load_data():
    df = pd.read_csv(get_data_path(INPUT_CSV), parse_dates=['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    df['row'] = df.groupby('Ticker').cumcount()
    return df[df['row'] >= 30].drop(columns='row')

def load_granger_and_features():
    granger_df = pd.read_csv(get_data_path(GRANGER_CSV))
    top_10 = granger_df['Ticker'].unique()[:10]
    ticker_feature_map = {
        ticker: [
            f.replace('_stationary', '') + '_scaled'
            for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
        ]
        for ticker in top_10
    }
    return top_10, ticker_feature_map

# ---------- FORECASTING ----------
def forecast_ticker(df_ticker, ticker, features):
    df_ticker = df_ticker[['Date', 'Close'] + features].dropna().copy()
    df_ticker.set_index('Date', inplace=True)

    train_size = int(len(df_ticker) * 0.8)
    train, test = df_ticker.iloc[:train_size], df_ticker.iloc[train_size:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(train['Close'], exog=train[features], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)

        pred = model_fit.predict(
            start=len(train),
            end=len(train) + len(test) - 1,
            exog=test[features]
        )

    mse = mean_squared_error(test['Close'], pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test['Close'], pred)
    r2 = r2_score(test['Close'], pred)

    return {
        'Ticker': ticker,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'df': df_ticker.reset_index(),
        'features': features
    }

# ---------- COMBINED PLOT ----------
def save_combined_plot(forecast_results):
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    axs = axs.flatten()

    for idx, result in enumerate(forecast_results):
        if 'Error' in result:
            continue

        ticker = result['Ticker']
        df_ticker = result['df']
        features = result['features']

        df_ticker.set_index('Date', inplace=True)

        train_size = int(len(df_ticker) * 0.8)
        train, test = df_ticker.iloc[:train_size], df_ticker.iloc[train_size:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(train['Close'], exog=train[features], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            pred = model_fit.predict(
                start=len(train),
                end=len(train) + len(test) - 1,
                exog=test[features]
            )

        ax = axs[idx]
        ax.plot(train.index, train['Close'], label='Train')
        ax.plot(test.index, test['Close'], label='Actual', color='black')
        ax.plot(test.index, pred, label='Predicted', color='orange')
        ax.set_title(ticker)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = get_artifact_path(PLOT_PNG)
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Combined forecast saved to: {plot_path}")

# ---------- MAIN ----------
def main():
    df = load_data()
    top_10_tickers, ticker_feature_map = load_granger_and_features()

    forecast_results = []
    for ticker in top_10_tickers:
        features = ticker_feature_map.get(ticker, [])
        if not features:
            continue
        df_ticker = df[df['Ticker'] == ticker]
        try:
            result = forecast_ticker(df_ticker, ticker, features)
            forecast_results.append(result)
        except Exception as e:
            forecast_results.append({'Ticker': ticker, 'Error': str(e)})

    # Save metrics
    result_df = pd.DataFrame([
        {k: v for k, v in res.items() if k in ['Ticker', 'RMSE', 'MAE', 'R2', 'Error']} for res in forecast_results
    ])
    result_df.to_csv(get_data_path(RESULT_CSV), index=False)
    print(f"✅ Forecasting results saved to: {get_data_path(RESULT_CSV)}")
    print(result_df)

    # Save combined plot
    save_combined_plot(forecast_results)

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()

