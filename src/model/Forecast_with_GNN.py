import os
import pandas as pd
import numpy as np
import torch
import argparse
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INDUSTRY_FOLDER = "data/industry"
CSV_STOCK = "stock_data_final_transformed.csv"
CSV_GRANGER = "granger_all_tickers.csv"
CSV_INDUSTRY = "Ticker-Industry.csv"
ARTIFACT_FOLDER = "artifacts"

# ---------- PATH UTILS ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(file):
    return os.path.join(get_project_root(), DATA_FOLDER, file)

def get_industry_path(file):
    return os.path.join(get_project_root(), INDUSTRY_FOLDER, file)

def get_artifact_path(file):
    path = os.path.join(get_project_root(), ARTIFACT_FOLDER)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, file)

# ---------- MODEL ----------
class GAT(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, 16, heads=4, dropout=0.3)
        self.out = Linear(16 * 4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        return self.out(x).squeeze()

# ---------- DATA LOADER ----------
def load_data():
    df = pd.read_csv(get_data_path(CSV_STOCK), parse_dates=['Date'])
    granger_df = pd.read_csv(get_data_path(CSV_GRANGER))
    industry_df = pd.read_csv(get_industry_path(CSV_INDUSTRY))

    df = df.sort_values(['Ticker', 'Date'])
    df['row'] = df.groupby('Ticker').cumcount()
    df = df[df['row'] >= 30].drop(columns='row')
    df['Close_scaled'] = df.groupby('Ticker')['Close'].transform(lambda x: (x - x.mean()) / x.std())
    df['time_index'] = df.groupby('Ticker').cumcount()
    df['time_index_scaled'] = df.groupby('Ticker')['time_index'].transform(lambda x: (x - x.mean()) / x.std())
    df['Close_scaled_diff'] = df.groupby('Ticker')['Close_scaled'].diff()

    df = df.merge(industry_df, on='Ticker', how='left')
    return df, granger_df

# ---------- BUILD GRAPH ----------
def build_graph(df, granger_df, ticker):
    df = df[df['Ticker'] == ticker].copy()
    if df.empty:
        raise ValueError(f"Ticker {ticker} not found in dataset")

    granger_df = granger_df[granger_df['Ticker'] == ticker]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    industry_encoded = encoder.fit_transform(df[['Industry']].fillna('Unknown'))
    industry_encoded_df = pd.DataFrame(industry_encoded, columns=encoder.get_feature_names_out(['Industry']))
    df = pd.concat([df.reset_index(drop=True), industry_encoded_df], axis=1)

    feature_cols = [
        f.replace('_stationary', '') + '_scaled'
        for f in granger_df['Factor'].tolist()
        if f.replace('_stationary', '') + '_scaled' in df.columns
    ] + ['time_index_scaled', 'Close_scaled_diff'] + list(industry_encoded_df.columns)

    df = df.dropna(subset=feature_cols + ['Close'])
    last_row = df.iloc[-1]
    feat_vals = [last_row[f] for f in feature_cols]

    x = torch.tensor(feat_vals, dtype=torch.float).unsqueeze(0)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index), last_row['Close'], df, feature_cols, len(feat_vals)

# ---------- FORECAST ----------
def forecast_percent_return(df, ticker, model_path, days):
    data, last_price, full_df, feature_cols, input_size = build_graph(df, granger_df, ticker)

    checkpoint = torch.load(model_path)
    model = GAT(input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        pred_price_scaled = model(data).item()
        mean = df[df['Ticker'] == ticker]['Close'].mean()
        std = df[df['Ticker'] == ticker]['Close'].std()
        pred_price = pred_price_scaled * std + mean

    pct_return = (pred_price - last_price) / last_price * 100
    return {
        "Ticker": ticker,
        "DaysAhead": days,
        "ForecastReturn%": round(pct_return, 2)
    }

# ---------- MAIN ----------
def main(ticker, days):
    global df, granger_df
    df, granger_df = load_data()
    model_path = get_artifact_path("gnn_model.pt")
    forecast_result = forecast_percent_return(df, ticker=ticker, model_path=model_path, days=days)
    print("\n🔮 Forecast:", forecast_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol to forecast")
    parser.add_argument("--days", type=int, required=True, help="Number of days to forecast ahead")
    args = parser.parse_args()
    main(ticker=args.ticker, days=args.days)

