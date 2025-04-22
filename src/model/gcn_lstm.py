import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear, LSTM
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INDUSTRY_FOLDER = "data/industry"
CSV_STOCK = "stock_data_final_transformed.csv"
CSV_GRANGER = "granger_all_tickers.csv"
CSV_INDUSTRY = "Ticker-Industry.csv"
CSV_TICKER_ERROR = "per_ticker_predictions.csv"
ARTIFACT_FOLDER = "artifacts"
PLOT_FILE = "gnn_vs_actual_prices_hd.png"

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

# ---------- DATA LOAD ----------
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

# ---------- GCN + LSTM Hybrid MODEL ----------
class GNN_LSTM(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 16)
        self.lstm = LSTM(input_size=16, hidden_size=8, batch_first=True)
        self.out = Linear(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, features)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        return self.out(x).squeeze()

# ---------- TRAINING ----------
def train_gcn(graph_data, epochs=100):
    model = GNN_LSTM(graph_data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = loss_fn(out, graph_data.y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model

# ---------- EVALUATION ----------
def evaluate(model, graph_data):
    model.eval()
    with torch.no_grad():
        pred = model(graph_data)
        y_true, y_pred = graph_data.y.numpy(), pred.numpy()
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"\n📈 GNN Forecast Results:\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return pred

# ---------- MAIN ----------
def main():
    df, granger_df = load_data()
    # Build graph manually
    tickers = granger_df['Ticker'].unique()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    industry_encoded = encoder.fit_transform(df[['Industry']].fillna('Unknown'))
    industry_encoded_df = pd.DataFrame(industry_encoded, columns=encoder.get_feature_names_out(['Industry']))
    df = pd.concat([df.reset_index(drop=True), industry_encoded_df], axis=1)

    ticker_features_map = {
        ticker: [
            f.replace('_stationary', '') + '_scaled'
            for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
            if f.replace('_stationary', '') + '_scaled' in df.columns
        ] + ['time_index_scaled', 'Close_scaled_diff'] + list(industry_encoded_df.columns)
        for ticker in tickers
    }

    all_features = sorted(set(f for feats in ticker_features_map.values() for f in feats))
    node_features, node_labels, ticker_index_map = [], [], {}

    for ticker in tickers:
        sub_df = df[df['Ticker'] == ticker].dropna()
        if sub_df.empty:
            continue
        last_row = sub_df.iloc[-1]
        feat_vals = [last_row.get(f, 0.0) for f in all_features]
        if pd.isna(last_row['Close_scaled']):
            continue
        x = torch.tensor(feat_vals, dtype=torch.float)
        y = torch.tensor([last_row['Close_scaled']], dtype=torch.float)
        ticker_index_map[ticker] = len(node_features)
        node_features.append(x)
        node_labels.append(y)

    valid_tickers = list(ticker_index_map.keys())
    edges = [
        [ticker_index_map[t1], ticker_index_map[t2]]
        for t1 in valid_tickers for t2 in valid_tickers
        if t1 != t2 and set(ticker_features_map[t1]) & set(ticker_features_map[t2])
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    graph_data = Data(x=torch.stack(node_features), edge_index=edge_index, y=torch.stack(node_labels).squeeze())

    model = train_gcn(graph_data)
    pred = evaluate(model, graph_data)

if __name__ == "__main__":
    main()
