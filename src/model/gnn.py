import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
CSV_STOCK = "stock_data_final_transformed.csv"
CSV_GRANGER = "granger_all_tickers.csv"
ARTIFACT_FOLDER = "artifacts"
PLOT_FILE = "gnn_vs_actual_prices_hd.png"

# ---------- PATH UTILS ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(file):
    return os.path.join(get_project_root(), DATA_FOLDER, file)

def get_artifact_path(file):
    path = os.path.join(get_project_root(), ARTIFACT_FOLDER)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, file)

# ---------- DATA LOAD ----------
def load_data():
    df = pd.read_csv(get_data_path(CSV_STOCK), parse_dates=['Date'])
    granger_df = pd.read_csv(get_data_path(CSV_GRANGER))
    df = df.sort_values(['Ticker', 'Date'])
    df['row'] = df.groupby('Ticker').cumcount()
    df = df[df['row'] >= 30].drop(columns='row')
    return df, granger_df

def build_graph(df, granger_df):
    tickers = granger_df['Ticker'].unique()
    df = df[df['Ticker'].isin(tickers)]
    granger_df = granger_df[granger_df['Ticker'].isin(tickers)]

    ticker_features_map = {
        ticker: [
            f.replace('_stationary', '') + '_scaled'
            for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
            if f.replace('_stationary', '') + '_scaled' in df.columns
        ]
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

    graph_data = Data(x=torch.stack(node_features),
                      edge_index=edge_index,
                      y=torch.stack(node_labels).squeeze())
    return graph_data, ticker_index_map, df

# ---------- GCN MODEL ----------
class GCN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        return self.out(x).squeeze()

def train_gcn(graph_data, epochs=100):
    model = GCN(graph_data.num_node_features)
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

def plot_scaled_predictions(graph_data, pred):
    plt.figure(figsize=(10, 4))
    plt.plot(graph_data.y.numpy(), label='Actual', color='black')
    plt.plot(pred.numpy(), label='Predicted', color='green')
    plt.title("GNN Prediction - Close_scaled")
    plt.legend()
    plt.show()

def plot_unscaled_predictions(pred, graph_data, df, ticker_index_map):
    import matplotlib
    matplotlib.use("Agg")  # Use a backend suitable for saving without display
    
    tickers_used = list(ticker_index_map.keys())
    price_bounds = df.groupby("Ticker")["Close"].agg(["min", "max"]).rename(columns={"min": "Close_min", "max": "Close_max"})

    unscaled_preds, unscaled_actuals = [], []
    for idx, ticker in enumerate(tickers_used):
        min_price, max_price = price_bounds.loc[ticker]
        scale = max_price - min_price
        unscaled_preds.append(pred[idx].item() * scale + min_price)
        unscaled_actuals.append(graph_data.y[idx].item() * scale + min_price)

    # ✨ Reasonable and viewable size (not too big for SVG)
    num_tickers = len(tickers_used)
    fig_width = min(50, max(12, num_tickers * 0.8))  # don't go over 50 inches
    fig_height = max(8, num_tickers * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(unscaled_actuals, label="Actual ₹", marker='o', color='black', markersize=6, linewidth=1.5)
    ax.plot(unscaled_preds, label="Predicted ₹", marker='x', color='green', markersize=6, linewidth=1.5)

    ax.set_title("Unscaled GNN Predictions vs Actual Stock Prices", fontsize=16)
    ax.set_xlabel("Ticker", fontsize=13)
    ax.set_ylabel("Close Price (₹)", fontsize=13)

    ax.set_xticks(range(len(tickers_used)))
    ax.set_xticklabels(tickers_used, rotation=90, ha='center', fontsize=10)
    ax.tick_params(axis='x', pad=8)
    ax.tick_params(axis='y', labelsize=11)

    plt.subplots_adjust(bottom=0.4, top=0.92, left=0.06, right=0.98)
    ax.legend(fontsize=11)
    ax.grid(True)

    # ✅ Save files with visible output guaranteed
    png_path = get_artifact_path("gnn_vs_actual_prices_hd.png")
    svg_path = get_artifact_path("gnn_vs_actual_prices_hd.svg")
    
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    print(f"✅ PNG saved to: {png_path}")
    print(f"✅ SVG saved to: {svg_path}")
    
    # Optional: reopen image if needed
    # from IPython.display import Image, display
    # display(Image(filename=png_path))


# ---------- MAIN ----------
def main():
    df, granger_df = load_data()
    graph_data, ticker_index_map, full_df = build_graph(df, granger_df)
    model = train_gcn(graph_data)
    pred = evaluate(model, graph_data)
    plot_scaled_predictions(graph_data, pred)
    plot_unscaled_predictions(pred, graph_data, full_df, ticker_index_map)

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()

