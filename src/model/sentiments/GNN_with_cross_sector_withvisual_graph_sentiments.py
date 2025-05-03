# Vertex AI Workbench-compatible GAT-based Stock Forecasting

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import json
import networkx as nx

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INDUSTRY_FOLDER = "data/industry"
CSV_STOCK = "stock_data_final_transformed.csv"
CSV_GRANGER = "granger_all_tickers.csv"
CSV_INDUSTRY = "Ticker-Industry.csv"
ARTIFACT_FOLDER = "artifacts_new"
PLOT_FILE = "gnn_vs_actual_prices_hd.png"

# ---------- PATH UTILS ----------
def get_project_root():
    return os.getcwd()

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
    sentiment_df = pd.read_csv(get_data_path("nifty500_sentiment_merged.csv"), parse_dates=["Date"])

    sentiment_df = sentiment_df.rename(columns={'TickerID': 'Ticker', 'Sentiment_Score': 'SentimentScore'})
    df = df.merge(sentiment_df[['Ticker', 'Date', 'SentimentScore']], on=['Ticker', 'Date'], how='left')
    df['SentimentScore'] = df['SentimentScore'].fillna(0)

    df = df.sort_values(['Ticker', 'Date'])
    df['row'] = df.groupby('Ticker').cumcount()
    df = df[df['row'] >= 30].drop(columns='row')
    df['Close_scaled'] = df.groupby('Ticker')['Close'].transform(lambda x: (x - x.mean()) / x.std())
    df['time_index'] = df.groupby('Ticker').cumcount()
    df['time_index_scaled'] = df.groupby('Ticker')['time_index'].transform(lambda x: (x - x.mean()) / x.std())
    df['Close_scaled_diff'] = df.groupby('Ticker')['Close_scaled'].diff()
    df = df.merge(industry_df, on='Ticker', how='left')
    return df, granger_df

# ---------- GAT MODEL ----------
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

def train_gcn(graph_data, epochs=100):
    model = GAT(graph_data.num_node_features)
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
    tickers_used = list(ticker_index_map.keys())
    unscaled_preds, unscaled_actuals = [], []
    for idx, ticker in enumerate(tickers_used):
        mean = df[df['Ticker'] == ticker]['Close'].mean()
        std = df[df['Ticker'] == ticker]['Close'].std()
        unscaled_preds.append(pred[idx].item() * std + mean)
        unscaled_actuals.append(graph_data.y[idx].item() * std + mean)

    fig, ax = plt.subplots(figsize=(min(50, max(12, len(tickers_used) * 0.8)), max(8, len(tickers_used) * 0.1)))
    ax.plot(unscaled_actuals, label="Actual ₹", marker='o', color='black')
    ax.plot(unscaled_preds, label="Predicted ₹", marker='x', color='green')
    ax.set_title("Unscaled GNN Predictions vs Actual Stock Prices")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Close Price (₹)")
    ax.set_xticks(range(len(tickers_used)))
    ax.set_xticklabels(tickers_used, rotation=90)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    png_path = get_artifact_path("gnn_vs_actual_prices_hd.png")
    svg_path = get_artifact_path("gnn_vs_actual_prices_hd.svg")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"✅ PNG saved to: {png_path}")
    print(f"✅ SVG saved to: {svg_path}")

def build_graph_temp(df, granger_df):
    tickers = granger_df['Ticker'].unique()
    df = df[df['Ticker'].isin(tickers)]
    granger_df = granger_df[granger_df['Ticker'].isin(tickers)]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    industry_encoded = encoder.fit_transform(df[['Industry']].fillna('Unknown'))
    industry_encoded_df = pd.DataFrame(industry_encoded, columns=encoder.get_feature_names_out(['Industry']))
    df = pd.concat([df.reset_index(drop=True), industry_encoded_df], axis=1)

    ticker_features_map = {
        ticker: [
            f.replace('_stationary', '') + '_scaled'
            for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
            if f.replace('_stationary', '') + '_scaled' in df.columns
        ] + ['time_index_scaled', 'Close_scaled_diff'] + list(industry_encoded_df.columns) for ticker in tickers
    }

    ticker_industry_map = df.drop_duplicates('Ticker').set_index('Ticker')['Industry'].to_dict()

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

    # cross_sector_edges = {
    #     'Banking': ['IT', 'Energy'],
    #     'Energy': ['Manufacturing', 'Industrials'],
    #     'IT': ['Consumer'],
    #     'Consumer': ['Retail'],
    #     # You can expand more...
    # }
    
    with open('cross_sector_edges.json', 'r') as f:
        cross_sector_edges = json.load(f)
    edges = []
    for t1 in valid_tickers:
        for t2 in valid_tickers:
            if t1 == t2:
                continue
            # Granger feature overlap
            granger_overlap = set(ticker_features_map[t1]) & set(ticker_features_map[t2])
            # Industry cross-sector logic
            industry_t1 = ticker_industry_map.get(t1, 'Unknown')
            industry_t2 = ticker_industry_map.get(t2, 'Unknown')
            industry_relation = industry_t2 in cross_sector_edges.get(industry_t1, [])

            if granger_overlap or industry_relation:
                edges.append([ticker_index_map[t1], ticker_index_map[t2]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    graph_data = Data(x=torch.stack(node_features), edge_index=edge_index, y=torch.stack(node_labels).squeeze())

    return graph_data, ticker_index_map, df

def build_graph(df, granger_df):
    tickers = granger_df['Ticker'].unique()
    df = df[df['Ticker'].isin(tickers)]
    granger_df = granger_df[granger_df['Ticker'].isin(tickers)]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    industry_encoded = encoder.fit_transform(df[['Industry']].fillna('Unknown'))
    industry_encoded_df = pd.DataFrame(industry_encoded, columns=encoder.get_feature_names_out(['Industry']))
    df = pd.concat([df.reset_index(drop=True), industry_encoded_df], axis=1)

    ticker_industry_map = df.drop_duplicates('Ticker').set_index('Ticker')['Industry'].to_dict()

    ticker_features_map = {
        ticker: [
            f.replace('_stationary', '') + '_scaled'
            for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
            if f.replace('_stationary', '') + '_scaled' in df.columns
        ] + ['time_index_scaled', 'Close_scaled_diff', 'SentimentScore'] + list(industry_encoded_df.columns) for ticker in tickers
    }

    with open('cross_sector_edges.json', 'r') as f:
        cross_sector_edges = json.load(f)
    node_features, node_labels, ticker_index_map = [], [], {}

    for ticker in tickers:
        sub_df = df[df['Ticker'] == ticker].dropna()
        if sub_df.empty:
            continue
        last_row = sub_df.iloc[-1]
        feat_vals = [last_row.get(f, 0.0) for f in sorted(set(f for feats in ticker_features_map.values() for f in feats))]
        if pd.isna(last_row['Close_scaled']):
            continue
        x = torch.tensor(feat_vals, dtype=torch.float)
        y = torch.tensor([last_row['Close_scaled']], dtype=torch.float)
        ticker_index_map[ticker] = len(node_features)
        node_features.append(x)
        node_labels.append(y)

    valid_tickers = list(ticker_index_map.keys())

    edges = []
    ticker_relations = {ticker: [] for ticker in valid_tickers}

    for t1 in valid_tickers:
        for t2 in valid_tickers:
            if t1 == t2:
                continue
            granger_overlap = set(ticker_features_map[t1]) & set(ticker_features_map[t2])
            industry_t1 = ticker_industry_map.get(t1, 'Unknown')
            industry_t2 = ticker_industry_map.get(t2, 'Unknown')
            industry_relation = industry_t2 in cross_sector_edges.get(industry_t1, [])

            if granger_overlap or industry_relation:
                edges.append([ticker_index_map[t1], ticker_index_map[t2]])
                ticker_relations[t1].append((t2, len(granger_overlap), industry_relation))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    # 🔥 Smarter recommendations
    ticker_recommendations = {}
    for ticker, related_list in ticker_relations.items():
        if not related_list:
            ticker_recommendations[ticker] = []
            continue

        # Sort: 1st by cross-sector match (industry_relation=True), then by highest feature overlap
        sorted_related = sorted(
            related_list,
            key=lambda x: (not x[2], -x[1])  # prefer cross-sector True first, then higher granger overlap
        )

        # Take top 2
        top_related = [r[0] for r in sorted_related[:2]]
        ticker_recommendations[ticker] = top_related

    graph_data = Data(x=torch.stack(node_features), edge_index=edge_index, y=torch.stack(node_labels).squeeze())

    return graph_data, ticker_index_map, df, ticker_recommendations


def forecast_future_prices(model, df, granger_df, ticker_recommendations, horizons=[30, 180, 365], n_samples=50):
    model.eval()
    tickers = df['Ticker'].unique()
    forecasts = []

    # One-Hot Encoding for Industry
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    industry_encoded = encoder.fit_transform(df[['Industry']].fillna('Unknown'))
    industry_encoded_df = pd.DataFrame(industry_encoded, columns=encoder.get_feature_names_out(['Industry']))
    df = pd.concat([df.reset_index(drop=True), industry_encoded_df], axis=1)

    for ticker in tickers:
        sub_df = df[df['Ticker'] == ticker].dropna()
        if sub_df.empty:
            continue
        last_row = sub_df.iloc[-1]

        # Get relevant features
        features = list(set(
            [
                f.replace('_stationary', '') + '_scaled'
                for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
                if f.replace('_stationary', '') + '_scaled' in df.columns
            ] + ['time_index_scaled', 'Close_scaled_diff'] + list(industry_encoded_df.columns)
        ))

        mean_close = sub_df['Close'].mean()
        std_close = sub_df['Close'].std()
        base_time_idx = last_row['time_index']

        for h in horizons:
            new_time_index = base_time_idx + h
            new_time_index_scaled = (new_time_index - sub_df['time_index'].mean()) / sub_df['time_index'].std()

            # Prepare feature tensor
            feat_vals = []
            for f in features:
                if f == 'time_index_scaled':
                    feat_vals.append(new_time_index_scaled)
                elif f == 'Close_scaled_diff':
                    feat_vals.append(0.0)  # Assume momentum = 0
                else:
                    try:
                        val = float(last_row[f])
                        if pd.isna(val):
                            val = 0.0
                    except (KeyError, ValueError, TypeError):
                        val = 0.0
                    feat_vals.append(val)

            x_tensor = torch.tensor([feat_vals], dtype=torch.float)
            # Pad features if needed
            x_tensor = torch.nn.functional.pad(x_tensor, (0, model.gat1.in_channels - x_tensor.shape[1]))

            dummy_data = Data(x=x_tensor.repeat(2, 1), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))

            # Monte Carlo Dropout: multiple forward passes
            model.train()  # Enable dropout
            predictions = []
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = model(dummy_data)[0].item()
                    predictions.append(pred)

            preds_np = np.array(predictions)
            pred_mean_scaled = preds_np.mean()
            pred_std_scaled = preds_np.std()

            # Final unscaled forecasted price
            pred_mean_unscaled = pred_mean_scaled * std_close + mean_close

            # Confidence % Calculation
            confidence_percent = max(0, min(100, 100 - (pred_std_scaled / abs(pred_mean_scaled)) * 100))

            forecasts.append({
                'Ticker': ticker,
                'Forecast_Horizon_Days': h,
                'Predicted_Close_Price': round(pred_mean_unscaled, 2),
                'Confidence_%': round(confidence_percent, 2)
            })

    # Save to CSV
    forecast_df = pd.DataFrame(forecasts)
    forecast_df['Suggested_Related_Stocks'] = forecast_df['Ticker'].apply(lambda t: ', '.join(ticker_recommendations.get(t, [])))
    forecast_csv_path = get_artifact_path("gnn_forecast_confidence_final.csv")
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"📤 Final forecasts with confidence % saved to: {forecast_csv_path}")
    
    import networkx as nx
from pyvis.network import Network

def visualize_stock_network(ticker_index_map, ticker_recommendations, df):
    # Create a NetworkX graph
    G = nx.DiGraph()

    # Reverse ticker_index_map
    index_ticker_map = {v: k for k, v in ticker_index_map.items()}

    # Map ticker to industry
    ticker_industry_map = df.drop_duplicates('Ticker').set_index('Ticker')['Industry'].to_dict()

    # Unique industries and color mapping
    industries = list(set(ticker_industry_map.values()))
    color_map = plt.colormaps.get_cmap('tab20')
    industry_color = {
        industry: f"#{''.join(f'{int(255*c):02x}' for c in color_map(i/len(industries))[:3])}"
        for i, industry in enumerate(industries)
    }

    # Add Nodes
    for ticker, industry in ticker_industry_map.items():
        G.add_node(
            ticker,
            title=f"{ticker} ({industry})",
            color=industry_color.get(industry, '#cccccc'),
            shape='box',
            size=15
        )

    # Add Edges based on recommendations
    for ticker, related_list in ticker_recommendations.items():
        for related_ticker in related_list:
            G.add_edge(ticker, related_ticker)

    # Visualize using pyvis
    net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', directed=True)
    net.from_nx(G)

    output_path = get_artifact_path("stock_cross_sector_network.html")
    net.save_graph(output_path)

    print(f"✅ Interactive network graph saved to: {output_path}")




# ---------- MAIN ----------
if __name__ == "__main__":
    df, granger_df = load_data()
    #graph_data, ticker_index_map, full_df = build_graph(df, granger_df)
    graph_data, ticker_index_map, full_df, ticker_recommendations = build_graph(df, granger_df)
    visualize_stock_network(ticker_index_map, ticker_recommendations, full_df)
    print("🔢 INPUT_FEATURE_SIZE =", graph_data.num_node_features)
    model = train_gcn(graph_data)
    pred = evaluate(model, graph_data)
    torch.save(model.state_dict(), get_artifact_path("gnn_model.pt"))
    plot_scaled_predictions(graph_data, pred)
    plot_unscaled_predictions(pred, graph_data, full_df, ticker_index_map)
    forecast_future_prices(model, full_df, granger_df, ticker_recommendations)

