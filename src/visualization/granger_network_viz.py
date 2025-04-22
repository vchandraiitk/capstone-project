import os
import pandas as pd
import matplotlib.pyplot as plt
from pyvis.network import Network
import webbrowser

# ---------- PATH SETUP ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_FILE = os.path.join(get_project_root(), "data/staging/granger_all_tickers.csv")
OUTPUT_HTML = os.path.join(get_project_root(), "granger_network.html")

# ---------- LOAD DATA ----------
if not os.path.exists(INPUT_FILE):
    print(f"❌ File not found: {INPUT_FILE}")
    exit(1)

granger_df = pd.read_csv(INPUT_FILE)

# 📌 Filter to top 10 tickers
top_10_tickers = granger_df['Ticker'].unique()[:10]
granger_df = granger_df[granger_df['Ticker'].isin(top_10_tickers)]

# ---------- COLOR MAPPING ----------
unique_factors = granger_df['Factor'].unique()
color_map = plt.colormaps.get_cmap('tab20')
factor_colors = {
    factor: f"#{''.join(f'{int(255*c):02x}' for c in color_map(i)[:3])}"
    for i, factor in enumerate(unique_factors)
}

# ---------- BUILD NETWORK ----------
net = Network(height='700px', width='100%', directed=True, cdn_resources='in_line')

# Add factor nodes
for factor in unique_factors:
    display_label = factor.replace('_stationary', '')
    net.add_node(factor, label=display_label, color=factor_colors[factor], shape='ellipse')

# Add ticker nodes
for ticker in granger_df['Ticker'].unique():
    net.add_node(ticker, label=ticker, color='salmon', shape='box')

# Add edges
for _, row in granger_df.iterrows():
    color = factor_colors[row['Factor']]
    net.add_edge(
        row['Factor'], row['Ticker'],
        title=f"p = {row['p-value']}",
        color=color,
        width=max(1, 5 - row['p-value'] * 50)
    )

# ---------- EXPORT AND OPEN ----------
net.write_html(OUTPUT_HTML)
print(f"✅ Granger causality network saved to: {OUTPUT_HTML}")
webbrowser.open(f"file://{OUTPUT_HTML}")

