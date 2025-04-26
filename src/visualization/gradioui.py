import pandas as pd
import gradio as gr

# Define paths for your files
FORECAST = '/Users/vikaschandra/workspace/capstone-project/artifacts/1_gnn_forecast_confidence_final.csv'
SECTOR = '/Users/vikaschandra/workspace/capstone-project/artifacts/2_stock_cross_sector_network.html'
GRANGER = '/Users/vikaschandra/workspace/capstone-project/artifacts/3_granger_network-Graph.html'

# Load your forecast data
df = pd.read_csv(FORECAST)

tickers = sorted(df['Ticker'].unique())

# Function to fetch forecast
def get_forecast(ticker):
    filtered = df[df['Ticker'] == ticker]
    output = {}
    for horizon in [30, 180, 365]:
        row = filtered[filtered['Forecast_Horizon_Days'] == horizon]
        if not row.empty:
            price = row.iloc[0]['Predicted_Close_Price']
            confidence = row.iloc[0]['Confidence_%']
            output[f"{horizon} Days"] = f"Price: {price:.2f}, Confidence: {confidence:.2f}%"
        else:
            output[f"{horizon} Days"] = "No data available"
    return output['30 Days'], output['180 Days'], output['365 Days']

# Function to fetch related stocks
def get_related_stocks(ticker):
    filtered = df[df['Ticker'] == ticker]
    if not filtered.empty:
        return filtered.iloc[0]['Suggested_Related_Stocks']
    else:
        return "No related stocks found."

# Gradio UI with multiple Tabs
with gr.Blocks() as demo:
    with gr.Tab("Forecast Viewer"):
        gr.Markdown("## 📈 Stock Forecast Viewer")
        ticker_dropdown = gr.Dropdown(label="Select Ticker", choices=tickers)

        forecast_30 = gr.Textbox(label="30 Days Forecast")
        forecast_180 = gr.Textbox(label="180 Days Forecast")
        forecast_365 = gr.Textbox(label="365 Days Forecast")

        related_button = gr.Button("Show Related Stocks")
        related_output = gr.Textbox(label="Suggested Related Stocks")

        ticker_dropdown.change(
            get_forecast,
            inputs=[ticker_dropdown],
            outputs=[forecast_30, forecast_180, forecast_365]
        )

        related_button.click(
            get_related_stocks,
            inputs=[ticker_dropdown],
            outputs=[related_output]
        )

    with gr.Tab("Cross Sector Network"):
        gr.Markdown("## 🔹 Cross Sector Stock Network")
        with open(SECTOR, 'r', encoding='utf-8') as f:
            html_content_sector = f.read()
        gr.HTML(html_content_sector)

    with gr.Tab("Granger Network"):
        gr.Markdown("## 🔹 Granger Causal Network")
        with open(GRANGER, 'r', encoding='utf-8') as f:
            html_content_granger = f.read()
        gr.HTML(html_content_granger)

# Launch app
demo.launch()

