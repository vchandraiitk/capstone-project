import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INDUSTRY_FOLDER = "data/industry"
SENTIMENT_FOLDER = "data/sentiment"
CSV_STOCK = "stock_data_final_transformed.csv"
CSV_GRANGER = "granger_all_tickers.csv"
CSV_INDUSTRY = "Ticker-Industry.csv"
ARTIFACT_FOLDER = "artifacts"
PLOT_FILE = "gnn_vs_actual_prices_hd.png"

# ---------- PATH UTILS ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(file):
    return os.path.join(get_project_root(), DATA_FOLDER, file)

def get_industry_path(file):
    return os.path.join(get_project_root(), INDUSTRY_FOLDER, file)

def get_sentiment_path(file):
    return os.path.join(get_project_root(), SENTIMENT_FOLDER, file)

def get_artifact_path(file):
    path = os.path.join(get_project_root(), ARTIFACT_FOLDER)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, file)

# ---------- SENTIMENT FUNCTION ----------
def add_financial_sentiment(df, text_column='Text', batch_size=32):
    """
    Adds a 'Sentiment' column using FinBERT on the specified text column.
    """
    print("🧠 Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def analyze_sentiment(texts):
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = sentiment_pipeline(batch)
            sentiments.extend([res['label'] for res in results])
        return sentiments

    print("📊 Running sentiment analysis...")
    df = df.copy()
    df['Sentiment'] = analyze_sentiment(df[text_column].tolist())
    return df

# ---------- MAIN ----------
def main():
    input_file = "nifty500_synthetic_twitter_news_full.csv"
    output_file = "nifty500_sentiment_enriched.csv"

    input_path = get_sentiment_path(input_file)
    output_path = get_sentiment_path(output_file)

    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)

    df_sent = add_financial_sentiment(df)

    print(f"💾 Saving: {output_path}")
    df_sent.to_csv(output_path, index=False)
    print("✅ Sentiment analysis complete.")

if __name__ == "__main__":
    main()

