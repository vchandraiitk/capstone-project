import os
import pandas as pd
import random
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --------- CONFIG ---------
ticker_df = pd.read_csv("Unique_Ticker_IDs.csv")
tickers = ticker_df.iloc[:, 0].dropna().unique().tolist()

start_date = datetime(2018, 1, 1)
end_date = datetime(2025, 4, 30)
date_range = pd.date_range(start=start_date, end=end_date)

TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# --------- LOAD FinBERT PIPELINE ---------
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# --------- TEXT GENERATORS ---------
def generate_text(ticker, source):
    tweets = [
        f"${ticker} is gaining momentum on social media.",
        f"Retail buzz around {ticker} surges this morning.",
        f"{ticker} looks bullish today. Watching closely!",
        f"Analysts are tweeting about {ticker}'s breakout.",
        f"Big day ahead for ${ticker}—watch the trend!",
        f"Just bought more {ticker}. Fingers crossed! 🚀",
        f"Why is nobody talking about ${ticker}? This looks ready to explode.",
        f"Traders are piling into {ticker} after early volume spike.",
        f"Anyone holding {ticker}? Thoughts?",
        f"{ticker} just broke resistance with high volume. Bullish!",
        f"$${ticker} is consolidating. Expecting a breakout.",
        f"{ticker} making headlines today. Momentum building!",
        f"Massive move expected in {ticker}—keep an eye!",
        f"{ticker} catching fire on Fintwit! 🔥",
        f"{ticker} trend reversal incoming?"
    ]
    news = [
        f"{ticker} announces strong Q results, stock reacts positively.",
        f"{ticker} enters partnership to expand business footprint.",
        f"{ticker} in focus as markets react to global cues.",
        f"{ticker} unveils new product in high-growth sector.",
        f"{ticker} board approves dividend for shareholders.",
        f"{ticker} under regulatory scanner after audit disclosure.",
        f"{ticker} stock rallies on positive macro data and earnings beat.",
        f"{ticker} CEO comments on strategic roadmap amid market volatility.",
        f"{ticker} gains after foreign investor inflows hit new high.",
        f"{ticker} announces share buyback to enhance shareholder value.",
        f"{ticker} sees sharp increase in retail participation post budget.",
        f"{ticker} tops analyst upgrade list amid sector optimism.",
        f"{ticker} launches green initiative for sustainable growth.",
        f"{ticker} faces margin pressure due to rising input costs.",
        f"{ticker} expands operations into new international markets."
    ]
    return random.choice(tweets if source == "Twitter" else news)

# --------- PROCESS EACH TICKER ---------
for ticker in tickers:
    print(f"🔍 Processing ticker: {ticker}")
    rows = []

    for date in date_range:
        source = random.choice(["Twitter", "News"])
        text = generate_text(ticker, source)
        rows.append((ticker, date.strftime("%Y-%m-%d"), text))

    df = pd.DataFrame(rows, columns=["TickerID", "Date", "Text"])

    # Apply FinBERT
    sentiments = sentiment_pipeline(df["Text"].tolist())
    df["Sentiment"] = [res["label"].upper() for res in sentiments]

    # Add sentiment score
    sentiment_score_map = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
    df["Sentiment_Score"] = df["Sentiment"].map(sentiment_score_map)

    # Save output
    output_path = os.path.join(TEMP_FOLDER, f"{ticker}_sentiment.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

print("🎉 All tickers processed with sentiment and score.")

