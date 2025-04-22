import os
import zipfile
import pandas as pd

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INPUT_ZIP = "stock_data_final_transformed.zip"
INPUT_CSV = "stock_data_final_transformed.csv"
GRANGER_CSV = "granger_all_tickers.csv"
OUTPUT_CSV = "feature_summary.csv"

# ---------- PATH HELPERS ----------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename):
    return os.path.join(get_project_root(), DATA_FOLDER, filename)

# ---------- UTILITIES ----------
def unzip_if_needed(zip_file, extracted_file):
    zip_path = get_data_path(zip_file)
    extract_path = get_data_path(extracted_file)
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(get_data_path(""))
        print(f"✅ Unzipped {zip_file}")
    else:
        print(f"📁 {extracted_file} already exists")

def load_cleaned_data():
    df = pd.read_csv(get_data_path(INPUT_CSV), parse_dates=['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    df['row'] = df.groupby('Ticker').cumcount()
    df = df[df['row'] >= 30].drop(columns='row')
    return df

def load_top10_granger():
    granger_df = pd.read_csv(get_data_path(GRANGER_CSV))
    #top_10_tickers = granger_df['Ticker'].unique()[:10]
    top_10_tickers = granger_df['Ticker'].unique()
    return granger_df[granger_df['Ticker'].isin(top_10_tickers)], top_10_tickers

def filter_top10_data(df, top_10_tickers):
    return df[df['Ticker'].isin(top_10_tickers)]

def build_feature_map(granger_df, tickers):
    return {
        ticker: [
            f.replace('_stationary', '') + '_scaled'
            for f in granger_df[granger_df['Ticker'] == ticker]['Factor'].tolist()
        ]
        for ticker in tickers
    }

def build_summary_df(feature_map):
    return pd.DataFrame([
        {'Ticker': ticker, 'GrangerFeatures': ', '.join(features)}
        for ticker, features in feature_map.items()
    ])

def save_summary(df):
    output_path = get_data_path(OUTPUT_CSV)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved feature summary to: {output_path}")

# ---------- MAIN ----------
def main():
    unzip_if_needed(INPUT_ZIP, INPUT_CSV)

    df = load_cleaned_data()
    granger_df, top_10_tickers = load_top10_granger()

    df = filter_top10_data(df, top_10_tickers)

    ticker_feature_map = build_feature_map(granger_df, top_10_tickers)
    feature_summary_df = build_summary_df(ticker_feature_map)

    save_summary(feature_summary_df)

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()

