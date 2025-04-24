import os
import zipfile
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from multiprocessing import Pool, cpu_count
from functools import partial

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
ZIP_FILE = "enriched_stock_data.zip"
CSV_INSIDE_ZIP = "enriched_stock_data.csv"
OUTPUT_NAME = "stock_data_with_stationary"
OUTPUT_CSV = f"{OUTPUT_NAME}.csv"
OUTPUT_ZIP = f"{OUTPUT_NAME}.zip"

RAW_COLUMNS = [
    'Close', 'Volume', '100DayMA', '50DayMA', 'BookValue', 'EBITDA', 'GPM',
    'Market-Cap-Sales', 'NPM', 'OPM', 'PriceToBookValue', 'Sales',
    'CBOE', 'CrudeOil', 'Dollar-INR', 'IndianBondYieldRate',
    'inflation-Monthly', 'inflation-yearly', 'GDP', 'UnemploymentRate'
]

# ---------- PATH HELPERS ----------

def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename: str) -> str:
    return os.path.join(get_project_root(), DATA_FOLDER, filename)

# ---------- FILE HANDLING ----------

def unzip_if_needed(zip_filename: str, extract_to: str, inner_csv: str):
    zip_path = get_data_path(zip_filename)
    extract_path = get_data_path(inner_csv)

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(get_data_path(""))  # unzip in staging
        print(f"✅ Unzipped {zip_filename} to {extract_to}")
    else:
        print(f"📁 {inner_csv} already exists.")

def load_enriched_data() -> pd.DataFrame:
    csv_path = get_data_path(CSV_INSIDE_ZIP)
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['Ticker', 'Date'])

def check_stationarity(series):
    series = series.dropna()
    return adfuller(series)[1] <= 0.05 if len(series) >= 10 and series.nunique() > 1 else False

def process_stationary_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df_sorted = df.sort_values(['Ticker', 'Date']).copy()
    df_sorted[f'{col}_stationary'] = df_sorted.groupby('Ticker')[col].transform(
        lambda s: s.diff() if not check_stationarity(s) else s
    )
    return df_sorted[['Date', 'Ticker', f'{col}_stationary']]

def save_compressed_csv(df: pd.DataFrame):
    csv_path = get_data_path(OUTPUT_CSV)
    zip_path = get_data_path(OUTPUT_ZIP)

    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"🗑️ Removed old CSV: {csv_path}")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"🗑️ Removed old ZIP: {zip_path}")

    df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=OUTPUT_CSV)
    os.remove(csv_path)
    print(f"✅ Saved compressed file to: {zip_path}")

# ---------- MAIN FUNCTION ----------

def main():
    zip_output_path = get_data_path(OUTPUT_ZIP)

    if os.path.exists(zip_output_path):
        print(f"✅ {OUTPUT_ZIP} already exists. Skipping processing.")
        return

    unzip_if_needed(ZIP_FILE, DATA_FOLDER, CSV_INSIDE_ZIP)
    df = load_enriched_data()
#    df = df.head(2)
    print("⚙️ Running stationarity checks...")
    #print(df.head(2))
    #df = df[df['Ticker'].str.strip().str.upper() == 'RELIANCE']
    #print(df.shape)
    # Create partial function to inject df
    process_func = partial(process_stationary_column, df)

    with Pool(cpu_count() - 1) as pool:
        results = pool.map(process_func, RAW_COLUMNS)

    for df_stat in results:
        df = pd.merge(df, df_stat, on=['Date', 'Ticker'], how='left')

    save_compressed_csv(df)
    print("🎉 Stationarity pipeline completed.")

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()

