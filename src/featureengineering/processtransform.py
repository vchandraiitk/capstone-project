import os
import zipfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from functools import partial

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INPUT_ZIP = "stock_data_with_stationary.zip"
INPUT_CSV = "stock_data_with_stationary.csv"
OUTPUT_NAME = "stock_data_final_transformed"
OUTPUT_CSV = f"{OUTPUT_NAME}.csv"
OUTPUT_ZIP = f"{OUTPUT_NAME}.zip"

# ---------- PATH HELPERS ----------

def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename: str) -> str:
    return os.path.join(get_project_root(), DATA_FOLDER, filename)

# ---------- FILE UTILS ----------

def unzip_if_needed(zip_filename: str, inner_csv: str):
    zip_path = get_data_path(zip_filename)
    csv_path = get_data_path(inner_csv)

    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(get_data_path(""))
        print(f"✅ Unzipped {zip_filename}")
    else:
        print(f"📁 {inner_csv} already exists.")

def cleanup_previous_outputs():
    for file in [OUTPUT_CSV, OUTPUT_ZIP]:
        full_path = get_data_path(file)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"🗑️ Deleted existing file: {full_path}")

# ---------- NORMALIZATION ----------

def load_input_dataframe() -> pd.DataFrame:
    csv_path = get_data_path(INPUT_CSV)
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    return df.sort_values(['Ticker', 'Date'])

def normalize_column_back(df: pd.DataFrame, col: str) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df = df.sort_values(['Ticker', 'Date']).copy()
    scaled_col_name = f"{col.replace('_stationary', '')}_scaled"

    df[scaled_col_name] = df.groupby('Ticker')[col].transform(
        lambda x: scaler.fit_transform(x.fillna(method='ffill').values.reshape(-1, 1)).flatten()
                  if x.notna().sum() > 1 else pd.Series([np.nan] * len(x), index=x.index)
    )
    return df[['Date', 'Ticker', scaled_col_name]]

def normalize_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df_sorted = df.sort_values(['Ticker', 'Date']).copy()
    scaled_col_name = f"{col.replace('_stationary', '')}_scaled"

    df_sorted[scaled_col_name] = df_sorted.groupby('Ticker')[col].transform(
        lambda x: scaler.fit_transform(x.fillna(method='ffill').values.reshape(-1, 1)).flatten()
    )
    return df_sorted[['Date', 'Ticker', scaled_col_name]]

def save_compressed_output(df: pd.DataFrame):
    csv_path = get_data_path(OUTPUT_CSV)
    zip_path = get_data_path(OUTPUT_ZIP)

    df.to_csv(csv_path, index=False)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=OUTPUT_CSV)

    ##os.remove(csv_path)
    print(f"✅ Final normalized data saved and zipped at: {zip_path}")

# ---------- MAIN ----------

def main():
    unzip_if_needed(INPUT_ZIP, INPUT_CSV)
    cleanup_previous_outputs()

    df = load_input_dataframe()
    stationary_columns = [col for col in df.columns if col.endswith('_stationary')]

    print("⚙️ Normalizing stationary columns in parallel...")
    normalize_partial = partial(normalize_column, df)

    with Pool(cpu_count() - 1) as pool:
        scaled_dfs = pool.map(normalize_partial, stationary_columns)

    for df_scaled in scaled_dfs:
        df = pd.merge(df, df_scaled, on=['Date', 'Ticker'], how='left')

    save_compressed_output(df)
    print("🎉 Transformation and compression complete.")

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()

