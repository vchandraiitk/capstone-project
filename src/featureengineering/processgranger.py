import os
import zipfile
import pandas as pd
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from statsmodels.tsa.stattools import grangercausalitytests

# ---------- CONFIG ----------
DATA_FOLDER = "data/staging"
INPUT_ZIP = "stock_data_final_transformed.zip"
INPUT_CSV = "stock_data_final_transformed.csv"
OUTPUT_FILE = "granger_all_tickers.csv"
MAX_LAG = 5

macro_factors = [
    'CBOE_stationary', 'CrudeOil_stationary', 'Dollar-INR_stationary',
    'IndianBondYieldRate_stationary', 'inflation-Monthly_stationary',
    'inflation-yearly_stationary', 'GDP_stationary', 'UnemploymentRate_stationary'
]

stock_factors = [
    'Volume_stationary', 'EBITDA_stationary', 'BookValue_stationary',
    'Sales_stationary', 'GPM_stationary', 'NPM_stationary', 'OPM_stationary',
    'PriceToBookValue_stationary', 'Market-Cap-Sales_stationary',
    '100DayMA_stationary', '50DayMA_stationary'
]

ALL_FACTORS = macro_factors + stock_factors

# ---------- PATH HELPERS ----------

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename):
    return os.path.join(get_project_root(), DATA_FOLDER, filename)

# ---------- FILE HANDLING ----------

def unzip_if_needed(zip_filename, inner_csv):
    zip_path = get_data_path(zip_filename)
    csv_path = get_data_path(inner_csv)

    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(get_data_path(""))
        print(f"✅ Unzipped {zip_filename}")
    else:
        print(f"📁 {inner_csv} already exists.")

def cleanup_output_if_exists():
    output_path = get_data_path(OUTPUT_FILE)
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🗑️ Deleted old output: {output_path}")

# ---------- GRANGER TEST LOGIC ----------

def load_input_df():
    csv_path = get_data_path(INPUT_CSV)
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    df_clean = df.groupby('Ticker').apply(lambda g: g.iloc[30:]).reset_index(drop=True)
    return df_clean

def granger_test_all(ticker_df, ticker, target='Close', max_lag=MAX_LAG):
    result_list = []
    ticker_df = ticker_df.sort_values('Date')

    for factor in ALL_FACTORS:
        df_temp = ticker_df[[target, factor]].dropna()
        if df_temp.shape[0] > max_lag + 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    test_result = grangercausalitytests(df_temp, maxlag=max_lag, verbose=False)

                p_values = [round(test_result[i + 1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
                min_p = min(p_values)
                if min_p < 0.05:
                    result_list.append({'Ticker': ticker, 'Factor': factor, 'p-value': min_p})
            except Exception:
                continue
    return result_list

def run_granger_for_ticker(ticker, df_grouped):
    return granger_test_all(df_grouped[ticker], ticker)

def run_granger_parallel(df_clean):
    tickers = df_clean['Ticker'].unique()
    df_grouped = {ticker: df_clean[df_clean['Ticker'] == ticker] for ticker in tickers}

    with Pool(cpu_count() - 1) as pool:
        func = partial(run_granger_for_ticker, df_grouped=df_grouped)
        all_results = pool.map(func, tickers)

    return [item for sublist in all_results for item in sublist]

# ---------- MAIN ----------

def main():
    unzip_if_needed(INPUT_ZIP, INPUT_CSV)
    cleanup_output_if_exists()

    df_clean = load_input_df()

    print("⚙️ Running Granger causality analysis...")
    flat_results = run_granger_parallel(df_clean)

##    granger_df = pd.DataFrame(flat_results).sort_values(['Ticker', 'p-value'])
##    output_path = get_data_path(OUTPUT_FILE)
##    granger_df.to_csv(output_path, index=False)
##    ✅ Keep only top 3 significant causal factors per ticker
    granger_df = (
       pd.DataFrame(flat_results)
         .sort_values(['Ticker', 'p-value'])
         .groupby('Ticker')
         .head(3)
      )

    output_path = get_data_path(OUTPUT_FILE)
    granger_df.to_csv(output_path, index=False)
    print(f"✅ Saved top 3 Granger results per ticker to {output_path}")

    print(f"✅ Saved Granger results to {output_path}")

# ---------- ENTRY POINT ----------

if __name__ == "__main__":
    main()

