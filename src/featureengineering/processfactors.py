import os
import pandas as pd
import sys 
# Add src to sys.path so you can import featureengineering modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from featureengineering import processstockdata

def get_project_root() -> str:
    """Returns the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename: str, subfolder: str) -> str:
    """Builds full path to a data file."""
    return os.path.join(get_project_root(), subfolder, filename)

def create_base_grid(df_stock: pd.DataFrame) -> pd.DataFrame:
    """Creates a complete [Date, Ticker] daily grid."""
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    daily_dates = pd.date_range(df_stock['Date'].min(), df_stock['Date'].max(), freq='D')
    all_tickers = df_stock['Ticker'].unique()
    return pd.MultiIndex.from_product([daily_dates, all_tickers], names=['Date', 'Ticker']).to_frame(index=False)

def load_and_process_stock_factors(df_base: pd.DataFrame, factor_map: dict, subfolder: str = "data/factors") -> list:
    """Loads and processes stock-specific factor files."""
    stock_feature_dfs = []
    for feature, file in factor_map.items():
        filepath = get_data_path(file, subfolder)
        df_feat = pd.read_csv(filepath)
        df_feat['Date'] = pd.to_datetime(df_feat['Date'])

        df_joined = pd.merge(df_base, df_feat, on=['Date', 'Ticker'], how='left').sort_values(['Ticker', 'Date'])
        df_joined[feature] = df_joined.groupby('Ticker')[feature].ffill()
        df_joined[feature + '_pct'] = df_joined.groupby('Ticker')[feature].pct_change()
        df_joined[feature + '_diff'] = df_joined.groupby('Ticker')[feature].diff()

        stock_feature_dfs.append(df_joined[['Date', 'Ticker', feature, feature + '_pct', feature + '_diff']])
    return stock_feature_dfs

def merge_stock_features(df_stock: pd.DataFrame, stock_feature_dfs: list) -> pd.DataFrame:
    """Merges stock features into the main stock dataframe."""
    df_enriched = df_stock.copy()
    for df_feat in stock_feature_dfs:
        df_enriched = pd.merge(df_enriched, df_feat, on=['Date', 'Ticker'], how='left')

    fill_cols = [col for col in df_enriched.columns if not col.endswith('_pct') and not col.endswith('_diff')]
    df_enriched[fill_cols] = (
        df_enriched.sort_values(['Ticker', 'Date'])
                   .groupby('Ticker')[fill_cols]
                   .ffill()
                   .bfill()
    )
    return df_enriched

def load_and_merge_economic_factors(df_stock: pd.DataFrame, factor_map: dict, subfolder: str = "data/factors") -> pd.DataFrame:
    """Loads and merges economic factors into the enriched stock dataframe."""
    all_dates = pd.date_range(df_stock['Date'].min(), df_stock['Date'].max(), freq='D')
    econ_feature_dfs = []

    for factor, file in factor_map.items():
        filepath = get_data_path(file, subfolder)
        df_econ = pd.read_csv(filepath)
        df_econ['Date'] = pd.to_datetime(df_econ['Date'])

        if df_econ.shape[1] == 2:
            df_econ.columns = ['Date', factor]
        else:
            df_econ = df_econ[['Date', df_econ.columns[1]]]
            df_econ.columns = ['Date', factor]

        df_econ = df_econ[~df_econ.duplicated(subset='Date', keep='first')]
        df_econ = df_econ.set_index('Date').reindex(all_dates).ffill().reset_index()
        df_econ.columns = ['Date', factor]
        df_econ[factor + '_pct'] = df_econ[factor].pct_change()

        econ_feature_dfs.append(df_econ)

    df_enriched = df_stock.copy()
    for df_econ in econ_feature_dfs:
        df_enriched = pd.merge(df_enriched, df_econ, on='Date', how='left')

    econ_cols = [col for col in df_enriched.columns if col in factor_map]
    df_enriched[econ_cols] = (
        df_enriched.sort_values(['Ticker', 'Date'])
                   .groupby('Ticker')[econ_cols]
                   .ffill()
                   .bfill()
    )
    return df_enriched

def save_enriched_data(df: pd.DataFrame, filename: str = "enriched_stock_data.csv", folder: str = "data/staging") -> None:
    """Saves the enriched DataFrame to CSV after deleting any existing version."""
    save_dir = get_data_path("", folder)
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, filename)

    # Delete existing file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🗑️ Deleted existing file: {output_path}")

    # Save the new enriched data
    df.to_csv(output_path, index=False)
    print(f"✅ Enriched data saved to: {output_path}")


def main(df_stock: pd.DataFrame) -> pd.DataFrame:
    """Main pipeline function for factor enrichment."""
    stock_factors = {
        '100DayMA': '100DayMovingAvg.csv',
        '50DayMA': '50DayMovingAvg.csv',
        'BookValue': 'BookValue.csv',
        'EBITDA': 'EBITDA.csv',
        'GPM': 'GPM.csv',
        'Market-Cap-Sales': 'Market-Cap-Sales.csv',
        'NPM': 'NPM.csv',
        'OPM': 'OPM.csv',
        'PriceToBookValue': 'PriceToBookValue.csv',
        'Sales': 'Sales.csv'
    }

    economy_factors = {
        'CBOE': 'CBOE.csv',
        'CrudeOil': 'CrudeOil.csv',
        'Dollar-INR': 'Dollar-INR.csv',
        'IndianBondYieldRate': 'IndianBondYieldRate.csv',
        'inflation-Monthly': 'inflation-Monthly.csv',
        'inflation-yearly': 'inflation-yearly.csv',
        'GDP': 'GDP.csv',
        'UnemploymentRate': 'UnemploymentRate.csv'
    }

    df_base = create_base_grid(df_stock)
    stock_feature_dfs = load_and_process_stock_factors(df_base, stock_factors)
    df_stock_enriched = merge_stock_features(df_stock, stock_feature_dfs)
    df_stock_enriched = load_and_merge_economic_factors(df_stock_enriched, economy_factors)

    print("✅ Final enriched DataFrame preview:")
    print(df_stock_enriched[['Date', 'Ticker'] + list(stock_factors.keys()) + list(economy_factors.keys())].head())

    return df_stock_enriched

# 👇 This block makes it runnable as a standalone script
if __name__ == "__main__":
    from featureengineering import processstockdata

    print("🚀 Running factor enrichment directly...")
    df_stock = processstockdata.main()
    df_enriched = main(df_stock)

    print("✅ Full enrichment completed.")
    print(df_enriched.head())
    print(df_stock.shape)
    print(df_enriched.shape)
    save_enriched_data(df_enriched)
    print("Saved enriched data")

