import os
import pandas as pd
import numpy as np

def get_project_root() -> str:
    """Returns the project root directory by going two levels up from this file."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(filename: str, subfolder: str = "data/stock") -> str:
    """Builds absolute path to a file inside the project (relative to script location)."""
    root = get_project_root()
    return os.path.join(root, subfolder, filename)

def load_stock_data(filename: str) -> pd.DataFrame:
    """Loads and returns the stock data from a CSV file."""
    filepath = get_data_path(filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Converts 'Date' to datetime and sorts the dataframe by date."""
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

def main() -> pd.DataFrame:
    """Load and preprocess the stock data, and return the final DataFrame."""
    df_stock = load_stock_data("stock_data.csv")
    df_stock = preprocess_stock_data(df_stock)
    print("✅ Data loaded and processed successfully!")
    return df_stock

# Only execute if running directly
if __name__ == "__main__":
    df = main()
    print(df.head())

