import os
import pandas as pd

# --------- CONFIG ---------
TEMP_FOLDER = "temp"
MERGED_OUTPUT = "nifty500_sentiment_merged.csv"

# --------- MERGE LOGIC ---------
all_files = [f for f in os.listdir(TEMP_FOLDER) if f.endswith(".csv")]

merged_df = pd.DataFrame()

for file in all_files:
    file_path = os.path.join(TEMP_FOLDER, file)
    try:
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    except Exception as e:
        print(f"❌ Failed to read {file}: {e}")

# --------- SAVE FINAL MERGED CSV ---------
merged_df.to_csv(MERGED_OUTPUT, index=False)
print(f"✅ All sentiment files merged into: {MERGED_OUTPUT}")

