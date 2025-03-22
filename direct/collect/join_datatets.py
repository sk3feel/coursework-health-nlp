import pandas as pd

df1 = pd.read_csv("../data/kaggle_data/kaggle_dataset.csv")
df2 = pd.read_csv("../data/reddit_data/reddit_dataset.csv")

df1['Disease'] = df1['Disease'].str.lower().str.strip()
df2['Disease'] = df2['Disease'].str.lower().str.strip()

df1.columns = ['Disease'] + [f"Symptom_{i+1}" for i in range(df1.shape[1] - 1)]
df2.columns = ['Disease'] + [f"Symptom_{i+1}" for i in range(df2.shape[1] - 1)]

merged_df = pd.concat([df1, df2], ignore_index=True)

symptom_cols = [col for col in merged_df.columns if col.startswith("Symptom_")]
merged_df = merged_df.dropna(subset=symptom_cols, how='all')

merged_df.to_csv("../data/final_dataset.csv", index=False)
