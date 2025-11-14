import pandas as pd

df = pd.read_csv("aggregated_data.csv")
df["saved_at"] = pd.to_datetime(df["saved_at"], errors="coerce")

print("Unique dates:", df["saved_at"].dt.date.nunique())
print(df["saved_at"].dt.date.value_counts())
