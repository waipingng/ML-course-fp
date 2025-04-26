import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np


file_path = "data/race_results_with_ids.csv" 
df = pd.read_csv(file_path)


columns_to_keep = {
    "Race ID": "Race ID",
    "Track Info": "Race Type",
    "Weather Icon": "Weather",
    "Grade": "Grade",
    "Finish Position": "Finish Position",
    "Horse ID": "Horse ID",
    "Age/Sex": "Age/Sex",
    "Final Time": "Final Time",
    "Odds": "Odds",
    "Horse Weight (kg)": "Horse Weight"
}
df_cleaned = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)


df_cleaned["Race Type"] = df_cleaned["Race Type"].astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))


df_cleaned["Horse Weight"] = df_cleaned["Horse Weight"].astype(str).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x))


df_cleaned["Age"] = df_cleaned["Age/Sex"].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if pd.notnull(x) else None)
df_cleaned["Sex"] = df_cleaned["Age/Sex"].apply(lambda x: re.findall(r'[A-Za-z]+', str(x))[0] if pd.notnull(x) else None)
df_cleaned = df_cleaned.drop(columns=["Age/Sex"]) 


df_cleaned["Top1"] = df_cleaned["Finish Position"].apply(lambda x: 1 if str(x) == "1" else 0)


df_cleaned["Top3"] = df_cleaned["Finish Position"].apply(lambda x: 1 if str(x) in ["1", "2", "3"] else 0)


one_hot_columns = ["Grade", "Weather", "Race Type"]
df_cleaned.replace("NR", np.nan, inplace=True)
df_encoded = pd.get_dummies(df_cleaned, columns=one_hot_columns, prefix=one_hot_columns)
for col in df_encoded.columns:
    if df_encoded[col].dropna().isin([True, False, 0, 1]).all():
        df_encoded[col] = df_encoded[col].astype(int)

df_encoded["Odds"] = pd.to_numeric(df_encoded["Odds"], errors='coerce')
df_encoded["Horse Weight"] = pd.to_numeric(df_encoded["Horse Weight"], errors='coerce')

scaler = MinMaxScaler()
df_encoded[["Odds", "Horse Weight"]] = scaler.fit_transform(df_encoded[["Odds", "Horse Weight"]])


df_encoded.to_csv("clean_data/cleaned_race_results.csv", index=False)
