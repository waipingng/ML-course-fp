import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
1. Load dataset
"""
file_path = "data/race_results_with_ids.csv"
df = pd.read_csv(file_path)

"""
2. Select and rename useful columns
"""
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

"""
3. Clean Race Type: extract numbers only
"""
df_cleaned["Race Type"] = df_cleaned["Race Type"].astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))

"""
4. Clean Horse Weight: remove comments in parentheses
"""
df_cleaned["Horse Weight"] = df_cleaned["Horse Weight"].astype(str).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x))

"""
5. Split Age/Sex into two features: Age and Sex
"""
df_cleaned["Age"] = df_cleaned["Age/Sex"].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if pd.notnull(x) else None)
df_cleaned["Sex"] = df_cleaned["Age/Sex"].apply(lambda x: re.findall(r'[A-Za-z]+', str(x))[0] if pd.notnull(x) else None)
df_cleaned = df_cleaned.drop(columns=["Age/Sex"])

"""
6. Encode Sex properly: One-hot encode horse type
"""
df_cleaned["Sex"] = df_cleaned["Sex"].fillna("Unknown")
df_cleaned = pd.get_dummies(df_cleaned, columns=["Sex"], prefix="Sex")
bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)

"""
7. Convert Final Time to total seconds
"""
def convert_final_time_to_seconds(time_str):
    try:
        if isinstance(time_str, (float, int)):
            return time_str
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + float(seconds)
    except Exception:
        return np.nan

df_cleaned["Final Time"] = df_cleaned["Final Time"].apply(convert_final_time_to_seconds)

"""
8. Clean Finish Position safely: ensure numeric integer only
"""
df_cleaned["Finish Position"] = pd.to_numeric(df_cleaned["Finish Position"], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=["Finish Position"])
df_cleaned = df_cleaned[np.isfinite(df_cleaned["Finish Position"])]
df_cleaned["Finish Position"] = df_cleaned["Finish Position"].astype(int)

"""
9. Create target variables: Top1 (Win) and Top3 (Top 3 Finish)
"""
df_cleaned["Top1"] = df_cleaned["Finish Position"].apply(lambda x: 1 if x == 1 else 0)
df_cleaned["Top3"] = df_cleaned["Finish Position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

"""
10. One-Hot Encode categorical features: Grade, Weather, Race Type
"""
one_hot_columns = ["Grade", "Weather", "Race Type"]
df_cleaned.replace("NR", np.nan, inplace=True)
df_encoded = pd.get_dummies(df_cleaned, columns=one_hot_columns, prefix=one_hot_columns)

"""
11. Normalize numerical features: Odds and Horse Weight
"""
df_encoded["Odds"] = pd.to_numeric(df_encoded["Odds"], errors='coerce')
df_encoded["Horse Weight"] = pd.to_numeric(df_encoded["Horse Weight"], errors='coerce')

scaler = MinMaxScaler()
df_encoded[["Odds", "Horse Weight"]] = scaler.fit_transform(df_encoded[["Odds", "Horse Weight"]])

"""
12. Save cleaned dataset
"""
output_path = "clean_data/cleaned_race_results.csv"
df_encoded.to_csv(output_path, index=False)

