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
    "Race ID": "race_id",
    "Track Info": "race_type",
    "Weather Icon": "weather",
    "Grade": "grade",
    "Finish Position": "finish_position",
    "Horse ID": "horse_id",
    "Age/Sex": "age_sex",
    "Final Time": "final_time",
    "Odds": "odds",
    "Horse Weight (kg)": "horse_weight"
}
df_cleaned = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

"""
3. Clean Race Type: extract numbers only
"""
df_cleaned["race_type"] = df_cleaned["race_type"].astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))

"""
4. Clean Horse Weight: remove comments in parentheses
"""
df_cleaned["horse_weight"] = df_cleaned["horse_weight"].astype(str).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x))

"""
5. Split Age/Sex into two features: Age and Sex
"""
df_cleaned["Age"] = df_cleaned["age_sex"].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if pd.notnull(x) else None)
df_cleaned["Sex"] = df_cleaned["age_sex"].apply(lambda x: re.findall(r'[A-Za-z]+', str(x))[0] if pd.notnull(x) else None)
df_cleaned = df_cleaned.drop(columns=["age_sex"])

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

df_cleaned["final_time"] = df_cleaned["final_time"].apply(convert_final_time_to_seconds)

"""
8. Clean Finish Position safely: ensure numeric integer only
"""
df_cleaned["finish_position"] = pd.to_numeric(df_cleaned["finish_position"], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=["finish_position"])
df_cleaned = df_cleaned[np.isfinite(df_cleaned["finish_position"])]
df_cleaned["finish_position"] = df_cleaned["finish_position"].astype(int)

"""
9. Create target variables: Top1 (Win) and Top3 (Top 3 Finish)
"""
df_cleaned["Top1"] = df_cleaned["finish_position"].apply(lambda x: 1 if x == 1 else 0)
df_cleaned["Top3"] = df_cleaned["finish_position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

"""
10. One-Hot Encode categorical features: Grade, Weather, Race Type
"""
one_hot_columns = ["grade", "weather", "race_type"]
df_cleaned.replace("NR", np.nan, inplace=True)
df_encoded = pd.get_dummies(df_cleaned, columns=one_hot_columns, prefix=one_hot_columns)
bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

"""
11. Normalize numerical features: Odds and Horse Weight
"""
df_encoded["odds"] = pd.to_numeric(df_encoded["odds"], errors='coerce')
df_encoded["horse_weight"] = pd.to_numeric(df_encoded["horse_weight"], errors='coerce')

scaler = MinMaxScaler()
df_encoded[["odds", "horse_weight"]] = scaler.fit_transform(df_encoded[["odds", "horse_weight"]])

"""
12. Save cleaned dataset
"""
df_encoded.columns = df_encoded.columns.str.lower()
output_path = "data/cleaned_race_results.csv"
df_encoded.to_csv(output_path, index=False)

