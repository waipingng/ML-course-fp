import pandas as pd
import re
import numpy as np

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
    "Weight (kg)": "jockey_weight",
    "Horse Weight (kg)": "horse_weight",
    "Weather Icon": "weather",
    "Grade": "grade",
    "Finish Position": "finish_position",
    "Horse ID": "horse_id",
    "Age/Sex": "age_sex",
    "Final Time": "final_time",
    "Favorite": "favorite",
    "Bracket Number": "bracket_number"

}
df_cleaned = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

"""
3. Clean Race Type: extract numbers only (track distance in meters)
"""
df_cleaned["race_type"] = df_cleaned["race_type"].astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))

"""
3.5 Extract numeric track distance BEFORE one-hot encoding
"""
df_cleaned["track_distance"] = pd.to_numeric(df_cleaned["race_type"], errors="coerce")
df_cleaned["track_distance"] = df_cleaned["track_distance"].fillna(2000)

"""
4. Clean Horse Weight: remove comments in parentheses and calculate total weight
"""
df_cleaned["horse_weight"] = df_cleaned["horse_weight"].astype(str).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x))
df_cleaned["horse_weight"] = pd.to_numeric(df_cleaned["horse_weight"], errors="coerce")
df_cleaned["jockey_weight"] = pd.to_numeric(df_cleaned["jockey_weight"], errors="coerce")
df_cleaned["total_weight"] = df_cleaned["horse_weight"] + df_cleaned["jockey_weight"]

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
11. Calculate speed (m/s) = track_distance / final_time
"""
df_encoded["speed_mps"] = df_encoded["track_distance"] / df_encoded["final_time"]

"""
12. Calculate historical average speed (excluding current race)
"""
df_encoded["avg_speed_mps"] = 0.0
horse_speed_history = df_encoded.groupby("horse_id")["speed_mps"].apply(lambda x: x.index.tolist()).to_dict()

for horse_id, indices in horse_speed_history.items():
    cumulative_speed = 0.0
    for i, idx in enumerate(indices):
        if i == 0:
            avg_speed = 0.0
        else:
            avg_speed = cumulative_speed / i
        df_encoded.at[idx, "avg_speed_mps"] = avg_speed
        cumulative_speed += df_encoded.at[idx, "speed_mps"]

"""
13. Calculate historical average final time (excluding current race)
"""
df_encoded["avg_final_time_hist"] = 0.0
horse_time_history = df_encoded.groupby("horse_id")["final_time"].apply(lambda x: x.index.tolist()).to_dict()

for horse_id, indices in horse_time_history.items():
    cumulative_time = 0.0
    for i, idx in enumerate(indices):
        if i == 0:
            avg_time = 0.0
        else:
            avg_time = cumulative_time / i
        df_encoded.at[idx, "avg_final_time_hist"] = avg_time
        cumulative_time += df_encoded.at[idx, "final_time"]

"""
13.9 Filter to races with exactly 16 horses
"""
df_encoded = df_encoded.groupby("race_id").filter(lambda x: len(x) == 16)
df_encoded = df_encoded.reset_index(drop=True)
df_encoded_count = df_encoded["race_id"].nunique()
print("Number of races with exactly 16 horses:", df_encoded_count)

"""
14. Save cleaned dataset (only 16-horse races)
"""
df_encoded.columns = df_encoded.columns.str.lower()
output_path = "data/cleaned_race_results_for_lightgbm.csv"
df_encoded.to_csv(output_path, index=False)




