import pandas as pd
import re


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


df_cleaned.to_csv("clean_data/cleaned_race_results.csv", index=False)
