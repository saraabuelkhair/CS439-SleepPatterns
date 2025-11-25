"""
This file loads and cleans the Sleep Health dataset

Run in terminal using: "py clean_sleep.py"

Load this cleaned dataset into MySQL using "db_utils.write_sleep_table(clean_sleep_dataset(...))"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_sleep_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.rename(columns={
        "Person ID": "person_id",
        "Gender": "gender",
        "Age": "age",
        "Occupation": "occupation",
        "Sleep Duration": "sleep_hours_per_day",
        "Quality of Sleep": "quality_of_sleep",
        "Physical Activity Level": "physical_activity_level",
        "Stress Level": "stress_level",
        "BMI Category": "bmi_category",
        "Blood Pressure": "blood_pressure",
        "Heart Rate": "heart_rate",
        "Daily Steps": "daily_steps",
        "Sleep Disorder": "sleep_disorder",
    })

    numeric_cols = [
        "age",
        "sleep_hours_per_day",
        "quality_of_sleep",
        "physical_activity_level",
        "stress_level",
        "heart_rate",
        "daily_steps",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = ["gender", "occupation", "bmi_category", "blood_pressure", "sleep_disorder"]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode().iloc[0] if df[col].mode().size > 0 else "Unknown"
            df[col] = df[col].fillna(mode_val).replace("", mode_val)

    def cap(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        return np.clip(series, lower, upper)

    for col in numeric_cols:
        df[col] = cap(df[col])

    bp_split = df["blood_pressure"].astype(str).str.split("/", expand=True)
    if bp_split.shape[1] == 2:
        df["systolic_bp"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["diastolic_bp"] = pd.to_numeric(bp_split[1], errors="coerce")
    else:
        df["systolic_bp"] = np.nan
        df["diastolic_bp"] = np.nan

    df[["systolic_bp", "diastolic_bp"]] = df[["systolic_bp", "diastolic_bp"]].fillna(
        df[["systolic_bp", "diastolic_bp"]].median()
    )

    scale_cols = numeric_cols + ["systolic_bp", "diastolic_bp"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[scale_cols])

    scaled_df = pd.DataFrame(
        scaled,
        columns=[col + "_z" for col in scale_cols],
        index=df.index,
    )

    df = pd.concat([df, scaled_df], axis=1)
    return df
