"""
This file loads and cleans the Student Lifestyle dataset.

Run in terminal using "py clean_student.py"

Run this code in MySQL by writing "db_utils.write_student_table(clean_student_dataset(...))"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import STUDENT_CSV_PATH


def clean_student_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.rename(columns={
        "Student_ID": "student_id",
        "Study_Hours_Per_Day": "study_hours_per_day",
        "Extracurricular_Hours_Per_Day": "extracurricular_hours_per_day",
        "Sleep_Hours_Per_Day": "sleep_hours_per_day",
        "Social_Hours_Per_Day": "social_hours_per_day",
        "Physical_Activity_Hours_Per_Day": "physical_activity_hours_per_day",
        "GPA": "gpa",
        "Stress_Level": "stress_level",
    })

    stress_map = {
        "None": 0,
        "Low": 1,
        "Mild": 1,
        "Moderate": 2,
        "Medium": 2,
        "High": 3,
        "Very High": 4,
        "Severe": 4,
    }

    if "stress_level" in df.columns:
        mapped = df["stress_level"].map(stress_map)
        numeric_fallback = pd.to_numeric(df["stress_level"], errors="coerce")
        df["stress_level"] = mapped.fillna(numeric_fallback)

    numeric_cols = [
        "study_hours_per_day",
        "extracurricular_hours_per_day",
        "sleep_hours_per_day",
        "social_hours_per_day",
        "physical_activity_hours_per_day",
        "gpa",
        "stress_level",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    def cap(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return np.clip(series, lower, upper)

    for col in numeric_cols:
        df[col] = cap(df[col])

    df["total_engagement_hours"] = (
        df["study_hours_per_day"]
        + df["extracurricular_hours_per_day"]
        + df["social_hours_per_day"]
        + df["physical_activity_hours_per_day"]
    )

    scaler = StandardScaler()
    scale_cols = numeric_cols + ["total_engagement_hours"]
    scaled = scaler.fit_transform(df[scale_cols])

    scaled_df = pd.DataFrame(
        scaled,
        columns=[col + "_z" for col in scale_cols],
        index=df.index
    )

    df = pd.concat([df, scaled_df], axis=1)
    return df
