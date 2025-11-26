"""
Database utility functions for the project.

This file handles all interactions with MySQL, including:
    - Connecting to the database
    - Writing the cleaned Student dataset to MySQL
    - Writing the cleaned Sleep dataset to MySQL
    - Creating indexes for faster queries
    - Creating the unified SQL view that combines both datasets

You do NOT run this file directly.

Make sure config.py is also updated to run this program
"""

import pandas as pd
from sqlalchemy import create_engine, text
from config import (
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DB,
    STUDENT_TABLE,
    SLEEP_TABLE,
    UNIFIED_VIEW,
)


def get_engine():
    conn_str = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )
    return create_engine(conn_str, echo=False)


def write_student_table(df):
    engine = get_engine()
    df.to_sql(STUDENT_TABLE, con=engine, if_exists="replace", index=False)


def write_sleep_table(df):
    engine = get_engine()
    df.to_sql(SLEEP_TABLE, con=engine, if_exists="replace", index=False)


def create_indexes_and_view():
    engine = get_engine()
    with engine.begin() as conn:
        # Make index dropping safe by ignoring "doesn't exist" errors
        for sql in [
            f"ALTER TABLE {STUDENT_TABLE} DROP INDEX idx_student_sleep;",
            f"ALTER TABLE {STUDENT_TABLE} DROP INDEX idx_student_stress;",
            f"ALTER TABLE {STUDENT_TABLE} DROP INDEX idx_student_gpa;",
            f"ALTER TABLE {SLEEP_TABLE} DROP INDEX idx_sleep_sleep;",
            f"ALTER TABLE {SLEEP_TABLE} DROP INDEX idx_sleep_stress;",
        ]:
            try:
                conn.execute(text(sql))
            except Exception:
                pass

        conn.execute(
            text(
                f"CREATE INDEX idx_student_sleep "
                f"ON {STUDENT_TABLE} (sleep_hours_per_day);"
            )
        )
        conn.execute(
            text(
                f"CREATE INDEX idx_student_stress "
                f"ON {STUDENT_TABLE} (stress_level);"
            )
        )
        conn.execute(
            text(
                f"CREATE INDEX idx_student_gpa "
                f"ON {STUDENT_TABLE} (gpa);"
            )
        )

        conn.execute(
            text(
                f"CREATE INDEX idx_sleep_sleep "
                f"ON {SLEEP_TABLE} (sleep_hours_per_day);"
            )
        )
        conn.execute(
            text(
                f"CREATE INDEX idx_sleep_stress "
                f"ON {SLEEP_TABLE} (stress_level);"
            )
        )

        # Recreate the unified view
        conn.execute(text(f"DROP VIEW IF EXISTS {UNIFIED_VIEW};"))

        view_sql = f"""
CREATE VIEW {UNIFIED_VIEW} AS

-- Student dataset (college)
SELECT
    student_id AS entity_id,
    'student'  AS source,
    sleep_hours_per_day,
    stress_level,
    physical_activity_hours_per_day AS physical_activity_raw,
    gpa,
    NULL AS heart_rate,
    NULL AS daily_steps
FROM {STUDENT_TABLE}

UNION ALL

-- General adult dataset
SELECT
    person_id AS entity_id,
    'general' AS source,
    sleep_hours_per_day,
    stress_level,
    physical_activity_level AS physical_activity_raw,
    NULL AS gpa,
    heart_rate,
    daily_steps
FROM {SLEEP_TABLE};
"""
        conn.execute(text(view_sql))
