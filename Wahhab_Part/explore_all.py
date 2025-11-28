"""
This file is used to view and verify all SQL tables and the unified view.

Run in terminal using: py explore_all.py

This code will:
    - Display the row count and columns of the cleaned Student table
    - Display the row count and columns of the cleaned Sleep table
    - Display the row count and first few rows of the unified SQL view

Use this file to understand the data structure before beginning modeling or visualization work.
"""

import pandas as pd
from sqlalchemy import text
from db_utils import get_engine
from config import STUDENT_TABLE, SLEEP_TABLE, UNIFIED_VIEW

def show_table_preview(engine, table_name, n=[INPUT HOW MANY YOU WANNA VIEW]):
    print(f"\n=== {table_name} (first {n} rows) ===")
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table_name} LIMIT {n}"), conn)
        print(df)


def show_table_info(engine, table_name):
    print(f"\n--- {table_name} INFO ---")
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        print(f"Rows: {count}")

        columns = conn.execute(
            text(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                 f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{table_name}'")
        ).fetchall()
        print("Columns:")
        for c in columns:
            print(" -", c[0])


def main():
    engine = get_engine()

    show_table_info(engine, STUDENT_TABLE)
    show_table_preview(engine, STUDENT_TABLE, n=[INPUT HOW MANY YOU WANNA VIEW])

    show_table_info(engine, SLEEP_TABLE)
    show_table_preview(engine, SLEEP_TABLE, n=[INPUT HOW MANY YOU WANNA VIEW])

    show_table_info(engine, UNIFIED_VIEW)
    show_table_preview(engine, UNIFIED_VIEW, n=[INPUT HOW MANY YOU WANNA VIEW])

if __name__ == "__main__":
    main()
