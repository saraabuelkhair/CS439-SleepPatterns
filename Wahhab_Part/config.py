"""
Configuration file for the project

This file stores all paths, MySQL settings, table names, and other variables that will be used across the cleaning and database scripts.

Do NOT run this file directly.
"""

Data_Directory = "Data Directory"

STUDENT_CSV_PATH = f"{Data_Directory}/student_lifestyle_dataset.csv"
SLEEP_CSV_PATH   = f"{Data_Directory}/sleep_health_dataset.csv"

MYSQL_USER = "[USE YOUR OWN USERNAME]"
MYSQL_PASSWORD = "[USE YOUR OWN PASSWORD]"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB   = "student_lifestyle_db"

STUDENT_TABLE = "student_lifestyle_clean"
SLEEP_TABLE   = "sleep_health_clean"
UNIFIED_VIEW  = "unified_sleep_stress_view"
