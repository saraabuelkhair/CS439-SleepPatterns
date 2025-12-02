Wahhab Part — Data Cleaning, SQL Pipeline, and Backend Integration

This folder includes all of my work for the CS439 Sleep Patterns project.
My part focuses on preparing and organizing both datasets, cleaning and structuring the data, and building the full SQL pipeline used for querying, merging, and analyzing sleep, stress, and lifestyle variables. I also helped set up the foundation for the modeling workflow by ensuring all datasets were standardized, indexed, and ready for analysis.

Folder Structure
1. data_preprocessing/

This folder contains all the scripts I wrote to clean, standardize, and transform both datasets:

clean_student.py

clean_sleep.py

These scripts include:

Renaming and normalizing column names

Converting stress categories to numeric scale

Fixing types and coercing bad values

Handling missing data using median/mode

Removing outliers using IQR capping

Feature engineering (e.g., total_engagement_hours)

Z-score scaling for all numeric fields

Exporting cleaned datasets for SQL and modeling

These cleaned files were later used for SQL ingestion, feature analysis, machine learning modeling, and visualization.

2. sql_pipeline/

This folder contains the scripts responsible for building and managing our SQL database.

Files include:

config.py

db_utils.py

explore_all.py

config.py

Stores all constants such as dataset paths, MySQL credentials, table names, and view names.

db_utils.py

This file handles:

Connecting to MySQL

Uploading cleaned datasets into the database

Creating all necessary indexes (sleep, stress, GPA)

Generating a unified view using UNION ALL

This view merges student and adult datasets into one combined table for cross-population analysis

explore_all.py

A quick inspection tool that:

Shows row counts

Lists all columns

Previews each table

Ensures the unified view is working correctly

3. analysis_support/

This folder contains the analysis pipeline that directly uses the cleaned datasets I prepared.

analysisandmodeling.py

This script runs:

EDA on student and adult datasets

Heatmaps, histograms, regression plots, and boxplots

GPA prediction models (Linear Regression & Random Forest)

High-stress classification model

K-means clustering

PCA visualization

Cluster summary profiles

Although modeling was assigned to another member, my cleaning and SQL integration ensured that this script ran on fully prepared, standardized data.

Summary of My Contribution

In my part of the project, I worked on:

Cleaning, transforming, and standardizing both datasets

Designing consistent schemas for SQL usage

Uploading cleaned data into MySQL with proper indexing

Creating a unified SQL view for comparing students and adults

Writing tools to validate database tables and structures

Preparing clean datasets for modeling, visualization, and Tableau analysis

Supporting the end-to-end workflow so all team members had reliable data

My goal was to build a strong backend foundation for the entire project by ensuring the data was organized, accurate, and ready for high-quality analysis.
