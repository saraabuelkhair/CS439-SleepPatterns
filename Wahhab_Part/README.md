# **Wahhab Part — Data Cleaning, SQL Pipeline, and Backend Integration**

This folder includes all of my work for the CS439 Sleep Patterns project.  
My part focuses on preparing and organizing both datasets, cleaning and structuring the data, and building the full SQL pipeline used for querying, merging, and analyzing sleep, stress, and lifestyle variables. I also helped set up the foundation for the modeling workflow by ensuring all datasets were standardized, indexed, and ready for analysis.

---

# **Folder Structure**

---

## **1. data_preprocessing/**

This folder contains all the scripts I wrote to clean, standardize, and transform both datasets:

- `clean_student.py`
- `clean_sleep.py`

These scripts include:

- Renaming and normalizing column names  
- Converting stress categories to numeric values  
- Fixing incorrect data types and coercing invalid values  
- Handling missing data using median/mode  
- Removing outliers using IQR capping  
- Engineering new features (e.g., `total_engagement_hours`)  
- Applying z-score scaling to all numeric fields  
- Exporting cleaned datasets for SQL and modeling  

These cleaned files were later used for SQL ingestion, feature analysis, machine learning modeling, and visualization.

---

## **2. sql_pipeline/**

This folder contains the scripts responsible for building and managing our SQL database.

Files include:

- `config.py`
- `db_utils.py`
- `explore_all.py`

### **config.py**
- Stores constants such as dataset paths, MySQL credentials, table names, and unified view names.

### **db_utils.py**
This script handles:

- Connecting to MySQL  
- Uploading cleaned datasets into the database  
- Creating necessary indexes (sleep hours, stress, GPA)  
- Generating a unified SQL view using `UNION ALL`  
  - This combined view merges student and adult datasets for cross-population analysis  

### **explore_all.py**
A quick inspection tool that:

- Shows row counts for each table  
- Lists all columns in each table  
- Previews sample rows  
- Ensures the unified view is working correctly  

---

## **3. analysis_support/**

This folder contains the analysis pipeline that directly uses the cleaned datasets I prepared.

- `analysisandmodeling.py`

This script runs:

- Exploratory Data Analysis (EDA) on student and adult datasets  
- Heatmaps, histograms, regression plots, and boxplots  
- GPA prediction models (Linear Regression & Random Forest)  
- High-stress classification model  
- K-means clustering for lifestyle grouping  
- PCA visualization for cluster interpretation  
- Cluster summary profiles  

Although modeling tasks were assigned to another member, my data cleaning and SQL integration ensured this script ran on fully cleaned, consistent, and standardized data.

---

# **Summary of My Contribution**

In my part of the project, I worked on:

- Cleaning, transforming, and standardizing both datasets  
- Designing consistent schemas for SQL ingestion and querying  
- Uploading cleaned datasets into MySQL with proper indexing  
- Creating a unified SQL view for comparing students and adults  
- Writing tools to validate database tables and structure  
- Preparing clean datasets for modeling, visualization, and Tableau work  
- Supporting the entire pipeline to ensure reliable data flow for all team members  

My goal was to build a strong backend foundation for the entire project by ensuring the data was organized, accurate, and ready for high-quality analysis.

