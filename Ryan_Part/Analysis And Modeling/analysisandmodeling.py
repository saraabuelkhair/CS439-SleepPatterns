import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


sns.set_theme(style="whitegrid", context="talk")

# load data
student_df = pd.read_csv("cleaned_student_dataset.csv")
sleep_df = pd.read_csv("cleaned_sleep_dataset.csv")


# EDA for student data
numeric_cols = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa",
    "stress_level"
]

# continuous variables
cont_cols = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa"
]

pretty_names = {
    "study_hours_per_day": "Study hours per day",
    "extracurricular_hours_per_day": "Extracurricular hours per day",
    "sleep_hours_per_day": "Sleep hours per day",
    "social_hours_per_day": "Social hours per day",
    "physical_activity_hours_per_day": "Physical activity hours per day",
    "gpa": "GPA"
}

fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
axes = axes.flatten()

for ax, col in zip(axes, cont_cols):
    sns.histplot(student_df[col], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribution of {pretty_names[col]}", fontsize=13)
    ax.set_xlabel(pretty_names[col], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

plt.show()

# stress levels distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=student_df, x="stress_level")
plt.title("Stress Level Distribution", fontsize=14)
plt.xlabel("Stress Level", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

# EDA for sleep data
sleep_numeric = [
    "sleep_hours_per_day",
    "quality_of_sleep",
    "physical_activity_level",
    "stress_level",
    "heart_rate",
    "daily_steps"
]

# histograms for sleep data
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
axes = axes.flatten()

sleep_pretty = {
    "sleep_hours_per_day": "Sleep Hours per Day",
    "quality_of_sleep": "Sleep Quality",
    "physical_activity_level": "Physical Activity Level",
    "stress_level": "Stress Level",
    "heart_rate": "Heart Rate",
    "daily_steps": "Daily Steps"
}

for ax, col in zip(axes, sleep_numeric):
    sns.histplot(sleep_df[col], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribution of {sleep_pretty[col]}", fontsize=13)
    ax.set_xlabel(sleep_pretty[col], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

plt.show()

# heatmap for sleep dataset
corr_sleep = sleep_df[sleep_numeric].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_sleep, dtype=bool))

sns.heatmap(
    corr_sleep,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    annot_kws={"size": 11}
)
plt.title("Correlation Heatmap – Sleep Health (Adults)", fontsize=18)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# sleep vs stress for sleep dataset
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=sleep_df,
    x="stress_level",
    y="sleep_hours_per_day",
    showfliers=False
)
plt.title("Adult Sleep Hours per Day by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Sleep Hours per Day")
plt.tight_layout()
plt.show()


# heatmap for student data
corr = student_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    annot_kws={"size": 11}
)
plt.title("Correlation Heatmap – Student Lifestyle", fontsize=18)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# scatterplots with regression lines for GPA vs sleep/study hours
plt.figure(figsize=(7, 5))
sns.regplot(
    data=student_df,
    x="sleep_hours_per_day",
    y="gpa",
    scatter_kws={"alpha": 0.15, "s": 20},
    line_kws={"color": "red", "linewidth": 2},
)
plt.title("Sleep Hours vs GPA (Regression Line)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
sns.regplot(
    data=student_df,
    x="study_hours_per_day",
    y="gpa",
    scatter_kws={"alpha": 0.3},
    line_kws={"linewidth": 2}
)
plt.title("Study Hours vs GPA (with Regression Line)")
plt.tight_layout()
plt.show()

# boxplot for sleep hours distribution by stress level
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=student_df,
    x="stress_level",
    y="sleep_hours_per_day"
)
plt.title("Sleep Hours per Day by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Sleep Hours per Day")
plt.tight_layout()
plt.show()

# boxplot for study hours distribution by stress level
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=student_df,
    x="stress_level",
    y="study_hours_per_day",
    showfliers=False
)
plt.title("Study Hours per Day by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Study Hours per Day")
plt.tight_layout()
plt.show()


# regession modeling for predicting GPA
feature_cols = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "stress_level"
]

X = student_df[feature_cols]
y = student_df["gpa"]

# train/test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

print("\nLINEAR REGRESSION RESULTS")
print("R²:", r2_score(y_test, y_pred_lin))
print("MAE:", mean_absolute_error(y_test, y_pred_lin))

# random forest regression
rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
).fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)

print("\nRANDOM FOREST REGRESSION RESULTS")
print("R²:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))

# feature importance
importances = pd.Series(
    rf_reg.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(importances)


# predicting high stress
# high stress: stress_level == 3
student_df["high_stress"] = (
    student_df["stress_level"] == 3
).astype(int)

X = student_df[feature_cols]
y = student_df["high_stress"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

print("\nHIGH-STRESS CLASSIFICATION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – High Stress Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# feature importance
clf_imp = pd.Series(rf_clf.feature_importances_, index=feature_cols)
print("\nSTRESS CLASSIFIER FEATURE IMPORTANCE")
print(clf_imp.sort_values(ascending=False))

# k-means clustering 
cluster_features = [
    "study_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "extracurricular_hours_per_day",
    "physical_activity_hours_per_day"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(student_df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
student_df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCLUSTER PROFILES")
cluster_profile = student_df.groupby("cluster")[cluster_features + ["gpa", "stress_level"]].mean()
print(cluster_profile)

# PCA visualization of clusters
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 6))
sns.scatterplot(
    x=coords[:, 0],
    y=coords[:, 1],
    hue=student_df["cluster"],
    palette="deep",
    alpha=0.6
)
plt.title("PCA Projection of Lifestyle Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()







