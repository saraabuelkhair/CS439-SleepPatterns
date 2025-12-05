#imports
#first thing we're doing is  importing the libraries and specific methods 
#that we will be using in this project for the EDA and modeling. 
#we are using pandas for data handling and manipulation, numpy for 
#math and numeric operations, matplotlib and seaborn for data
#visualization, and variuous modules from sklearn for our modeling, 
#including train_test splotting, a linear regression model
#a random forest regression model, a random forest classifier model, 
#and r2_score, mean_absolute_error, accuracy_score, classification_report,
#and confusion_matrix for our model evaluation matrix. 
#we are also using standardscaler for feature scaling, 
#KMeans for clustering, and PCA for dimensional reduction so that we 
#can visualize our K means clusters.
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
#loading data from the cleaned csv files done by Wahhab
student_df = pd.read_csv("cleaned_student_dataset.csv")
sleep_df = pd.read_csv("cleaned_sleep_dataset.csv")

# EDA for student data
#Several plots to help us visualize the student dataset.
#nothing done to the data, simply visualizing it to understand it better.

#all columns of the dataset that are numeric
numeric_cols = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa",
    "stress_level"
]

# continuous columns. all numeric columns except stress level
#since it is categorical
cont_cols = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa"
]


#renaming columns for readability
pretty_names = {
    "study_hours_per_day": "Study hours per day",
    "extracurricular_hours_per_day": "Extracurricular hours per day",
    "sleep_hours_per_day": "Sleep hours per day",
    "social_hours_per_day": "Social hours per day",
    "physical_activity_hours_per_day": "Physical activity hours per day",
    "gpa": "GPA"
}


#histograms that show us the distribution of each continuous variable
#in the student dataset
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
axes = axes.flatten()

for ax, col in zip(axes, cont_cols):
    sns.histplot(student_df[col], bins=20, ax=ax)
    ax.set_title(f"Distribution of {pretty_names[col]}", fontsize=13)
    ax.set_xlabel(pretty_names[col], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

plt.show()

#histogram that shows us the distribution of stress levels
#in the student dataset 
plt.figure(figsize=(5, 4))
sns.countplot(data=student_df, x="stress_level")
plt.title("(Student Dataset) Stress Level Distribution", fontsize=14)
plt.xlabel("Stress Level", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

# boxplot for study hours distribution by stress level.
#allows us to visualize the spread of study hours per day by stress level
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=student_df,
    x="stress_level",
    y="study_hours_per_day",
    showfliers=False
)

plt.title("(Student Dataset) Study Hours per Day by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Study Hours per Day")
plt.tight_layout()
plt.show()

# boxplot for sleep hours distribution by stress level
#allows us to visualize the spread of sleep hours per day by stress level
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=student_df,
    x="stress_level",
    y="sleep_hours_per_day"
)

plt.title("(Student Dataset) Sleep Hours per Day by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Sleep Hours per Day")
plt.tight_layout()
plt.show()

#overall these boxplots allow us to visualize the relationships between
#how stress tends to correlate with how much students study and sleep.

# heatmap for student data
#allows us to visualize any correlations in our student dataset.
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
plt.title("Correlation Heatmap – Student Dataset", fontsize=18)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# scatterplots with regression lines for GPA vs sleep/study hours
#allows us to visualize two important relationships that we are 
#investigating: sleep hours vs. GPA and study hours vs. GPA. 
#through these plots we can see that there is a clear linear relationship
#between study hours and GPA, while there is not much of a relationship
#between sleep hours and GPA.

plt.figure(figsize=(7, 5))
sns.regplot(
    data=student_df,
    x="sleep_hours_per_day",
    y="gpa",
    scatter_kws={"alpha": 0.15, "s": 20},
    line_kws={"color": "red", "linewidth": 2},
)
plt.title("(Student Dataset) Sleep Hours vs GPA with Regression Line)")
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
plt.title("(Student Dataset) Study Hours vs GPA with Regression Line")
plt.tight_layout()
plt.show()

# EDA for adult data
#allows us to get a broad overview of our adult dataset through 
#various visualizations. 

#numeric columns in sleep/adult dataset
sleep_numeric = [
    "sleep_hours_per_day",
    "quality_of_sleep",
    "physical_activity_level",
    "stress_level",
    "heart_rate",
    "daily_steps"
]

# histograms for sleep data
#lets us visualize the distribution of each numeric variable in the 
#adult dataset. 
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
axes = axes.flatten()

#renaming columns for readability
sleep_pretty = {
    "sleep_hours_per_day": "Sleep Hours per Day",
    "quality_of_sleep": "Sleep Quality",
    "physical_activity_level": "Physical Activity Level",
    "stress_level": "Stress Level",
    "heart_rate": "Heart Rate",
    "daily_steps": "Daily Steps"
}


#histograms for each numeric column in the adult dataset. 
#allows us to visualize the distribution of each variable 
for ax, col in zip(axes, sleep_numeric):
    sns.histplot(sleep_df[col], bins=20, ax=ax)
    ax.set_title(f"Distribution of {sleep_pretty[col]}", fontsize=13)
    ax.set_xlabel(sleep_pretty[col], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

plt.show()


#distribution of stress levels in the adult dataset.
#separate from numeric histograms because stress level is categorical
plt.figure(figsize=(5, 4))
sns.countplot(data=sleep_df, x="stress_level")
plt.title("(Adult Dataset) Stress Level Distribution", fontsize=14)
plt.xlabel("Stress Level", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

# heatmap for sleep/adult dataset
#allows us to investigate any correlations in the adult dataset. 
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
 
plt.title("Correlation Heatmap – Adult Dataset", fontsize=18)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# boxplot for sleep vs stress for adult dataset
#allows us to visualize the distribution of sleep hours per stress level
#for the adult dataset.
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=sleep_df,
    x="stress_level",
    y="sleep_hours_per_day",
    showfliers=False
)

plt.title("(Adult Dataset) Sleep Hours per Day by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Sleep Hours per Day")
plt.tight_layout()
plt.show()

# regession modeling for predicting GPA
#regression models (linear and random forest) which take in the 
#lifestyle factors from the student dataset and predict GPA. 

#features that the models will use ot predict GPA
feature_cols = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "stress_level"
]

#defines the feature cols as the input variables and
#GPA as the output variable for training
X = student_df[feature_cols]
y = student_df["gpa"]

# train/test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling features
#certain features have different numeric ranges that could throw off
#how the model weights them. so we scale the features so that the model
#can accurately weight them and factor them into the prediction. 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# linear regression
#linear regression tries to model GPA as a weighted combination 
#of the input features. it assigns a coefficient to each feature,
#which indicates how strongly it increases or decreases GPA.
#linear regression is most effective when there is a roughly linear
#relationship between the input features and the output variable,
#in our case the GPA.

#instantiating the linear regression model and fitting it to our
#training data. this is where it learns the coefficients for
#each feature.
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

#this is where the model predicts the GPAs for each student in the dataset.
#it uses the learned coefficients from the previous step
y_pred_lin = lin_reg.predict(X_test_scaled)


#assessment of our linear regression model performance using 
#R squared and mean absolute error metrics, by passing in the test data
#GPAs and the predicted GPAs from the model. 
#R squared tells us how much of the variation in GPA the model
#is able to explain. 
#our R squared value is 0.55, which means that our model can explain
#55% of the variation in GPA based on the factors we inputted. 
#The remaining 45% is likely due to factors not included in our dataset.
#R squared helps us assess the predictive power of the features we 
#use for the model. 
#mean absolute error tells us on average how far off our model's
#GPA prediction is. Our model's MAE is about 0.16 which tells us that
#the model on average predicts GPAs either 0.16 points higher or lower than
#the actual GPA.
print("\nLINEAR REGRESSION RESULTS")
print("R²:", r2_score(y_test, y_pred_lin))
print("MAE:", mean_absolute_error(y_test, y_pred_lin))

# random forest regression
#random forest regression builds multiple decision trees and each one
#makes its own prediction based on the input features. The final 
#prediction averages the prediction from all of the trees to get the
#final GPA prediction. 
#can be more accurate than linear regression if there are nonlinear 
#relationships between features that influence GPA.
#in general, random forest is ideal when there are complex relationships
#in the data while linear regression is ideal when relationships are
#more simple and linear. 

#instantiating the random forest regression model with 200 trees and
#fitting it to the training data
rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
).fit(X_train, y_train)

#using the model to produce predictions based on the input test data
y_pred_rf = rf_reg.predict(X_test)


#evaluating random forest model performance based on the same metrics as 
#the linear regression model, R squared and mean absolute error.
#the R squared is slightly lower compared to the linear regression model
#which means that this model can explain less of the variance in GPA.
#this suggests that there is a relatively linear relationship between the 
#input features and GPA. 
#Mean absolute error is also slightly higher, suggesting that the 
#linear regression model is also slightly more accurate. 
print("\nRANDOM FOREST REGRESSION RESULTS")
print("R²:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))

# feature importance
#determines which features were most influential in accurately
#predicting GPA in our random forest regression model. study hours per day
#is by far the most important feature, with 0.596 importance score. 
#this helps us understand why the linear regression model might have
#performed better, since in our data GPA has a strong linear relationship
#with study hours per day. 
importances = pd.Series(
    rf_reg.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(importances)


# predicting high stress (high stress means stress level of 3)
#we are predicting high stress using a random forest classifier model.

#defining high stress as a boolean where high stress = true if stress level
#equals 3 and false otherwise. 
student_df["high_stress"] = (
    student_df["stress_level"] == 3
).astype(int)

#features that we will input into the model. all of the lifestyle features
#except for stress level since that is what we are trying to classify. 
feature_cols_no_stress = [
    "study_hours_per_day",
    "extracurricular_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day"
]

X = student_df[feature_cols_no_stress]
y = student_df["high_stress"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#instantiating the random forest classifier model and training it 
#on the training data
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_train)

#using the model we created and trained to produce predictions for whether
#or not each student is high stress, using the test data
y_pred = rf_clf.predict(X_test)

#evaluating the classifier model performance using precision, recall, 
#and f1 score metrics. we also generate a report that gives us an 
#overview of each of these metrics.
#classifier model appears to be perfectly accurate, indicating that
#a tree model can very effectively classify high stress students
#based on our data.
print("\nHIGH-STRESS CLASSIFICATION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#visualizing our classifier model performance using a confusion matrix,
#which shows us how many TP, FP, TN, and FN our model produced.
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – High Stress Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# feature importance
#feature importance for the random forest classifier model. tells us 
#which features were most influential in accurately classifying high stress.
#similarly to the regression model, study hours per day is significantly
#the most important metric for classifying high stress, but here we
#can also see that sleep hours per day is also highly influential at 0.309.
#based off of these results, we can see that the classifier model 
#can classify high stress almost entirely by these two features alone,
#which can help explain why the model is so accurate. 
clf_imp = pd.Series(rf_clf.feature_importances_, index=feature_cols_no_stress)
print("\nSTRESS CLASSIFIER FEATURE IMPORTANCE")
print(clf_imp.sort_values(ascending=False))

# k-means clustering 
#using lifestyle features from our student dataset 
#to identify distinct clusters that represent 
#lifestyle "groups" of students. once these clusters are established, 
#we will analyze how GPA and stress differ by cluster groups.
cluster_features = [
    "study_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "extracurricular_hours_per_day",
    "physical_activity_hours_per_day"
]

#scaling features similarly to how we did for regression models 
#to ensure that clusters are accurate.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(student_df[cluster_features])

#using K means clustering to assign each student to a cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
student_df["cluster"] = kmeans.fit_predict(X_scaled)

#here, we can see that the k means clustering identifed three clusters.
#generally, the three categories are:
#0: highest study hours, sleep hours, extracurricular hours, GPA, and
#stress. lowest social and physical activity hours (academic oriented?)
#1: lowest study hours, sleep hours extracurricular hours, GPA, and stress.
#highest physical activity. (athletes?)
#2: median sleep hours, extracurricular hours, physical activity hours,
#stress, and GPA. highest social hours by a sigificant margin (greek life?)
print("\nCLUSTER PROFILES")
cluster_profile = student_df.groupby("cluster")[cluster_features + ["gpa", "stress_level"]].mean()
print(cluster_profile)

# PCA visualization of clusters
#clusters are hard to visualize as they are since each cluster is
#5 dimensional. PCA allows us to compress these dimensions into a 
#2D space so that we can visualize the clusters. finds the two axes
#that captures the most variation in the data. once these axes
#are established we plot each student according to their respective 
#cluster and where they lie on the grid according to the two principal
#axes.

#instantiating the PCA model and fitting it into our scaled data
#that we used for the clusters
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

#plotting our clusters after the PCA transformation 
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




