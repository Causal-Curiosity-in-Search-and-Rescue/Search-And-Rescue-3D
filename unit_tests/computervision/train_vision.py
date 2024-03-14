# Step 1: Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import dump,load
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load your dataset
CSV_PATH = "/home/docsun/Documents/git-repo/juliangdz/Search-And-Rescue-3D/unit_tests/computervision/data/texture_dataset.csv"
df = pd.read_csv(CSV_PATH)  # Replace 'your_file.csv' with your file path

# Step 3: Preprocess the data
# Convert 'box_type' column to categorical labels
mapping = {8: 0, 6: 1, 7: 2}
df['box_type'] = df['box_type'].map(mapping)

# Splitting data into features and target
X = df.drop(['box_type'], axis=1)
y = df['box_type']
print(X.head())
print(y.head())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training, validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Print the distribution of each class
print("Distribution in Training Set:\n", y_train.value_counts())
print("Distribution in Validation Set:\n", y_val.value_counts())

# Step 6: Train a classifier (for example, a RandomForestClassifier)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 7: Predict and evaluate on the validation set
y_pred = clf.predict(X_val)
print("Classification Report:\n", classification_report(y_val, y_pred))

# # Saving the  Model
# dump(clf,'resources/models/unsup_txture_clsf_rf.joblib')
# dump(scaler,'resources/models/unsup_txture_clsf_scaler.joblib')
