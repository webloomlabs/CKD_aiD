import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv("Chronic_Kidney_Dsease_data.csv")

# Drop unnecessary columns
df = df.drop(columns=["DoctorInCharge", "PatientID"])

# Separate features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Identify categorical and numeric columns
categorical_cols = [
    'Gender','Ethnicity','SocioeconomicStatus','EducationLevel',
    'Smoking','FamilyHistoryKidneyDisease','FamilyHistoryHypertension',
    'FamilyHistoryDiabetes','PreviousAcuteKidneyInjury','UrinaryTractInfections',
    'ACEInhibitors','Diuretics','Statins','AntidiabeticMedications',
    'Edema','HeavyMetalsExposure','OccupationalExposureChemicals','WaterQuality'
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Define the model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Export the trained model
model_filename = "ckd_prediction_model.pkl"
model_path = os.path.join(os.path.dirname(__file__), model_filename)

# Save the model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved successfully to: {model_path}")

# Save model metadata
metadata = {
    'model_type': 'LogisticRegression',
    'accuracy': accuracy_score(y_test, y_pred),
    'feature_columns': list(X.columns),
    'categorical_columns': categorical_cols,
    'numeric_columns': numeric_cols,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

metadata_filename = "model_metadata.pkl"
metadata_path = os.path.join(os.path.dirname(__file__), metadata_filename)
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Model metadata saved to: {metadata_path}")

# Optional: Create a simple loading function demonstration
def load_model():
    """Load the trained model and metadata"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return model, metadata

print(f"\nTo load the model in another script, use:")
print(f"import pickle")
print(f"with open('{model_filename}', 'rb') as f:")
print(f"    model = pickle.load(f)")
print(f"with open('{metadata_filename}', 'rb') as f:")
print(f"    metadata = pickle.load(f)")
