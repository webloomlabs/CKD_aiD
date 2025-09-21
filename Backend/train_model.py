import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("/Chronic_Kidney_Dsease_data.csv")

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
