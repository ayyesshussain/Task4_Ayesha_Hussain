import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/clean_telco.csv")

# Target Encoding
print(df["Churn"].unique())

# only if already numeric
df["Churn"] = df["Churn"].astype(int)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Numerical Columns
numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Categorical Columns
categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Pipeline
pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Save Pipeline
joblib.dump(
    pipeline,
    "models/churn_pipeline.pkl"
)

print("Pipeline Saved Successfully")