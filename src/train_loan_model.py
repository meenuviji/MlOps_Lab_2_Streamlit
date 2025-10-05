"""
Train Loan Approval Prediction Model
This script should be placed in FastAPI_Labs/src/ directory
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic loan application data
n_samples = 5000

print("Generating synthetic loan application data...")

# Generate features
age = np.random.randint(18, 75, n_samples)
income = np.random.lognormal(11, 0.7, n_samples).clip(20000, 500000)  # Log-normal income distribution
loan_amount = np.random.uniform(5000, 500000, n_samples)
loan_term = np.random.choice([60, 120, 180, 240, 360], n_samples)  # 5, 10, 15, 20, 30 years
credit_score = np.random.normal(700, 100, n_samples).clip(300, 850)
employment_length = np.random.exponential(5, n_samples).clip(0, 40)
home_ownership = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])  # Rent, Own, Mortgage
purpose = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
monthly_debt = np.random.uniform(0, 5000, n_samples)
dti_ratio = (monthly_debt * 12) / income
has_cosigner = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
previous_defaults = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

# Create approval labels based on realistic criteria
# Initialize approval probability
approval_prob = np.zeros(n_samples)

# Credit score impact (most important factor)
approval_prob += np.where(credit_score >= 750, 0.4, 
                         np.where(credit_score >= 700, 0.3,
                                 np.where(credit_score >= 650, 0.2,
                                         np.where(credit_score >= 600, 0.1, 0.0))))

# Income to loan ratio impact
income_loan_ratio = income / loan_amount
approval_prob += np.where(income_loan_ratio >= 0.5, 0.2,
                         np.where(income_loan_ratio >= 0.3, 0.15,
                                 np.where(income_loan_ratio >= 0.2, 0.1, 0.0)))

# DTI ratio impact
approval_prob += np.where(dti_ratio <= 0.3, 0.15,
                         np.where(dti_ratio <= 0.4, 0.1,
                                 np.where(dti_ratio <= 0.5, 0.05, 0.0)))

# Employment length impact
approval_prob += np.where(employment_length >= 5, 0.1,
                         np.where(employment_length >= 2, 0.05, 0.0))

# Home ownership impact
approval_prob += np.where(home_ownership == 1, 0.1,  # Own
                         np.where(home_ownership == 2, 0.05, 0.0))  # Mortgage

# Cosigner impact
approval_prob += np.where(has_cosigner == 1, 0.1, 0.0)

# Previous defaults impact (negative)
approval_prob -= np.where(previous_defaults == 1, 0.3, 0.0)

# Add some randomness
approval_prob += np.random.normal(0, 0.1, n_samples)

# Convert to binary approval (1 = approved, 0 = rejected)
approval = (approval_prob >= 0.5).astype(int)

# Ensure realistic approval rate (around 65-70%)
current_approval_rate = approval.mean()
print(f"Initial approval rate: {current_approval_rate:.2%}")

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'credit_score': credit_score,
    'employment_length': employment_length,
    'home_ownership': home_ownership,
    'purpose': purpose,
    'dti_ratio': dti_ratio,
    'has_cosigner': has_cosigner,
    'previous_defaults': previous_defaults,
    'approval': approval
})

# Save the dataset
df.to_csv('loan_applications.csv', index=False)
print(f"Dataset saved to loan_applications.csv with {n_samples} samples")

# Prepare features and target
X = df.drop('approval', axis=1)
y = df['approval']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to loan_model.pkl")

# Save the scaler
with open('loan_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to loan_scaler.pkl")

# Save model metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'features': list(X.columns),
    'accuracy': float(accuracy),
    'approval_rate': float(y_train.mean()),
    'training_samples': len(X_train),
    'feature_importance': feature_importance.to_dict('records')
}

with open('loan_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Model metadata saved to loan_model_metadata.json")

# Create a sample test file
sample_application = {
    "input_test": {
        "age": 35,
        "income": 75000,
        "loan_amount": 200000,
        "loan_term": 240,
        "credit_score": 720,
        "employment_length": 5,
        "home_ownership": 1,  # Own
        "purpose": 0,  # Debt consolidation
        "dti_ratio": 0.35,
        "has_cosigner": 0,
        "previous_defaults": 0
    }
}

with open('test_loan_application.json', 'w') as f:
    json.dump(sample_application, f, indent=2)
print("\nSample test file created: test_loan_application.json")

# Test the model with sample data
sample_df = pd.DataFrame([sample_application["input_test"]])
sample_scaled = scaler.transform(sample_df)
sample_pred = model.predict(sample_scaled)
sample_proba = model.predict_proba(sample_scaled)

print(f"\nSample prediction: {'Approved' if sample_pred[0] == 1 else 'Rejected'}")
print(f"Approval probability: {sample_proba[0][1]:.2%}")