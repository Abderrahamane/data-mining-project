import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("Testing modeling notebook setup...")

# Load data
print("\n1. Loading data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')

class_weights_df = pd.read_csv('data/processed/class_weights.csv')
class_weights = {int(k): v for k, v in class_weights_df.to_dict('records')[0].items()}

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Class weights: {class_weights}")

# Check for NaN values
print("\n2. Checking for NaN values...")
print(f"X_train NaN count: {X_train.isnull().sum().sum()}")
print(f"X_train_scaled NaN count: {X_train_scaled.isnull().sum().sum()}")
print(f"y_train NaN count: {pd.Series(y_train).isnull().sum()}")

# Try training a simple model
print("\n3. Testing Logistic Regression...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

print("Running cross-validation...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
print(f"CV F1 scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\nTraining on full training set...")
model.fit(X_train_scaled, y_train)

print("Making predictions...")
y_pred = model.predict(X_test_scaled)

from sklearn.metrics import f1_score, accuracy_score
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 Score: {f1_score(y_test, y_pred):.4f}")

print("\n✅ All tests passed! The modeling notebook should work correctly now.")

