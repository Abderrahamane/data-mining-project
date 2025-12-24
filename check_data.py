import pandas as pd
import numpy as np

print("Checking for NaN/Inf values in processed data...")
print("="*60)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')

print(f"\nX_train shape: {X_train.shape}")
print(f"X_train NaN count: {X_train.isnull().sum().sum()}")
print(f"X_train Inf count: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")

print(f"\nX_test shape: {X_test.shape}")
print(f"X_test NaN count: {X_test.isnull().sum().sum()}")
print(f"X_test Inf count: {np.isinf(X_test.select_dtypes(include=[np.number])).sum().sum()}")

print(f"\nX_train_scaled shape: {X_train_scaled.shape}")
print(f"X_train_scaled NaN count: {X_train_scaled.isnull().sum().sum()}")
print(f"X_train_scaled Inf count: {np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum()}")

print(f"\nX_test_scaled shape: {X_test_scaled.shape}")
print(f"X_test_scaled NaN count: {X_test_scaled.isnull().sum().sum()}")
print(f"X_test_scaled Inf count: {np.isinf(X_test_scaled.select_dtypes(include=[np.number])).sum().sum()}")

print("\n" + "="*60)
if (X_train.isnull().sum().sum() == 0 and X_test.isnull().sum().sum() == 0 and
    X_train_scaled.isnull().sum().sum() == 0 and X_test_scaled.isnull().sum().sum() == 0):
    print("✅ All data is clean - no NaN or Inf values!")
else:
    print("⚠️  Warning: Found NaN or Inf values in data!")

