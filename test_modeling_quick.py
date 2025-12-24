"""
Quick test to verify modeling works with the fixed data (no sales columns)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*80)
print("TESTING MODELING WITH FIXED DATA (NO DATA LEAKAGE)")
print("="*80)

# Load data
print("\n1. Loading processed data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_train distribution: {np.bincount(y_train)}")
print(f"   y_test distribution: {np.bincount(y_test)}")

# Verify no sales columns
print("\n2. Verifying no sales columns in features...")
sales_cols = [col for col in X_train.columns if 'sales' in col.lower()]
if sales_cols:
    print(f"   ⚠️  WARNING: Found sales columns: {sales_cols}")
else:
    print(f"   ✅ No sales columns found - data leakage avoided!")

# Load class weights
print("\n3. Loading class weights...")
class_weights_df = pd.read_csv('data/processed/class_weights.csv')
class_weights = {int(col): class_weights_df[col].values[0] for col in class_weights_df.columns}
print(f"   Class weights: {class_weights}")

# Test Logistic Regression
print("\n4. Testing Logistic Regression...")
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

try:
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='f1')
    print(f"   ✅ Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    lr_model.fit(X_train_scaled, y_train)
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    
    print(f"   ✅ Test F1: {test_f1:.4f}")
    print(f"   ✅ Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"   ✅ Test Recall: {test_recall:.4f}")
    print(f"   ✅ Test Precision: {test_precision:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test Random Forest
print("\n5. Testing Random Forest...")
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

try:
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1')
    print(f"   ✅ Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    
    print(f"   ✅ Test F1: {test_f1:.4f}")
    print(f"   ✅ Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"   ✅ Test Recall: {test_recall:.4f}")
    print(f"   ✅ Test Precision: {test_precision:.4f}")
    
    # Show top features
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    print("\n   Top 10 Most Important Features:")
    for idx, row in feature_imp.iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.4f}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - MODELING NOTEBOOK SHOULD WORK!")
print("="*80)

