"""
Regenerate processed data WITHOUT sales columns to avoid data leakage
"""
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA PREPARATION - REMOVING SALES COLUMNS TO AVOID DATA LEAKAGE")
print("="*80)

# Load data
dataset_path = kagglehub.dataset_download("asaniczka/video-game-sales-2024")
csv_file = [f for f in os.listdir(dataset_path) if f.endswith('.csv')][0]
file_path = os.path.join(dataset_path, csv_file)

df = pd.read_csv(file_path)
print(f"\nOriginal dataset shape: {df.shape}")

# Handle missing values
print("\n1. Handling missing values...")
rows_before = len(df)
df = df.dropna(subset=['na_sales', 'pal_sales'], how='all')
rows_after = len(df)
print(f"   Dropped {rows_before - rows_after} rows with missing both NA and PAL sales")

# Calculate total_sales for rows where it's missing
missing_total_mask = df['total_sales'].isnull()
if missing_total_mask.sum() > 0:
    df['na_sales'].fillna(0, inplace=True)
    df['jp_sales'].fillna(0, inplace=True)
    df['pal_sales'].fillna(0, inplace=True)
    df['other_sales'].fillna(0, inplace=True)

    df.loc[missing_total_mask, 'total_sales'] = (
        df.loc[missing_total_mask, 'na_sales'] +
        df.loc[missing_total_mask, 'jp_sales'] +
        df.loc[missing_total_mask, 'pal_sales'] +
        df.loc[missing_total_mask, 'other_sales']
    )
    print(f"   Recalculated total_sales for {missing_total_mask.sum()} rows")

if df['total_sales'].isnull().sum() > 0:
    df = df.dropna(subset=['total_sales'])
    print(f"   Dropped remaining rows with missing total_sales")

# Handle critic_score
df['has_critic_score'] = df['critic_score'].notna().astype(int)
genre_median_scores = df.groupby('genre')['critic_score'].median()
df['critic_score'] = df.apply(
    lambda row: genre_median_scores[row['genre']] if pd.isna(row['critic_score']) else row['critic_score'],
    axis=1
)
overall_median = df['critic_score'].median()
df['critic_score'].fillna(overall_median, inplace=True)
print(f"   Filled missing critic_scores")

# Handle developer and publisher
df['developer'].fillna('Unknown', inplace=True)
df['publisher'].fillna('Unknown', inplace=True)
print(f"   Filled missing developers/publishers with 'Unknown'")

# Define target
print("\n2. Defining target variable...")
df['target'] = (df['total_sales'] >= 1.0).astype(int)
print(f"   Target distribution:")
print(f"   Miss (0): {(df['target']==0).sum()} ({(df['target']==0).sum()/len(df)*100:.1f}%)")
print(f"   Hit (1): {(df['target']==1).sum()} ({(df['target']==1).sum()/len(df)*100:.1f}%)")

# Feature engineering
print("\n3. Feature engineering...")
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
median_year = df['release_year'].median()
df['release_year'].fillna(median_year, inplace=True)

current_year = 2025
df['game_age'] = current_year - df['release_year']
print(f"   Created features: release_year, game_age, has_critic_score")

# Drop irrelevant columns
drop_cols = ['img', 'last_update', 'release_date', 'title']
existing_drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=existing_drop_cols)
print(f"   Dropped columns: {existing_drop_cols}")

# Remove duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   Removed {duplicates} duplicate rows")

# Simplify categorical variables
print("\n4. Simplifying categorical variables...")
def simplify_categorical(df, column, top_n=10):
    top_categories = df[column].value_counts().nlargest(top_n).index
    df[column] = df[column].apply(lambda x: x if x in top_categories else 'Other')
    return df

df = simplify_categorical(df, 'console', top_n=10)
df = simplify_categorical(df, 'publisher', top_n=10)
df = simplify_categorical(df, 'developer', top_n=10)
print(f"   Simplified console, publisher, developer to Top 10 + 'Other'")

# One-hot encoding
print("\n5. Applying one-hot encoding...")
categorical_cols = ['console', 'genre', 'publisher', 'developer']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print(f"   Encoded columns: {categorical_cols}")
print(f"   Dataset shape after encoding: {df_encoded.shape}")

# Prepare features and target (EXCLUDE ALL SALES COLUMNS)
print("\n6. Preparing features and target...")
print("   ⚠️  EXCLUDING ALL SALES COLUMNS TO AVOID DATA LEAKAGE ⚠️")
exclude_from_features = [
    'target',
    'total_sales',
    'na_sales',
    'jp_sales',
    'pal_sales',
    'other_sales'
]
print(f"   Excluded columns: {exclude_from_features}")

feature_cols = [col for col in df_encoded.columns if col not in exclude_from_features]
X = df_encoded[feature_cols]
y = df_encoded['target']

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Number of features: {len(feature_cols)}")

# Train-test split
print("\n7. Train-test split (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Feature scaling
print("\n8. Feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
print(f"   Applied StandardScaler")

# Calculate class weights
print("\n9. Calculating class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"   Class 0 (Miss): {class_weight_dict[0]:.4f}")
print(f"   Class 1 (Hit): {class_weight_dict[1]:.4f}")

# Save processed data
print("\n10. Saving processed data...")
output_dir = './data/processed/'
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(output_dir + 'X_train.csv', index=False)
X_test.to_csv(output_dir + 'X_test.csv', index=False)
y_train.to_csv(output_dir + 'y_train.csv', index=False)
y_test.to_csv(output_dir + 'y_test.csv', index=False)

X_train_scaled.to_csv(output_dir + 'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(output_dir + 'X_test_scaled.csv', index=False)

pd.DataFrame([class_weight_dict]).to_csv(output_dir + 'class_weights.csv', index=False)
pd.DataFrame({'feature': feature_cols}).to_csv(output_dir + 'feature_names.csv', index=False)

print(f"   ✅ All files saved to: {output_dir}")
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE - NO DATA LEAKAGE!")
print("="*80)

