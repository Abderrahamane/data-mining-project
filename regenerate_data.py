import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("Loading raw data...")
df = pd.read_csv('data/raw/vgchartz-2024.csv')
print(f"Original dataset shape: {df.shape}")

# Handle missing values
print("\n1. Handling missing values...")
rows_before = len(df)
df = df.dropna(subset=['na_sales', 'pal_sales'], how='all')
rows_after = len(df)
print(f"Dropped {rows_before - rows_after} rows with missing both NA and PAL sales")

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
    print(f"Recalculated total_sales for {missing_total_mask.sum()} rows")

# Drop remaining missing total_sales
df = df.dropna(subset=['total_sales'])
print(f"Dataset shape after handling sales: {df.shape}")

# Handle critic_score
print("\n2. Handling critic_score...")
df['has_critic_score'] = (~df['critic_score'].isnull()).astype(int)
critic_score_by_genre = df.groupby('genre')['critic_score'].median()
df['critic_score'] = df.apply(
    lambda row: critic_score_by_genre[row['genre']] if pd.isnull(row['critic_score']) else row['critic_score'],
    axis=1
)
df['critic_score'].fillna(df['critic_score'].median(), inplace=True)
print(f"Filled missing critic_score")

# Handle developer/publisher
print("\n3. Handling developer and publisher...")
df['developer'].fillna('Unknown', inplace=True)
df['publisher'].fillna('Unknown', inplace=True)

# Define target
print("\n4. Defining target variable...")
df['target'] = (df['total_sales'] >= 1.0).astype(int)
print("Target distribution:")
print(df['target'].value_counts())

# Feature engineering
print("\n5. Feature engineering...")
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
median_year = df['release_year'].median()
df['release_year'].fillna(median_year, inplace=True)
current_year = 2025
df['game_age'] = current_year - df['release_year']

# Create sales ratio features with proper NaN handling
df['na_sales_ratio'] = df['na_sales'] / (df['total_sales'] + 1e-6)
df['jp_sales_ratio'] = df['jp_sales'] / (df['total_sales'] + 1e-6)
df['pal_sales_ratio'] = df['pal_sales'] / (df['total_sales'] + 1e-6)
df['other_sales_ratio'] = df['other_sales'] / (df['total_sales'] + 1e-6)

# Fill NaN values in ratios with 0 (for games with 0 total_sales)
df['na_sales_ratio'].fillna(0, inplace=True)
df['jp_sales_ratio'].fillna(0, inplace=True)
df['pal_sales_ratio'].fillna(0, inplace=True)
df['other_sales_ratio'].fillna(0, inplace=True)

print("Created new features")

# Drop irrelevant columns
print("\n6. Dropping irrelevant columns...")
drop_cols = ['img', 'last_update', 'release_date', 'title']
existing_drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=existing_drop_cols)

# Handle duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")

# Simplify categorical variables
print("\n7. Simplifying categorical variables...")
def simplify_categorical(df, column, top_n=10):
    top_categories = df[column].value_counts().nlargest(top_n).index
    df[column] = df[column].apply(lambda x: x if x in top_categories else 'Other')
    return df

df = simplify_categorical(df, 'console', top_n=10)
df = simplify_categorical(df, 'publisher', top_n=10)
df = simplify_categorical(df, 'developer', top_n=10)
print("Simplified console, publisher, and developer")

# One-hot encoding
print("\n8. One-hot encoding...")
categorical_cols = ['console', 'genre', 'publisher', 'developer']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print(f"Dataset shape after encoding: {df_encoded.shape}")

# Prepare features and target
print("\n9. Preparing features and target...")
exclude_from_features = ['target', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
feature_cols = [col for col in df_encoded.columns if col not in exclude_from_features]
X = df_encoded[feature_cols]
y = df_encoded['target']

# Check for any remaining NaN values
print(f"\nChecking for NaN values in features...")
nan_counts = X.isnull().sum()
if nan_counts.sum() > 0:
    print("WARNING: Found NaN values:")
    print(nan_counts[nan_counts > 0])
    print("\nFilling remaining NaN values with 0...")
    X = X.fillna(0)
else:
    print("No NaN values found in features!")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
print("\n10. Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
print("\n11. Feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Calculate class weights
print("\n12. Calculating class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class 0 (Miss): {class_weight_dict[0]:.4f}")
print(f"Class 1 (Hit): {class_weight_dict[1]:.4f}")

# Save processed data
print("\n13. Saving processed data...")
output_dir = 'data/processed/'

X_train.to_csv(output_dir + 'X_train.csv', index=False)
X_test.to_csv(output_dir + 'X_test.csv', index=False)
y_train.to_csv(output_dir + 'y_train.csv', index=False)
y_test.to_csv(output_dir + 'y_test.csv', index=False)

X_train_scaled.to_csv(output_dir + 'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(output_dir + 'X_test_scaled.csv', index=False)

pd.DataFrame([class_weight_dict]).to_csv(output_dir + 'class_weights.csv', index=False)
pd.DataFrame({'feature': feature_cols}).to_csv(output_dir + 'feature_names.csv', index=False)

print("\n✅ All processed data saved successfully!")
print(f"Total features: {len(feature_cols)}")

