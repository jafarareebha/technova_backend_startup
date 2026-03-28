import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv('startups_scored.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. CREATE TARGET SCORE ────────────────────────────────────────────────────
# We don't have a raw score column — we build it from the outcome label
# IPO = highest success, acquired = good, active = neutral, failed = low
# Then we blend in the dummy scores to add nuance

outcome_base = {
    'ipo':      85,
    'acquired': 70,
    'active':   55,
    'failed':   25
}
df['base_score'] = df['outcome'].map(outcome_base)

# Blend with dummy scores — weighted average nudges the base score up or down
# This is your success score formula
dummy_weights = {
    'market_demand':           0.25,
    'scalability':             0.20,
    'revenue_model_strength':  0.20,
    'barriers_to_entry':       0.15,
    'willingness_to_pay':      0.10,
    'idea_novelty':            0.10,
}

# Normalize dummy scores to 0-100 scale (they're currently 1-10)
dummy_score = sum(
    (df[col] / 10 * 100) * weight
    for col, weight in dummy_weights.items()
)

# Final score = 60% from outcome, 40% from dummy scores
df['success_score'] = (df['base_score'] * 0.6 + dummy_score * 0.4).round(1)
df['success_score'] = df['success_score'].clip(0, 100)

print(f"\nSuccess score distribution:")
print(df['success_score'].describe().round(2))

# ── 3. ENCODE CATEGORICAL COLUMNS ────────────────────────────────────────────
cat_cols = ['industry_sector', 'market_type', 'country_region']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# ── 4. DEFINE FEATURES ────────────────────────────────────────────────────────
feature_cols = [
    # Real data features
    'industry_sector_enc',
    'founding_team_size',
    'initial_funding_usd',
    'num_funding_rounds',
    'market_type_enc',
    'country_region_enc',
    'num_competitors_proxy',
    'year_founded',
    # Dummy scored features
    'market_demand',
    'pain_point_severity',
    'idea_novelty',
    'scalability',
    'barriers_to_entry',
    'revenue_model_strength',
    'acquisition_difficulty',
    'willingness_to_pay',
]

X = df[feature_cols]
y = df['success_score']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target range: {y.min()} - {y.max()}")

# ── 5. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 6. TRAIN MODELS ───────────────────────────────────────────────────────────
print("\n--- Training Random Forest ---")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2  = r2_score(y_test, rf_preds)
print(f"Random Forest  →  MAE: {rf_mae:.2f}  |  R²: {rf_r2:.3f}")

print("\n--- Training Gradient Boosting ---")
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_preds)
gb_r2  = r2_score(y_test, gb_preds)
print(f"Gradient Boost →  MAE: {gb_mae:.2f}  |  R²: {gb_r2:.3f}")

# ── 7. PICK BEST MODEL ────────────────────────────────────────────────────────
best_model = rf if rf_mae <= gb_mae else gb
best_name  = "Random Forest" if rf_mae <= gb_mae else "Gradient Boosting"
print(f"\nBest model: {best_name}")

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature':   feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

# ── 9. SCORE → CATEGORY FUNCTION ─────────────────────────────────────────────
def score_to_category(score):
    if score >= 75:   return "Strong"
    elif score >= 55: return "Moderate"
    elif score >= 35: return "Weak"
    else:             return "Poor"

# ── 10. PREDICTION FUNCTION ───────────────────────────────────────────────────
def predict_startup(input_dict):
    """
    Pass a dictionary of startup inputs.
    Returns: score (float), category (str)

    Example:
    predict_startup({
        'industry_sector':      'EdTech',
        'founding_team_size':   3,
        'initial_funding_usd':  50000,
        'num_funding_rounds':   1,
        'market_type':          'B2C',
        'country_region':       'USA',
        'num_competitors_proxy':50,
        'year_founded':         2024,
        'market_demand':        7,
        'pain_point_severity':  6,
        'idea_novelty':         8,
        'scalability':          7,
        'barriers_to_entry':    5,
        'revenue_model_strength':6,
        'acquisition_difficulty':6,
        'willingness_to_pay':   7,
    })
    """
    row = {}
    for col in cat_cols:
        val = input_dict.get(col, 'Unknown')
        le  = encoders[col]
        if val in le.classes_:
            row[col + '_enc'] = le.transform([val])[0]
        else:
            row[col + '_enc'] = 0  # fallback for unseen categories

    for col in feature_cols:
        if col not in row:
            row[col] = input_dict.get(col, 0)

    X_input = pd.DataFrame([row])[feature_cols]
    score   = float(best_model.predict(X_input)[0])
    score   = round(min(max(score, 0), 100), 1)
    category = score_to_category(score)
    return score, category

# ── 11. TEST THE PREDICTION FUNCTION ─────────────────────────────────────────
print("\n--- Sample Prediction ---")
score, category = predict_startup({
    'industry_sector':        'EdTech',
    'founding_team_size':     3,
    'initial_funding_usd':    50000,
    'num_funding_rounds':     1,
    'market_type':            'B2C',
    'country_region':         'USA',
    'num_competitors_proxy':  50,
    'year_founded':           2024,
    'market_demand':          7,
    'pain_point_severity':    6,
    'idea_novelty':           8,
    'scalability':            7,
    'barriers_to_entry':      5,
    'revenue_model_strength': 6,
    'acquisition_difficulty': 6,
    'willingness_to_pay':     7,
})
print(f"Score: {score}/100  |  Category: {category}")

# ── 12. SAVE MODEL + ENCODERS ─────────────────────────────────────────────────
with open('startup_model.pkl', 'wb') as f:
    pickle.dump({
        'model':        best_model,
        'encoders':     encoders,
        'feature_cols': feature_cols,
        'model_name':   best_name
    }, f)
print("\nModel saved to startup_model.pkl")
print("Done.")