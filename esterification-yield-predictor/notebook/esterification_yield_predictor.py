# =============================================================================
# Esterification Yield Predictor
# CH3COOH + C2H5OH -> CH3COOC2H5 + H2O
#
# Built this to estimate esterification yield using some basic ML.
# Idea was mainly to mimic how a process engineer might screen reaction
# conditions before actually running wet-lab trials.
#
# NOTE:
# The dataset here is synthetic, but the trends are based loosely on
# Fischer esterification chemistry.
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")



# folders / paths


# kinda overdoing the folder setup here but it keeps things organized
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_folder   = os.path.join(ROOT_DIR, "data")
image_folder  = os.path.join(ROOT_DIR, "images")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)


print("\nStarting esterification ML pipeline...")
print("-" * 60)


# -------------------------------------------------------------------
# 1) generate synthetic data
# -------------------------------------------------------------------

np.random.seed(42)

sample_count = 200

# temperature range picked from some typical esterification conditions
temps = np.random.uniform(50, 90, sample_count)

# sulfuric acid catalyst loading
catalyst_g = np.random.uniform(0.5, 5.0, sample_count)

reaction_minutes = np.random.uniform(30, 180, sample_count)

# ethanol excess ratio
etoh_ratio = np.random.uniform(1.0, 3.0, sample_count)


# trying to imitate realistic chemistry trends here
# not physically rigorous obviously lol
yield_pct = (
    20
    + (0.60 * temps)
    - (0.005 * temps**2)
    + (3.5 * catalyst_g)
    + (0.12 * reaction_minutes)
    + (5.0 * etoh_ratio)
)

# experimental noise
noise = np.random.normal(0, 2.5, sample_count)

yield_pct = yield_pct + noise

# just to avoid weird impossible values
yield_pct = np.clip(yield_pct, 10, 99)


# assemble dataframe
reaction_df = pd.DataFrame({
    "temperature": temps,
    "catalyst_amount": catalyst_g,
    "reaction_time": reaction_minutes,
    "ethanol_ratio": etoh_ratio,
    "yield_percentage": yield_pct
})


# rounding because too many decimals looks ugly in csv output honestly
reaction_df = reaction_df.round(2)


csv_file = os.path.join(data_folder, "esterification_data.csv")

reaction_df.to_csv(csv_file, index=False)

print(f"dataset exported -> {csv_file}")
print(f"total experiments generated: {len(reaction_df)}")


# -------------------------------------------------------------------
# 2) reload dataset
# -------------------------------------------------------------------

# yes technically unnecessary because dataframe already exists
# but I wanted to simulate a real workflow pipeline
df = pd.read_csv(csv_file)

print("\nQuick peek at dataset:")
print(df.head())


# -------------------------------------------------------------------
# 3) quick sanity checks
# -------------------------------------------------------------------

print("\nchecking for missing values...")

null_counts = df.isnull().sum()

for c in df.columns:
    print(c, "->", null_counts[c])

dupes = df.duplicated().sum()

if dupes > 0:
    print("duplicates found:", dupes)

    # probably won't happen but leaving this anyway
    df = df.drop_duplicates()

else:
    print("no duplicate rows found")


# some basic validation
assert df["yield_percentage"].between(0, 100).all()

print("validation checks passed")


# -------------------------------------------------------------------
# 4) correlation inspection
# -------------------------------------------------------------------

corr_matrix = df.corr(numeric_only=True)

yield_corr = corr_matrix["yield_percentage"]

print("\nFeature correlations against yield:\n")

for thing in yield_corr.index:

    if thing == "yield_percentage":
        continue

    corr_val = yield_corr[thing]

    print(f"{thing:20s}: {corr_val:.3f}")


# -------------------------------------------------------------------
# 5) train model
# -------------------------------------------------------------------

feature_cols = [
    "temperature",
    "catalyst_amount",
    "reaction_time",
    "ethanol_ratio"
]

X = df[feature_cols]

y = df["yield_percentage"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

print("\nmodel training done")


# coefficients
print("\ncoefficients:\n")

for feat_name, coeff in zip(feature_cols, lin_reg.coef_):

    print(feat_name, "=", round(coeff, 4))


# -------------------------------------------------------------------
# 6) predictions + evaluation
# -------------------------------------------------------------------

preds = lin_reg.predict(X_test)

rmse_val = np.sqrt(mean_squared_error(y_test, preds))
r2_val = r2_score(y_test, preds)

print("\nmodel performance")
print("----------------------------")
print("RMSE:", round(rmse_val, 3))
print("R2  :", round(r2_val, 4))


# honestly linear regression performs better here than expected
# probably because synthetic data was generated from mostly linear trends



# 7) predict new reaction


test_reaction = pd.DataFrame([{
    "temperature": 70,
    "catalyst_amount": 2.5,
    "reaction_time": 120,
    "ethanol_ratio": 2.0
}])


predicted = lin_reg.predict(test_reaction)[0]

print("\nPredicted Yield =", round(predicted, 2), "%")


if predicted > 80:
    print("looks like a pretty strong process window")

elif predicted > 60:
    print("decent yield, maybe optimize catalyst loading")

else:
    print("yield is kinda low, reaction conditions probably need work")


print("\nPipeline complete.")
