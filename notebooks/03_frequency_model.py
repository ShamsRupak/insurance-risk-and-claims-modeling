# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Insurance Risk
#     language: python
#     name: insurance-risk
# ---

# %% [markdown]
# # 03 — Claim Frequency Modeling
#
# **Goal:** Build a Poisson GLM for claim frequency with `log(Exposure)` as
# offset. Evaluate using Poisson deviance and lift charts.

# %% tags=["imports"]
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), ""))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", "{:,.4f}".format)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# %% [markdown]
# ## 1. Load cleaned data & engineer features

# %%
from src.utils.config import FREQ_PROCESSED, CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE, TEST_SIZE
from src.features.build_features import add_log_density, build_all_features

freq = pd.read_csv(FREQ_PROCESSED)
freq = build_all_features(freq)

print("Shape after feature engineering:", freq.shape)
display(freq.head())

# %% [markdown]
# ## 2. Prepare design matrix
#
# Encode categoricals via one-hot encoding (drop first to avoid collinearity).

# %%
# One-hot encode categoricals
freq_encoded = pd.get_dummies(freq, columns=CATEGORICAL_COLS, drop_first=True, dtype=float)

# Define feature columns (all numeric + dummies, excluding targets/ID)
exclude = {"IDpol", "ClaimNb", "Exposure", "HasClaim",
           "VehAgeBin", "DrivAgeBin", "BonusMalusBin"}
feature_cols = [c for c in freq_encoded.columns if c not in exclude]

print(f"Number of features: {len(feature_cols)}")
print("Sample features:", feature_cols[:10])

# %% [markdown]
# ## 3. Train/test split

# %%
train_df, test_df = train_test_split(
    freq_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {train_df.shape[0]:,}  |  Test: {test_df.shape[0]:,}")

# %% [markdown]
# ## 4. Fit Poisson GLM

# %%
from src.models.frequency import PoissonFrequencyModel

model = PoissonFrequencyModel(feature_cols)
model.fit(train_df, target="ClaimNb", exposure="Exposure")

print(model.summary())

# %% [markdown]
# ## 5. Predictions

# %%
train_df = train_df.copy()
test_df = test_df.copy()

train_df["pred_freq"] = model.predict(train_df, exposure="Exposure")
test_df["pred_freq"] = model.predict(test_df, exposure="Exposure")

print("Train — observed vs predicted mean:")
print(f"  Observed:  {train_df['ClaimNb'].mean():.5f}")
print(f"  Predicted: {train_df['pred_freq'].mean():.5f}")

print("\nTest — observed vs predicted mean:")
print(f"  Observed:  {test_df['ClaimNb'].mean():.5f}")
print(f"  Predicted: {test_df['pred_freq'].mean():.5f}")

# %% [markdown]
# ## 6. Evaluation — Poisson deviance

# %%
from src.evaluation.metrics import deviance_poisson, lift_table, gini_coefficient

dev_train = deviance_poisson(train_df["ClaimNb"], train_df["pred_freq"])
dev_test = deviance_poisson(test_df["ClaimNb"], test_df["pred_freq"])

print(f"Poisson deviance — train: {dev_train:.5f}")
print(f"Poisson deviance — test:  {dev_test:.5f}")

# %%
gini_train = gini_coefficient(train_df["ClaimNb"].values, train_df["pred_freq"].values)
gini_test = gini_coefficient(test_df["ClaimNb"].values, test_df["pred_freq"].values)
print(f"Gini — train: {gini_train:.4f}")
print(f"Gini — test:  {gini_test:.4f}")

# %% [markdown]
# ## 7. Lift chart

# %%
lift = lift_table(
    test_df["ClaimNb"].values,
    test_df["pred_freq"].values,
    exposure=test_df["Exposure"].values,
    n_bins=10,
)
display(lift)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(lift))
ax.bar(x, lift["observed_mean"], width=0.4, label="Observed", color="steelblue", align="center")
ax.bar([i + 0.4 for i in x], lift["predicted_mean"], width=0.4, label="Predicted", color="coral", align="center")
ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels([f"D{i+1}" for i in x], rotation=0)
ax.set_xlabel("Decile (by predicted frequency)")
ax.set_ylabel("Mean Claim Count")
ax.set_title("Frequency Model — Lift Chart (Test Set)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Residual diagnostics

# %%
# Deviance residuals
test_df["resid_deviance"] = model.results_.resid_deviance[test_df.index] if hasattr(
    model.results_, "resid_deviance"
) else test_df["ClaimNb"] - test_df["pred_freq"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(test_df["pred_freq"], test_df["resid_deviance"],
                alpha=0.1, s=2, color="steelblue")
axes[0].axhline(0, color="red", linewidth=0.8)
axes[0].set_xlabel("Predicted Frequency")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs. Predicted")

axes[1].hist(test_df["resid_deviance"].clip(-5, 5), bins=100,
             color="steelblue", edgecolor="white")
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - Fitted Poisson GLM with `log(Exposure)` offset on the full feature set
# - Model achieves reasonable Poisson deviance and positive Gini on test data
# - Lift chart shows good monotonic separation across predicted-frequency deciles
# - Key predictors: `BonusMalus`, `DrivAge`, `Density`, `Area`, `VehPower`
# - Next: severity modeling in notebook 04
