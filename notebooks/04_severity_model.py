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
# # 04 — Claim Severity Modeling
#
# **Goal:** Build a Gamma GLM with log-link on strictly positive claim amounts.
# Evaluate using Gamma deviance and residual diagnostics.

# %% tags=["imports"]
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), ""))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", "{:,.4f}".format)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# %% [markdown]
# ## 1. Load & prepare severity data
#
# We need to merge severity (claim-level) with frequency (policy-level) to
# access the feature columns.

# %%
from src.utils.config import (
    FREQ_PROCESSED, SEV_PROCESSED, CATEGORICAL_COLS, RANDOM_STATE, TEST_SIZE,
)
from src.features.build_features import add_log_density

freq = pd.read_csv(FREQ_PROCESSED)
sev = pd.read_csv(SEV_PROCESSED)

# Aggregate severity to policy level
sev_pol = sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()
sev_pol.columns = ["IDpol", "ClaimAmount"]

# Merge with policy features (keep only policies with claims)
sev_merged = freq.merge(sev_pol, on="IDpol", how="inner")
sev_merged = add_log_density(sev_merged)

print("Policies with claims:", sev_merged.shape[0])
print("Columns:", list(sev_merged.columns))
display(sev_merged.head())

# %% [markdown]
# ## 2. Prepare design matrix

# %%
sev_encoded = pd.get_dummies(sev_merged, columns=CATEGORICAL_COLS, drop_first=True, dtype=float)

exclude = {"IDpol", "ClaimNb", "Exposure", "ClaimAmount"}
feature_cols = [c for c in sev_encoded.columns if c not in exclude]

print(f"Number of features: {len(feature_cols)}")

# %% [markdown]
# ## 3. Train/test split

# %%
train_df, test_df = train_test_split(
    sev_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {train_df.shape[0]:,}  |  Test: {test_df.shape[0]:,}")

# %% [markdown]
# ## 4. Fit Gamma GLM

# %%
from src.models.severity import GammaSeverityModel

model = GammaSeverityModel(feature_cols)
model.fit(train_df, target="ClaimAmount")

print(model.summary())

# %% [markdown]
# ## 5. Predictions

# %%
train_df = train_df.copy()
test_df = test_df.copy()

train_df["pred_sev"] = model.predict(train_df)
test_df["pred_sev"] = model.predict(test_df)

print("Train — observed vs predicted mean severity:")
print(f"  Observed:  {train_df['ClaimAmount'].mean():,.2f}")
print(f"  Predicted: {train_df['pred_sev'].mean():,.2f}")

print("\nTest — observed vs predicted mean severity:")
print(f"  Observed:  {test_df['ClaimAmount'].mean():,.2f}")
print(f"  Predicted: {test_df['pred_sev'].mean():,.2f}")

# %% [markdown]
# ## 6. Evaluation — Gamma deviance

# %%
from src.evaluation.metrics import deviance_gamma, rmse, mae, lift_table

dev_train = deviance_gamma(train_df["ClaimAmount"].values, train_df["pred_sev"].values)
dev_test = deviance_gamma(test_df["ClaimAmount"].values, test_df["pred_sev"].values)

print(f"Gamma deviance — train: {dev_train:.4f}")
print(f"Gamma deviance — test:  {dev_test:.4f}")

rmse_test = rmse(test_df["ClaimAmount"].values, test_df["pred_sev"].values)
mae_test = mae(test_df["ClaimAmount"].values, test_df["pred_sev"].values)
print(f"\nRMSE (test): {rmse_test:,.2f}")
print(f"MAE  (test): {mae_test:,.2f}")

# %% [markdown]
# ## 7. Residual diagnostics

# %%
test_df["resid"] = test_df["ClaimAmount"] - test_df["pred_sev"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Residuals vs predicted
axes[0].scatter(test_df["pred_sev"], test_df["resid"], alpha=0.3, s=5, color="darkorange")
axes[0].axhline(0, color="red", linewidth=0.8)
axes[0].set_xlabel("Predicted Severity")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs. Predicted")

# Residual distribution
axes[1].hist(test_df["resid"].clip(-5000, 5000), bins=100,
             color="darkorange", edgecolor="white")
axes[1].set_title("Residual Distribution (clipped)")
axes[1].set_xlabel("Residual")

# Q-Q plot of standardised residuals
resid_std = (test_df["resid"] - test_df["resid"].mean()) / test_df["resid"].std()
stats.probplot(resid_std.clip(-5, 5), dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot (Standardised Residuals)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Observed vs predicted by decile

# %%
lift = lift_table(
    test_df["ClaimAmount"].values,
    test_df["pred_sev"].values,
    n_bins=10,
)
display(lift)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(lift))
ax.bar(x, lift["observed_mean"], width=0.4, label="Observed", color="darkorange", align="center")
ax.bar([i + 0.4 for i in x], lift["predicted_mean"], width=0.4, label="Predicted", color="steelblue", align="center")
ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels([f"D{i+1}" for i in x], rotation=0)
ax.set_xlabel("Decile (by predicted severity)")
ax.set_ylabel("Mean Claim Amount (€)")
ax.set_title("Severity Model — Lift Chart (Test Set)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - Fitted Gamma GLM (log-link) on policy-level aggregated claim amounts
# - Heavy right-tail makes severity inherently harder to predict than frequency
# - Gamma deviance is moderate; Q-Q plot shows expected heavy-tail departure
# - Severity predictions are relatively stable across features — frequency
#   is the primary driver of pure-premium differentiation
# - Next: combine frequency × severity → expected loss in notebook 05
