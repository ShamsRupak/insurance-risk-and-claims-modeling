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
# # 02 — Exploratory Data Analysis
#
# **Goal:** Understand the distributions, correlations, and key patterns in the
# cleaned MTPL frequency and severity data before modelling.

# %% tags=["imports"]
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), ""))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.4f}".format)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# %% [markdown]
# ## 1. Load processed data

# %%
from src.utils.config import FREQ_PROCESSED, SEV_PROCESSED

freq = pd.read_csv(FREQ_PROCESSED)
sev = pd.read_csv(SEV_PROCESSED)

print("Frequency shape:", freq.shape)
print("Severity shape:", sev.shape)

# %% [markdown]
# ## 2. Portfolio overview

# %%
print("=== Frequency dataset summary ===")
display(freq.describe())

print("\n=== Severity dataset summary ===")
display(sev.describe())

# %%
# Claim rate
total_exposure = freq["Exposure"].sum()
total_claims = freq["ClaimNb"].sum()
observed_freq = total_claims / total_exposure
print(f"Total exposure: {total_exposure:,.1f}")
print(f"Total claims: {total_claims:,}")
print(f"Observed frequency: {observed_freq:.4f}")

# %% [markdown]
# ## 3. Target variable distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ClaimNb distribution
freq["ClaimNb"].value_counts().sort_index().plot.bar(ax=axes[0], color="steelblue")
axes[0].set_title("ClaimNb Distribution")
axes[0].set_xlabel("Number of Claims")
axes[0].set_ylabel("Count")

# ClaimAmount distribution (log scale)
sev["ClaimAmount"].clip(upper=sev["ClaimAmount"].quantile(0.99)).hist(
    bins=100, ax=axes[1], color="darkorange", edgecolor="white"
)
axes[1].set_title("ClaimAmount Distribution (99th percentile cap)")
axes[1].set_xlabel("Claim Amount (€)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# %%
# Log-scale severity
fig, ax = plt.subplots(figsize=(10, 4))
np.log10(sev["ClaimAmount"]).hist(bins=100, ax=ax, color="darkorange", edgecolor="white")
ax.set_title("log₁₀(ClaimAmount) Distribution")
ax.set_xlabel("log₁₀(ClaimAmount)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Exposure distribution

# %%
fig, ax = plt.subplots(figsize=(10, 4))
freq["Exposure"].hist(bins=50, ax=ax, color="seagreen", edgecolor="white")
ax.set_title("Exposure Distribution")
ax.set_xlabel("Exposure (years)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Numeric feature distributions

# %%
numeric_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    freq[col].hist(bins=50, ax=axes[i], color="steelblue", edgecolor="white")
    axes[i].set_title(col)

# Density on log scale
axes[5].hist(np.log1p(freq["Density"]), bins=50, color="steelblue", edgecolor="white")
axes[5].set_title("log(1 + Density)")

plt.suptitle("Numeric Feature Distributions", y=1.01, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Categorical feature distributions

# %%
cat_cols = ["VehBrand", "VehGas", "Area", "Region"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    freq[col].value_counts().sort_index().plot.bar(ax=axes[i], color="steelblue")
    axes[i].set_title(col)
    axes[i].tick_params(axis="x", rotation=45)

plt.suptitle("Categorical Feature Distributions", y=1.01, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Claim frequency by feature

# %%
from src.features.build_features import (
    add_vehage_bin, add_drivage_bin, add_bonusmalus_bin, add_claim_flag,
)

freq_feat = add_vehage_bin(freq)
freq_feat = add_drivage_bin(freq_feat)
freq_feat = add_bonusmalus_bin(freq_feat)
freq_feat = add_claim_flag(freq_feat)

# %%
def plot_claim_rate(df, group_col, ax=None):
    """Plot observed claim frequency by group."""
    grouped = df.groupby(group_col, observed=True).agg(
        exposure=("Exposure", "sum"),
        claims=("ClaimNb", "sum"),
    )
    grouped["frequency"] = grouped["claims"] / grouped["exposure"]
    grouped["frequency"].plot.bar(ax=ax, color="coral")
    if ax:
        ax.set_title(f"Claim Frequency by {group_col}")
        ax.set_ylabel("Claims / Exposure")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(["VehAgeBin", "DrivAgeBin", "BonusMalusBin",
                          "VehGas", "Area", "Region"]):
    plot_claim_rate(freq_feat, col, ax=axes[i])
    axes[i].tick_params(axis="x", rotation=45)

plt.suptitle("Observed Claim Frequency by Feature", y=1.01, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Severity by feature
#
# Merge severity onto frequency to analyse severity by policy features.

# %%
# Aggregate severity to policy level (mean claim amount per policy)
sev_agg = sev.groupby("IDpol")["ClaimAmount"].mean().reset_index()
sev_agg.columns = ["IDpol", "AvgClaimAmount"]

freq_sev = freq_feat.merge(sev_agg, on="IDpol", how="inner")
print("Policies with claims:", freq_sev.shape[0])

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(["DrivAgeBin", "VehAgeBin", "Area"]):
    freq_sev.groupby(col, observed=True)["AvgClaimAmount"].median().plot.bar(
        ax=axes[i], color="darkorange"
    )
    axes[i].set_title(f"Median Severity by {col}")
    axes[i].set_ylabel("Median Claim Amount (€)")
    axes[i].tick_params(axis="x", rotation=45)

plt.suptitle("Claim Severity by Feature", y=1.01, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Correlation heatmap (numeric features)

# %%
corr_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge",
             "DrivAge", "BonusMalus", "Density"]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    freq[corr_cols].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax
)
ax.set_title("Correlation Matrix — Numeric Features")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# **Key EDA findings:**
#
# 1. **Zero-inflation:** ~95% of policies have zero claims → Poisson GLM with
#    exposure offset is appropriate.
# 2. **Severity skew:** Claim amounts are extremely right-skewed (skew > 100);
#    Gamma GLM with log-link is a natural choice.
# 3. **Young drivers** (18-25) show notably higher claim frequency.
# 4. **BonusMalus > 100** is a strong predictor of claim occurrence.
# 5. **Urban areas** (high Density / Area F) have higher claim rates.
# 6. **Severity** appears less variable across features than frequency — the
#    frequency model will drive most of the pricing signal.
