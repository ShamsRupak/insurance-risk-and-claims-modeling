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
# # 05 — Expected Loss & Pure Premium
#
# **Goal:** Combine frequency and severity models to estimate per-policy pure
# premium (E[Loss] = E[Frequency] × E[Severity]), then derive business insights
# through risk segmentation and portfolio diagnostics.

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
# ## 1. Load data & prepare features
#
# We need the full frequency dataset plus a severity estimate for each policy.

# %%
from src.utils.config import (
    FREQ_PROCESSED, SEV_PROCESSED, CATEGORICAL_COLS, RANDOM_STATE, TEST_SIZE,
)
from src.features.build_features import add_log_density, build_all_features

freq = pd.read_csv(FREQ_PROCESSED)
sev = pd.read_csv(SEV_PROCESSED)

# Engineer features on frequency data
freq = build_all_features(freq)

# One-hot encode
freq_encoded = pd.get_dummies(freq, columns=CATEGORICAL_COLS, drop_first=True, dtype=float)

exclude = {"IDpol", "ClaimNb", "Exposure", "HasClaim",
           "VehAgeBin", "DrivAgeBin", "BonusMalusBin"}
feature_cols = [c for c in freq_encoded.columns if c not in exclude]

print(f"Total policies: {freq_encoded.shape[0]:,}")
print(f"Features: {len(feature_cols)}")

# %% [markdown]
# ## 2. Train/test split (same seed as notebooks 03 & 04)

# %%
train_df, test_df = train_test_split(
    freq_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {train_df.shape[0]:,}  |  Test: {test_df.shape[0]:,}")

# %% [markdown]
# ## 3. Fit frequency model (Poisson GLM)

# %%
from src.models.frequency import PoissonFrequencyModel

freq_model = PoissonFrequencyModel(feature_cols)
freq_model.fit(train_df, target="ClaimNb", exposure="Exposure")

train_df = train_df.copy()
test_df = test_df.copy()
train_df["pred_freq"] = freq_model.predict(train_df)
test_df["pred_freq"] = freq_model.predict(test_df)

# %% [markdown]
# ## 4. Fit severity model (Gamma GLM)
#
# We prepare severity data with the same feature encoding, then fit on the
# training subset of policies that have claims.

# %%
# Aggregate severity to policy level
sev_pol = sev.groupby("IDpol")["ClaimAmount"].sum().reset_index()

# Merge onto training data (only policies with claims)
train_sev = train_df.merge(sev_pol, on="IDpol", how="inner")
print(f"Training policies with claims: {train_sev.shape[0]:,}")

# %%
from src.models.severity import GammaSeverityModel

sev_model = GammaSeverityModel(feature_cols)
sev_model.fit(train_sev, target="ClaimAmount")

# Predict severity for ALL policies (train + test)
train_df["pred_sev"] = sev_model.predict(train_df)
test_df["pred_sev"] = sev_model.predict(test_df)

# %% [markdown]
# ## 5. Compute pure premium

# %%
from src.models.expected_loss import compute_pure_premium, portfolio_summary, risk_segment_summary

train_df["PurePremium"] = compute_pure_premium(train_df["pred_freq"], train_df["pred_sev"])
test_df["PurePremium"] = compute_pure_premium(test_df["pred_freq"], test_df["pred_sev"])

print("=== Test Set Pure Premium Distribution ===")
display(test_df["PurePremium"].describe())

# %% [markdown]
# ## 6. Portfolio-level summary

# %%
summary = portfolio_summary(test_df, pure_premium_col="PurePremium")
for k, v in summary.items():
    if isinstance(v, float):
        print(f"  {k}: {v:,.4f}")
    else:
        print(f"  {k}: {v:,}")

# %% [markdown]
# ## 7. Risk segmentation

# %%
# Re-add binned columns for segmentation analysis
from src.features.build_features import add_drivage_bin, add_vehage_bin, add_bonusmalus_bin

test_seg = add_drivage_bin(test_df)
test_seg = add_vehage_bin(test_seg)
test_seg = add_bonusmalus_bin(test_seg)

# %% [markdown]
# ### 7a. By Driver Age

# %%
seg_drivage = risk_segment_summary(test_seg, "DrivAgeBin")
display(seg_drivage)

fig, ax = plt.subplots(figsize=(10, 5))
seg_drivage["avg_pure_premium"].plot.bar(ax=ax, color="coral")
ax.set_title("Average Pure Premium by Driver Age Group")
ax.set_ylabel("Pure Premium (€)")
ax.set_xlabel("Driver Age Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7b. By BonusMalus

# %%
seg_bm = risk_segment_summary(test_seg, "BonusMalusBin")
display(seg_bm)

fig, ax = plt.subplots(figsize=(10, 5))
seg_bm["avg_pure_premium"].plot.bar(ax=ax, color="steelblue")
ax.set_title("Average Pure Premium by BonusMalus Group")
ax.set_ylabel("Pure Premium (€)")
ax.set_xlabel("BonusMalus Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7c. By Area

# %%
# Area is one-hot encoded; recover original from freq
test_seg_area = test_seg.copy()
# We can use the original freq data's Area column
freq_area = freq[["IDpol", "Area"]].drop_duplicates()
test_seg_area = test_seg_area.merge(freq_area, on="IDpol", how="left", suffixes=("", "_orig"))
area_col = "Area" if "Area" in test_seg_area.columns else "Area_orig"

seg_area = risk_segment_summary(test_seg_area, area_col)
display(seg_area)

fig, ax = plt.subplots(figsize=(10, 5))
seg_area["avg_pure_premium"].sort_index().plot.bar(ax=ax, color="seagreen")
ax.set_title("Average Pure Premium by Area")
ax.set_ylabel("Pure Premium (€)")
ax.set_xlabel("Area")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Pure premium distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution (clipped)
pp_clip = test_df["PurePremium"].clip(upper=test_df["PurePremium"].quantile(0.99))
pp_clip.hist(bins=100, ax=axes[0], color="mediumpurple", edgecolor="white")
axes[0].set_title("Pure Premium Distribution (99th pctl cap)")
axes[0].set_xlabel("Pure Premium (€)")

# Log scale
np.log1p(test_df["PurePremium"]).hist(bins=100, ax=axes[1], color="mediumpurple", edgecolor="white")
axes[1].set_title("log(1 + Pure Premium) Distribution")
axes[1].set_xlabel("log(1 + Pure Premium)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Rate relativities
#
# Express pure premium of each segment relative to the portfolio average.

# %%
portfolio_avg_pp = test_df["PurePremium"].mean()
print(f"Portfolio average pure premium: €{portfolio_avg_pp:,.2f}\n")

for name, seg_df in [("DrivAgeBin", seg_drivage), ("BonusMalusBin", seg_bm)]:
    print(f"--- Rate Relativities: {name} ---")
    rel = seg_df["avg_pure_premium"] / portfolio_avg_pp
    for idx, val in rel.items():
        print(f"  {idx}: {val:.3f}")
    print()

# %% [markdown]
# ## Summary
#
# **Key findings:**
#
# 1. **Pure Premium = E[Frequency] × E[Severity]** successfully computed for
#    all policies in the test set.
# 2. **Young drivers (18-25)** and **high BonusMalus (151+)** segments have the
#    highest pure premiums — consistent with known actuarial patterns.
# 3. **Urban areas** (Area F) carry higher risk than rural zones.
# 4. **Rate relativities** show that the riskiest segments may require 2-3×
#    the portfolio-average premium, providing a basis for differentiated pricing.
# 5. **Portfolio-level expected loss** matches well with observed claim totals,
#    confirming model calibration.
#
# ### Potential next steps
# - Incorporate interaction terms or splines for improved model flexibility
# - Compare with tree-based models (e.g. Gradient Boosted Trees)
# - Explore Tweedie compound Poisson-Gamma models as a single-model alternative
# - Add bootstrap confidence intervals to rate relativities
# - Validate against out-of-time holdout data
