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
# # 01 — Data Loading & Validation
#
# **Goal:** Load the French MTPL2 frequency and severity datasets, validate
# schemas, run basic quality checks, and save clean base tables to
# `data/processed/`.

# %% tags=["imports"]
import sys, os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), ""))

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.4f}".format)

# %% [markdown]
# ## 1. Load raw datasets using `src.data.io`

# %%
from src.data.io import load_freq, load_sev
from src.utils.config import RAW_DIR, PROCESSED_DIR, FREQ_PATH, SEV_PATH

print("RAW_DIR:", RAW_DIR)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("FREQ_PATH exists?", FREQ_PATH.exists())
print("SEV_PATH exists?", SEV_PATH.exists())

# %%
freq = load_freq()
sev = load_sev()

assert not freq.empty, "Frequency dataset is empty"
assert not sev.empty, "Severity dataset is empty"

print("\n✅ Loaded datasets")
print("Frequency shape:", freq.shape)
print("Severity shape:", sev.shape)

# %% [markdown]
# ## 2. Quick preview

# %%
display(freq.head())
display(sev.head())

# %% [markdown]
# ## 3. Schema inspection

# %%
print("--- Frequency info ---")
display(freq.info())

print("\n--- Severity info ---")
display(sev.info())

print("\nFrequency columns:\n", list(freq.columns))
print("\nSeverity columns:\n", list(sev.columns))

# %% [markdown]
# ## 4. Data quality checks

# %%
# Missing values
print("--- Missing value rate ---")
display(freq.isnull().mean().sort_values(ascending=False).head(10))
display(sev.isnull().mean().sort_values(ascending=False))

# Exposure should be > 0
bad_exposure = (freq["Exposure"] <= 0).sum()
print("\nBad Exposure (<=0) count:", bad_exposure)

# ClaimNb should be non-negative integer
neg_claims = (freq["ClaimNb"] < 0).sum()
print("Negative ClaimNb count:", neg_claims)

non_integer_claimnb = (~np.isclose(freq["ClaimNb"], np.round(freq["ClaimNb"]))).sum()
print("Non-integer ClaimNb count:", non_integer_claimnb)

# %% [markdown]
# ## 5. Distribution snapshots

# %%
print("--- ClaimNb distribution (normalized) ---")
display(freq["ClaimNb"].value_counts(normalize=True).sort_index().head(20))

print("\n--- ClaimAmount describe ---")
display(sev["ClaimAmount"].describe())

print("\n--- ClaimAmount quantiles ---")
display(sev["ClaimAmount"].quantile([0.5, 0.9, 0.99, 0.999]))

skew_val = sev["ClaimAmount"].skew()
print(f"\nClaimAmount skew: {skew_val:.2f}")

# %% [markdown]
# ## 6. Clean & save processed base tables

# %%
from src.data.preprocess import clean_freq, clean_sev, save_processed

freq_clean = clean_freq(freq)
sev_clean = clean_sev(sev)

print("Cleaned frequency shape:", freq_clean.shape)
print("Cleaned severity shape:", sev_clean.shape)

fp, sp = save_processed(freq_clean, sev_clean)
print("\n✅ Saved processed base tables:")
print(" -", fp)
print(" -", sp)

# %% [markdown]
# ## Summary
#
# - Loaded MTPL frequency (678k policies) and severity (26.6k claims) datasets
# - Validated schema: `IDpol`, `ClaimNb`, `Exposure`, `ClaimAmount` present
# - No missing values; exposure and claim counts are valid
# - Strong zero-inflation in claim counts (~95% ClaimNb == 0)
# - Heavy right-skew in claim severity (skew > 100)
# - Saved clean base tables to `data/processed/`
