"""
Project configuration — paths, filenames, and column constants.
"""
from pathlib import Path

# ── Project root (two levels up from this file) ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data directories ──
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Raw filenames ──
FREQ_FILE = "freMTPL2freq.csv"
SEV_FILE = "freMTPL2sev.csv"

FREQ_PATH = RAW_DIR / FREQ_FILE
SEV_PATH = RAW_DIR / SEV_FILE

# ── Processed filenames ──
FREQ_PROCESSED = PROCESSED_DIR / "frequency_base.csv"
SEV_PROCESSED = PROCESSED_DIR / "severity_base.csv"
FEATURES_PROCESSED = PROCESSED_DIR / "features.csv"

# ── Schema constants ──
FREQ_REQUIRED_COLS = ["IDpol", "ClaimNb", "Exposure"]
SEV_REQUIRED_COLS = ["IDpol", "ClaimAmount"]

FREQ_ALL_COLS = [
    "IDpol", "ClaimNb", "Exposure", "VehPower", "VehAge",
    "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area",
    "Density", "Region",
]

CATEGORICAL_COLS = ["VehBrand", "VehGas", "Area", "Region"]
NUMERIC_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

# ── Modeling constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.2
