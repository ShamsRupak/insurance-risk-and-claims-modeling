.PHONY: venv jupyter test clean

PYTHON ?= python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

## Create virtual environment and install dependencies
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PY) -m ipykernel install --user --name insurance-risk --display-name "Insurance Risk"
	@echo "\n✅  Virtual environment ready. Activate with: source $(VENV)/bin/activate"

## Launch Jupyter notebook server
jupyter:
	$(VENV)/bin/jupyter notebook --notebook-dir=notebooks

## Smoke-test: verify all key imports succeed
test:
	$(PY) -c "\
	import pandas, numpy, sklearn, statsmodels, matplotlib, seaborn, scipy; \
	from src.data.io import load_freq, load_sev; \
	from src.data.preprocess import clean_freq, clean_sev; \
	from src.features.build_features import add_log_density, add_vehage_bin; \
	from src.models.frequency import PoissonFrequencyModel; \
	from src.models.severity import GammaSeverityModel; \
	from src.models.expected_loss import compute_pure_premium; \
	from src.evaluation.metrics import deviance_poisson, deviance_gamma; \
	print('\n✅  All imports OK')"

## Remove compiled Python files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "✅  Cleaned"
