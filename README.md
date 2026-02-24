# Insurance Risk & Claims Modeling

End-to-end data science project for **insurance risk, claim frequency, and claim severity modeling** using the French Motor Third-Party Liability (MTPL) datasets. The project emphasizes statistical rigor, model interpretability, and actionable business insights for pricing and reserving.

## Project Goals

| Objective | Approach |
|---|---|
| **Claim Frequency** | Poisson GLM with log-link; offset by log(Exposure) |
| **Claim Severity** | Gamma GLM with log-link on positive claims |
| **Pure Premium** | E[Loss] = E[Frequency] × E[Severity] |
| **Business Insight** | Risk segmentation, rate relativities, portfolio diagnostics |

## Dataset Description

This project uses the **French MTPL2** dataset, a widely used actuarial benchmark:

- **`freMTPL2freq.csv`** — Policy-level data with claim counts  
  Key columns: `IDpol`, `ClaimNb`, `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area`, `Density`, `Region`  
  ~678k policies

- **`freMTPL2sev.csv`** — Individual claim amounts  
  Key columns: `IDpol`, `ClaimAmount`  
  ~26.6k claims

Source: [OpenML freMTPL2](https://www.openml.org/d/41214) / R `CASdatasets` package

## Repository Structure

```
├── data/
│   ├── raw/               # Place freMTPL2freq.csv & freMTPL2sev.csv here
│   └── processed/         # Cleaned outputs from preprocessing
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_frequency_model.ipynb
│   ├── 04_severity_model.ipynb
│   └── 05_expected_loss.ipynb
├── src/
│   ├── data/              # Loading & preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Frequency, severity, expected-loss models
│   ├── evaluation/        # Metrics & diagnostics
│   └── utils/             # Configuration & helpers
├── Makefile
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/ShamsRupak/insurance-risk-and-claims-modeling.git
cd insurance-risk-and-claims-modeling
make venv
source .venv/bin/activate
```

### 2. Download raw data

Download the two CSV files from [OpenML](https://www.openml.org/d/41214) and place them in:

```
data/raw/freMTPL2freq.csv
data/raw/freMTPL2sev.csv
```

### 3. Verify setup

```bash
make test
```

### 4. Run notebooks in order

```bash
make jupyter
```

Then open and run the notebooks sequentially:

| # | Notebook | Purpose | Key Outputs |
|---|----------|---------|-------------|
| 1 | `01_data_loading.ipynb` | Load, validate, and save clean base tables | `data/processed/frequency_base.csv`, `severity_base.csv` |
| 2 | `02_eda.ipynb` | Exploratory data analysis & visualizations | Distribution plots, correlation analysis, portfolio summary |
| 3 | `03_frequency_model.ipynb` | Poisson GLM for claim frequency | Model coefficients, deviance residuals, lift charts |
| 4 | `04_severity_model.ipynb` | Gamma GLM for claim severity | Model coefficients, residual diagnostics |
| 5 | `05_expected_loss.ipynb` | Pure premium estimation & business insights | Risk segmentation, rate relativities, portfolio-level metrics |

## Modeling Summary

### Frequency Modeling
- **Poisson GLM** with log-link and `log(Exposure)` as offset
- Features: `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area`, `Density`, `Region`
- Evaluated via Poisson deviance, observed-vs-predicted lift charts

### Severity Modeling
- **Gamma GLM** with log-link on strictly positive claim amounts
- Same feature set as frequency model
- Evaluated via Gamma deviance, residual Q-Q plots

### Pure Premium (Expected Loss)
- **E[Loss] = E[Frequency] × E[Severity]**
- Combines both model predictions for per-policy pricing
- Portfolio-level aggregation for business diagnostics

## License

See [LICENSE](LICENSE) for details.
