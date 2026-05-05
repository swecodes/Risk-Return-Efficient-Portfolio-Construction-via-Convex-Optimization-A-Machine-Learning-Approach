# ML-Driven Portfolio Optimisation

> **Repository link:** `https://github.com/<your-username>/ml-portfolio-optimisation`
> *(Replace with your actual GitHub URL before submitting your report)*

A machine-learning pipeline that predicts next-day cross-sectional stock returns and feeds those predictions into three portfolio construction frameworks: **Mean-Variance Optimisation (MVO)**, **CVaR-constrained optimisation**, and **Hierarchical Risk Parity (HRP)**.

---

## Project Overview

| Stage | Module | Description |
|---|---|---|
| Data | `data_prep.py` | Price download, feature engineering, VADER sentiment, train/val/test splits, sequence construction |
| Models | `models.py` | Linear Regression, XGBoost, LSTM, BiLSTM, CNN-LSTM, SACRE |
| Evaluation | `evaluate.py` | IC, ICIR, SMAPE, Direction Accuracy, MSE |
| Optimisation | `optimize.py` | MVO, CVaR, HRP, walk-forward backtest, tail risk, robustness |
| Visualisation | `visualize.py` | Six publication-quality charts |
| Orchestration | `main.py` | End-to-end pipeline runner |

---

## Stock Universe

| Tickers | Benchmark | Macro | Sectors |
|---|---|---|---|
| NVDA, GOOGL, AAPL, MSFT, AMZN, TSM, AVGO, META, TSLA, JPM | ^GSPC | ^VIX, ^TNX, ^IRX, DX-Y.NYB | XLK, XLF, XLY |

**Data range:** 2015-01-01 to 2025-12-31  
**Train / Val / Test:** 2015–2022 / 2023 / 2024–2025

---

## Models

### Baseline
- **Linear Regression** — multi-output OLS, trained on the full scaled feature matrix

### Tree-based
- **XGBoost** — one XGBoost regressor per target stock via `MultiOutputRegressor`

### Deep Learning (sequence-based, lookback = 20 days)
- **LSTM** — 64 units, dropout 0.3, Adam 1e-3
- **BiLSTM** — 32 units per direction, L2 regularisation, Adam 5e-4
- **CNN-LSTM** — multi-scale causal convolutions (k ∈ {3,5,10}), two stacked LSTMs, learned temporal attention (AttentionReduce), Adam 5e-4
- **SACRE** *(Sentiment-Augmented Cross-Sectional Ranking Encoder)* — per-stock CNN+LSTM encoders, cross-stock multi-head attention, trained with the **ListNet** ranking loss on VADER-scored Yahoo Finance headlines

---

## Portfolio Construction

| Method | Objective | Constraint |
|---|---|---|
| MVO | Maximise alpha | Variance ≤ k; w ∈ [0, 0.25] |
| CVaR | Maximise alpha | CVaR₉₅ ≤ budget; w ∈ [0, 0.25] |
| HRP | Variance-weighted | Hierarchical clustering; no ML signal |

Walk-forward backtesting: 7 rebalancing periods over the 2024–2025 test window.

---

## Evaluation Metrics

| Metric | Primary use |
|---|---|
| **Mean IC** (Spearman) | Cross-sectional forecasting quality — primary ranking metric |
| **ICIR** | Signal consistency over time |
| SMAPE | Error magnitude (reporting only) |
| Direction Accuracy | Sign-prediction rate |
| Ex-post Sharpe | Walk-forward risk-adjusted return |
| Max Drawdown | Worst peak-to-trough loss |
| Realised VaR / CVaR | Tail-risk measurement |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ml-portfolio-optimisation.git
cd ml-portfolio-optimisation

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

> **Note:** Training all deep-learning models end-to-end takes approximately 30–60 minutes on a CPU. A GPU is recommended for faster training.

---

## Repository Structure

```
ml-portfolio-optimisation/
├── data_prep.py          # Data pipeline
├── models.py             # Model definitions and training
├── evaluate.py           # Metrics and comparison utilities
├── optimize.py           # Portfolio optimisation and backtesting
├── visualize.py          # Plotting functions
├── main.py               # End-to-end orchestration script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Key Findings

- **SACRE** achieves the lowest realised volatility and drawdown among all MVO portfolios, validating the defensive value of cross-sectional ranking with sentiment enrichment.
- **LSTM** delivers the highest ex-post Sharpe and return, at the cost of significantly elevated tail risk.
- **CVaR** optimisation consistently increases return at the expense of a lower Sharpe ratio compared to MVO, suggesting the CVaR constraint is less binding than the variance constraint for these predictions.
- A statistically significant **negative correlation (ρ ≈ −0.886)** exists between a model's Test IC and its MVO Sharpe degradation under weight noise — higher-IC models are paradoxically more brittle to perturbation.
- **HRP** remains competitive, demonstrating the value of robust correlation-based diversification even without a predictive signal.

---

## Citation / Academic Use

If you reference this codebase in a report, please cite the repository URL and note the data source (Yahoo Finance via `yfinance`) and the sentiment library (`vaderSentiment`).
