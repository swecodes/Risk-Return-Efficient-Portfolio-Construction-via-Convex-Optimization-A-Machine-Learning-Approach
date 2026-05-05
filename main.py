

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_prep import (
    build_all_datasets,
    TICKERS, BENCHMARK, WINDOW,
    TRAIN_END, VAL_START, VAL_END, TEST_START
)
from models import (
    train_linear_regression,
    train_xgboost,
    train_lstm,
    train_bilstm,
    train_cnnlstm,
    train_sacre,
)
from evaluate import (
    compute_metrics,
    build_results_table,
    ic_robustness_linkage,
)
from optimize import (
    compute_betas,
    build_alpha_vectors,
    min_variance_portfolio,
    build_mvo_portfolios,
    build_cvar_portfolios,
    calculate_hrp_weights,
    walk_forward_backtest,
    tail_risk_stats,
    perturbed_sharpe,
    diversification_analysis,
)
from visualize import (
    plot_prediction_quality,
    plot_sharpe_comparison,
    plot_risk_reward,
    plot_robustness_linkage,
    plot_allocation_dashboard,
    plot_hrp_comparison,
)


RF = 0.05  # annual risk-free rate


print("\n" + "=" * 70)
print("STEP 1: DATA PREPARATION")
print("=" * 70)

data = build_all_datasets()

base         = data['base']
sent         = data['sent']
n_features   = data['n_features']
n_feat_sent  = data['n_features_sent']
n_outputs    = data['n_outputs']

X_tr_seq = data['X_tr_seq'];  y_tr_seq = data['y_tr_seq']
X_vl_seq = data['X_vl_seq'];  y_vl_seq = data['y_vl_seq']
X_ts_seq = data['X_ts_seq'];  y_ts_seq = data['y_ts_seq']
test_idx_seq = data['test_idx_seq']

X_tr_sacre = data['X_tr_sacre'];  y_tr_sacre = data['y_tr_sacre']
X_vl_sacre = data['X_vl_sacre'];  y_vl_sacre = data['y_vl_sacre']
X_ts_sacre = data['X_ts_sacre'];  y_ts_sacre = data['y_ts_sacre']
y_ts_raw_sacre = data['y_ts_raw_sacre']
sacre_idx_seq  = data['sacre_idx_seq']

stock_rets = data['stock_rets']
bench_rets = data['bench_rets']


print("\n" + "=" * 70)
print("STEP 2: MODEL TRAINING")
print("=" * 70)

# Linear Regression 
print("\nTraining Linear Regression …")
lr_model, lr_val_preds, lr_test_preds = train_linear_regression(
    base['X_train_sc'], base['y_train'],
    base['X_val_sc'],   base['X_test_sc']
)

# XGBoost
print("Training XGBoost …")
xgb_model, xgb_val_preds, xgb_test_preds = train_xgboost(
    base['X_train_sc'], base['y_train'],
    base['X_val_sc'],   base['X_test_sc']
)

# LSTM 
print("Training LSTM …")
lstm_model, _, lstm_val_preds, lstm_test_preds = train_lstm(
    X_tr_seq, y_tr_seq, X_vl_seq, y_vl_seq, X_ts_seq,
    WINDOW, n_features, n_outputs
)

#BiLSTM 
print("Training BiLSTM …")
bilstm_model, _, bilstm_val_preds, bilstm_test_preds = train_bilstm(
    X_tr_seq, y_tr_seq, X_vl_seq, y_vl_seq, X_ts_seq,
    WINDOW, n_features, n_outputs
)

# CNN-LSTM 
print("Training CNN-LSTM …")
cnn_model, _, cnn_val_preds, cnn_test_preds = train_cnnlstm(
    X_tr_seq, y_tr_seq, X_vl_seq, y_vl_seq, X_ts_seq,
    WINDOW, n_features, n_outputs
)

# SACRE 
print("Training SACRE …")
sacre_model, _, sacre_val_preds, sacre_test_preds = train_sacre(
    X_tr_sacre, y_tr_sacre, X_vl_sacre, y_vl_sacre, X_ts_sacre,
    WINDOW, n_feat_sent, len(TICKERS)
)


print("\n" + "=" * 70)
print("STEP 3: EVALUATION")
print("=" * 70)

all_metrics = {
    'Linear Regression': compute_metrics(
        base['y_val'].values,    lr_val_preds,
        base['y_test'].values,   lr_test_preds,
        label='Linear Regression'
    ),
    'XGBoost': compute_metrics(
        base['y_val'].values,    xgb_val_preds,
        base['y_test'].values,   xgb_test_preds,
        label='XGBoost'
    ),
    'LSTM': compute_metrics(
        y_vl_seq, lstm_val_preds,
        y_ts_seq, lstm_test_preds,
        label='LSTM'
    ),
    'BiLSTM': compute_metrics(
        y_vl_seq, bilstm_val_preds,
        y_ts_seq, bilstm_test_preds,
        label='BiLSTM'
    ),
    'CNN-LSTM': compute_metrics(
        y_vl_seq, cnn_val_preds,
        y_ts_seq, cnn_test_preds,
        label='CNN-LSTM'
    ),
    'SACRE': compute_metrics(
        y_vl_sacre, sacre_val_preds,
        y_ts_sacre, sacre_test_preds,
        label='SACRE'
    ),
}

eval_df = build_results_table(all_metrics)
print("\n── ML Prediction Quality (ranked by Val IC) ──")
print(eval_df[['Val IC', 'Test IC', 'Test ICIR', 'Val SMAPE', 'Test SMAPE', 'Dir. Acc']])
print(f"\nBest predictor (Val IC): {eval_df['Val IC'].idxmax()}")

actual_seq_df  = pd.DataFrame(y_ts_seq,       columns=TICKERS, index=test_idx_seq)
sacre_actual_df = pd.DataFrame(y_ts_raw_sacre, columns=TICKERS, index=sacre_idx_seq)

pred_map = {
    'Linear Regression': (actual_seq_df, pd.DataFrame(lr_test_preds[WINDOW:],   columns=TICKERS, index=test_idx_seq)),
    'XGBoost':           (actual_seq_df, pd.DataFrame(xgb_test_preds[WINDOW:],  columns=TICKERS, index=test_idx_seq)),
    'LSTM':              (actual_seq_df, pd.DataFrame(lstm_test_preds,           columns=TICKERS, index=test_idx_seq)),
    'BiLSTM':            (actual_seq_df, pd.DataFrame(bilstm_test_preds,         columns=TICKERS, index=test_idx_seq)),
    'CNN-LSTM':          (actual_seq_df, pd.DataFrame(cnn_test_preds,            columns=TICKERS, index=test_idx_seq)),
    'SACRE':             (sacre_actual_df, pd.DataFrame(sacre_test_preds,        columns=TICKERS, index=sacre_idx_seq)),
}

Rm_train     = bench_rets.loc[:TRAIN_END]
beta_vector  = compute_betas(base['y_train'], TICKERS, bench_rets)
Rm_annual    = Rm_train.mean() * 252

Sigma = base['y_train'][TICKERS].cov().values * 252

mu_vectors, alpha_vectors = build_alpha_vectors(pred_map, beta_vector, Rm_annual)

print("\n" + "=" * 70)
print("STEP 5: PORTFOLIO OPTIMISATION")
print("=" * 70)

min_w, min_var = min_variance_portfolio(Sigma)

print("\nMVO tangency portfolios …")
mvo_weights, mvo_summary = build_mvo_portfolios(
    pred_map, alpha_vectors, mu_vectors, Sigma, min_var, rf=RF
)
print(mvo_summary.to_string())

print("\nCVaR portfolios …")
scenarios = base['y_train'][TICKERS].values
cvar_weights, cvar_summary = build_cvar_portfolios(
    pred_map, alpha_vectors, mu_vectors, Sigma, scenarios, min_var, rf=RF
)
print(cvar_summary.to_string())

print("\nHierarchical Risk Parity …")
hrp_weights_series, linkage_mat = calculate_hrp_weights(base['y_train'][TICKERS])
hrp_w_array = hrp_weights_series[TICKERS].values
print((hrp_weights_series[TICKERS] * 100).round(2))


print("\n" + "=" * 70)
print("STEP 6: WALK-FORWARD BACKTEST")
print("=" * 70)

def get_actual(name):
    return sacre_actual_df if name == 'SACRE' else actual_seq_df

# MVO
wf_mvo_rows = [
    walk_forward_backtest({n: w}, get_actual(n), n_periods=7, rf=RF)
    for n, w in mvo_weights.items()
]
wf_df = pd.concat(wf_mvo_rows)
print("\nMVO Walk-Forward (ex-post):")
print(wf_df.to_string())

# CVaR
wf_cvar_rows = [
    walk_forward_backtest({n: w}, get_actual(n), n_periods=7, rf=RF)
    for n, w in cvar_weights.items()
]
wf_cvar_df = pd.concat(wf_cvar_rows)
print("\nCVaR Walk-Forward (ex-post):")
print(wf_cvar_df.to_string())

# HRP
hrp_weights_map = {'HRP': hrp_w_array}
wf_hrp = walk_forward_backtest(hrp_weights_map, actual_seq_df, n_periods=7, rf=RF)
print("\nHRP Walk-Forward (ex-post):")
print(wf_hrp.to_string())

# Benchmark
gspc_test    = bench_rets.reindex(actual_seq_df.index).fillna(0).values
gspc_ann_ret = float(np.mean(gspc_test) * 252)
gspc_ann_vol = float(np.std(gspc_test)  * np.sqrt(252))
gspc_sharpe  = (gspc_ann_ret - RF) / gspc_ann_vol
print(f"\nS&P 500  Return: {gspc_ann_ret*100:.2f}%  Vol: {gspc_ann_vol*100:.2f}%  Sharpe: {gspc_sharpe:.4f}")

print("\n" + "=" * 70)
print("STEP 7: RISK AND ROBUSTNESS ANALYSIS")
print("=" * 70)

tail_mvo_df  = tail_risk_stats(mvo_weights,  get_actual)
tail_cvar_df = tail_risk_stats(cvar_weights, get_actual)
tail_hrp_df  = tail_risk_stats(hrp_weights_map, lambda n: actual_seq_df)

print("\nRealized Tail-Risk — MVO:")
print(tail_mvo_df.to_string())
print("\nRealized Tail-Risk — CVaR:")
print(tail_cvar_df.to_string())
print("\nRealized Tail-Risk — HRP:")
print(tail_hrp_df.to_string())

rob_mvo  = perturbed_sharpe(mvo_weights,  get_actual).add_prefix('MVO ')
rob_cvar = perturbed_sharpe(cvar_weights, get_actual).add_prefix('CVaR ')
robustness_df = rob_mvo.join(rob_cvar)
print("\nRobustness Summary:")
print(robustness_df.to_string())

linkage_df, rho_mvo, p_mvo, rho_cvar, p_cvar = ic_robustness_linkage(eval_df, robustness_df)
print(f"\nSpearman ρ (Test IC vs MVO degradation):  {rho_mvo:.3f}  (p={p_mvo:.3f})")
print(f"Spearman ρ (Test IC vs CVaR degradation): {rho_cvar:.3f}  (p={p_cvar:.3f})")

alloc_df = diversification_analysis(mvo_weights, cvar_weights)
print("\nDiversification Analysis:")
print(alloc_df.to_string())


print("\n" + "=" * 70)
print("STEP 8: VISUALISATION")
print("=" * 70)

plot_prediction_quality(eval_df)

comparison_df = pd.DataFrame({
    'MVO Sharpe':      wf_df['Ex-post Sharpe'],
    'CVaR Sharpe':     wf_cvar_df['Ex-post Sharpe'],
    'MVO Alpha (%)':   {n: round(wf_df.loc[n, 'Ann. Return (%)'] - gspc_ann_ret * 100, 2)
                        for n in wf_df.index if n in wf_cvar_df.index},
    'CVaR Alpha (%)':  {n: round(wf_cvar_df.loc[n, 'Ann. Return (%)'] - gspc_ann_ret * 100, 2)
                        for n in wf_cvar_df.index},
    'MVO Return (%)':  wf_df['Ann. Return (%)'],
    'CVaR Return (%)': wf_cvar_df['Ann. Return (%)'],
    'MVO MaxDD (%)':   wf_df['Max Drawdown (%)'],
    'CVaR MaxDD (%)':  wf_cvar_df['Max Drawdown (%)'],
})

plot_sharpe_comparison(comparison_df, gspc_sharpe)
plot_risk_reward(wf_df, wf_cvar_df, gspc_ann_vol, gspc_ann_ret)
plot_robustness_linkage(linkage_df, rho_mvo, p_mvo)

target_model = 'SACRE'
plot_allocation_dashboard(
    model_name=f"{target_model} Portfolio",
    tickers=TICKERS,
    mvo_w=mvo_weights[target_model],
    cvar_w=cvar_weights[target_model],
)

df_compare = pd.DataFrame({
    'MVO (SACRE)':      mvo_weights[target_model],
    'CVaR (SACRE)':     cvar_weights[target_model],
    'HRP (No ML)':      hrp_weights_series[TICKERS].values,
}, index=TICKERS)

plot_hrp_comparison(linkage_mat, TICKERS, df_compare, target_model)

print("\n" + "=" * 70)
print("Pipeline complete.")
print("=" * 70)
