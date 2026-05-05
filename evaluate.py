
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def smape(y_true, y_pred):
    
    num   = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-10
    return float(np.mean(num / denom) * 100)


def direction_accuracy(y_true, y_pred):
    
    return float(np.mean(np.sign(y_pred) == np.sign(np.array(y_true))))


def mean_ic(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ics = []
    for i in range(len(y_true)):
        rt, rp = y_true[i], y_pred[i]
        if np.std(rt) > 0 and np.std(rp) > 0:
            ic, _ = spearmanr(rt, rp)
            if not np.isnan(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float('nan')


def icir(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ics = []
    for i in range(len(y_true)):
        rt, rp = y_true[i], y_pred[i]
        if np.std(rt) > 0 and np.std(rp) > 0:
            ic, _ = spearmanr(rt, rp)
            if not np.isnan(ic):
                ics.append(ic)
    if len(ics) > 1:
        return float(np.mean(ics) / (np.std(ics) + 1e-8))
    return float('nan')


def mse_score(y_true, y_pred):
    
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


# ---------------------------------------------------------------------------
# Per-model metric bundle
# ---------------------------------------------------------------------------

def compute_metrics(y_val, y_val_pred, y_test, y_test_pred, label=""):
   
    metrics = dict(
        val_smape  = smape(y_val,  y_val_pred),
        test_smape = smape(y_test, y_test_pred),
        val_ic     = mean_ic(y_val,  y_val_pred),
        test_ic    = mean_ic(y_test, y_test_pred),
        test_icir  = icir(y_test,    y_test_pred),
        test_mse   = mse_score(y_test, y_test_pred),
        dir_acc    = direction_accuracy(y_test, y_test_pred),
    )
    if label:
        print(
            f"{label:20s}  "
            f"Val IC: {metrics['val_ic']:+.5f}  "
            f"Test IC: {metrics['test_ic']:+.5f}  "
            f"ICIR: {metrics['test_icir']:+.4f}  "
            f"DirAcc: {metrics['dir_acc']:.4f}"
        )
    return metrics




def build_results_table(all_metrics):
    

    df = pd.DataFrame(all_metrics).T.rename(columns={
        'val_smape':  'Val SMAPE',
        'test_smape': 'Test SMAPE',
        'val_ic':     'Val IC',
        'test_ic':    'Test IC',
        'test_icir':  'Test ICIR',
        'test_mse':   'Test MSE',
        'dir_acc':    'Dir. Acc',
    })
    df.index.name = 'Model'
    return df.sort_values('Val IC', ascending=False).round(6)



def ic_robustness_linkage(results_df, robustness_df):
   
    ic_series = results_df.loc[robustness_df.index, 'Test IC']
    mvo_deg   = robustness_df['MVO Degradation']
    cvar_deg  = robustness_df['CVaR Degradation']

    rho_mvo,  p_mvo  = spearmanr(ic_series, mvo_deg)
    rho_cvar, p_cvar = spearmanr(ic_series, cvar_deg)

    linkage_df = pd.DataFrame({
        'Test IC':           ic_series,
        'MVO Degradation':   mvo_deg,
        'CVaR Degradation':  cvar_deg,
        'IC Rank':           ic_series.rank(ascending=False),
        'MVO Degrad. Rank':  mvo_deg.rank(ascending=False),
        'CVaR Degrad. Rank': cvar_deg.rank(ascending=False),
    })
    return linkage_df, rho_mvo, p_mvo, rho_cvar, p_cvar
