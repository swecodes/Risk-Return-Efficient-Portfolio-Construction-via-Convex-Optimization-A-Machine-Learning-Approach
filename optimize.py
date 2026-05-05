
import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
#beta
def compute_betas(y_train, tickers, benchmark_rets):
    betas = {}
    Rm = benchmark_rets.reindex(y_train.index).fillna(0)
    for t in tickers:
        cov = np.cov(y_train[t], Rm)[0, 1]
        var = Rm.var()
        betas[t] = cov / var if var > 0 else 1.0
    return np.array([betas[t] for t in tickers])

#alpha
def compute_alpha_vector(mu_predicted, beta_vector, Rm_annual):
    return mu_predicted - beta_vector * Rm_annual


def build_alpha_vectors(pred_map, beta_vector, Rm_annual):
    mu_vectors    = {}
    alpha_vectors = {}
    for name, (_, pred_df) in pred_map.items():
        mu = pred_df.values.mean(axis=0) * 252
        mu_vectors[name]    = mu
        alpha_vectors[name] = compute_alpha_vector(mu, beta_vector, Rm_annual)
    return mu_vectors, alpha_vectors


#mv
def min_variance_portfolio(Sigma):
    n = Sigma.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Sigma)),
        [cp.sum(w) == 1, w >= 0]
    )
    prob.solve(solver=cp.CLARABEL, warm_start=True)
    return w.value, float(w.value @ Sigma @ w.value)

#mvo tangency
def markowitz_mvo(alpha_vec, Sigma, k=0.10, w_max=0.25):
    n = len(alpha_vec)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(alpha_vec @ w),
        [cp.quad_form(w, Sigma) <= k, cp.sum(w) == 1, w >= 0, w <= w_max]
    )
    prob.solve(solver=cp.CLARABEL, warm_start=True)
    if prob.status in ['optimal', 'optimal_inaccurate']:
        return w.value
    return None


def build_mvo_portfolios(pred_map, alpha_vectors, mu_vectors, Sigma,
                          min_var, rf=0.05, n_frontier=30, w_max=0.25):
    k_values = np.linspace(min_var * 1.10, np.max(np.diag(Sigma)), n_frontier)

    mvo_weights = {}
    summary_rows = []

    for name in pred_map:
        alpha_vec = alpha_vectors[name]
        frontier  = []

        for k in k_values:
            w = markowitz_mvo(alpha_vec, Sigma, k=k, w_max=w_max)
            if w is not None:
                frontier.append({
                    'k': k,
                    'Vol': np.sqrt(w @ Sigma @ w),
                    'Alpha': float(alpha_vec @ w),
                    'Expected Mu': float(mu_vectors[name] @ w),
                    'Weights': w
                })

        if not frontier:
            continue

        sharpe_vals = [
            (p['Expected Mu'] - rf) / p['Vol'] if p['Vol'] > 0 else -np.inf
            for p in frontier
        ]
        best = frontier[int(np.argmax(sharpe_vals))]
        mvo_weights[name] = best['Weights']
        summary_rows.append({
            'Model':            name,
            'Ex-ante Sharpe':   round(max(sharpe_vals), 4),
            'Port. Vol (%)':    round(best['Vol'] * 100, 2),
            'Port. Alpha':      round(best['Alpha'], 4)
        })

    return mvo_weights, pd.DataFrame(summary_rows).set_index('Model')
#cvar
def cvar_optimize(alpha_vec, scenarios, Sigma, min_var,
                  confidence=0.95, w_max=0.25, budget_mult=5.0):
    n = alpha_vec.shape[0]
    S = scenarios.shape[0]

    w    = cp.Variable(n, nonneg=True)
    zeta = cp.Variable()
    u    = cp.Variable(S, nonneg=True)

    losses   = -scenarios @ w
    budget   = float(np.sqrt(min_var)) * budget_mult
    cvar_exp = zeta + (1 / (S * (1 - confidence))) * cp.sum(u)

    prob = cp.Problem(
        cp.Maximize(alpha_vec @ w),
        [u >= losses - zeta, cvar_exp <= budget, cp.sum(w) == 1, w <= w_max]
    )
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
        port_ret  = float(alpha_vec @ w.value)
        port_vol  = float(np.sqrt(w.value @ Sigma @ w.value))
        port_cvar = float(
            zeta.value
            + (1 / (S * (1 - confidence)))
            * np.sum(np.maximum(-scenarios @ w.value - zeta.value, 0))
        )
        return w.value, port_ret, port_vol, port_cvar
    return None, None, None, None


def build_cvar_portfolios(pred_map, alpha_vectors, mu_vectors, Sigma,
                           scenarios, min_var, rf=0.05):
    cvar_weights = {}
    summary_rows = []

    for name in pred_map:
        alpha_vec = alpha_vectors[name]
        w, ret, vol, cvar_val = cvar_optimize(
            alpha_vec, scenarios, Sigma, min_var
        )
        if w is not None:
            sharpe = (mu_vectors[name] @ w - rf) / vol if vol > 0 else -np.inf
            cvar_weights[name] = w
            summary_rows.append({
                'Model':            name,
                'Ex-ante Sharpe':   round(sharpe, 4),
                'Port. Vol (%)':    round(vol * 100, 2),
                'CVaR (95%)':       round(cvar_val * 100, 6),
                'Port. Alpha':      round(ret, 4)
            })
            print(f"  ✓ {name:22s}  Sharpe: {sharpe:.4f}  CVaR: {cvar_val*100:.4f}%")
        else:
            print(f"  ✗ {name:22s}  INFEASIBLE")

    return cvar_weights, pd.DataFrame(summary_rows).set_index('Model')

def _get_quasi_diag(link):
    num_items = int(link[-1, 3])
    sort_ix   = pd.Series([int(link[-1, 0]), int(link[-1, 1])])

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i   = df0.index
        j   = df0.values - num_items
        sort_ix[i] = link[j, 0].astype(int)
        df2 = pd.Series(link[j, 1].astype(int), index=i + 1)
        sort_ix = pd.concat([sort_ix, df2]).sort_index()
        sort_ix.index = range(len(sort_ix))

    return sort_ix.tolist()


def _get_cluster_var(cov, c_items):
    cov_ = cov.loc[c_items, c_items]
    ivp  = 1.0 / np.diag(cov_)
    ivp /= ivp.sum()
    return float(ivp @ cov_.values @ ivp)


def _recursive_bisect(cov, sort_ix):
    w       = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]

    while c_items:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            g0 = c_items[i]
            g1 = c_items[i + 1]
            v0 = _get_cluster_var(cov, g0)
            v1 = _get_cluster_var(cov, g1)
            alpha = 1 - v0 / (v0 + v1)
            w[g0] *= alpha
            w[g1] *= (1 - alpha)

    return w


def calculate_hrp_weights(returns_df):
    cov  = returns_df.cov()
    corr = returns_df.corr()

    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    np.fill_diagonal(dist.values, 0)
    dist_condensed = ssd.squareform(np.maximum(dist.values, 0))

    linkage_mat = sch.linkage(dist_condensed, method='single')

    sort_ix  = _get_quasi_diag(linkage_mat)
    sort_ix  = corr.index[sort_ix].tolist()
    weights  = _recursive_bisect(cov, sort_ix)

    return weights.sort_index(), linkage_mat

def walk_forward_backtest(weights_map, actual_returns_df,
                           n_periods=7, rf=0.05):
    rf_daily = rf / 252
    splits   = np.array_split(np.arange(len(actual_returns_df)), n_periods)
    rows     = []

    for name, w in weights_map.items():
        port_rets = []
        for split in splits:
            period_rets = actual_returns_df.iloc[split].values
            port_rets.extend((period_rets @ w).tolist())

        port_rets  = np.array(port_rets)
        ann_ret    = float(np.mean(port_rets) * 252)
        ann_vol    = float(np.std(port_rets) * np.sqrt(252))
        sharpe     = (ann_ret - rf) / ann_vol if ann_vol > 0 else float('nan')
        cum        = np.cumprod(1 + port_rets)
        running_max = np.maximum.accumulate(cum)
        max_dd     = float(np.min((cum - running_max) / running_max) * 100)

        rows.append({
            'Model':            name,
            'Ann. Return (%)':  round(ann_ret * 100, 2),
            'Ann. Vol (%)':     round(ann_vol * 100, 2),
            'Ex-post Sharpe':   round(sharpe, 4),
            'Max Drawdown (%)': round(max_dd, 2),
            'Rebalances':       n_periods,
        })

    return pd.DataFrame(rows).set_index('Model')

def tail_risk_stats(weights_map, actual_fn, confidence=0.95):
    rows = []
    for name, w in weights_map.items():
        rets  = actual_fn(name).values @ w
        var_  = float(np.percentile(rets, (1 - confidence) * 100))
        cvar_ = float(rets[rets <= var_].mean()) if (rets <= var_).any() else var_
        rows.append({
            'Model':               name,
            'Realized VaR 95%':    round(var_  * 100, 4),
            'Realized CVaR 95%':   round(cvar_ * 100, 4),
            'Return Skewness':     round(float(pd.Series(rets).skew()), 4),
            'Excess Kurtosis':     round(float(pd.Series(rets).kurt()), 4),
            'Ann. Volatility (%)': round(float(np.std(rets) * np.sqrt(252) * 100), 4),
        })
    return pd.DataFrame(rows).set_index('Model')

def perturbed_sharpe(weights_map, actual_fn, sigma_noise=0.10, n_sim=200, rf=0.05):
    rows = []
    for name, w in weights_map.items():
        act  = actual_fn(name).values
        base_rets   = act @ w
        base_sharpe = (np.mean(base_rets) * 252 - rf) / (np.std(base_rets) * np.sqrt(252))

        noisy = []
        for _ in range(n_sim):
            noise   = np.random.normal(0, sigma_noise, size=w.shape)
            w_noisy = np.clip(w + noise, 0, None)
            if w_noisy.sum() > 0:
                w_noisy /= w_noisy.sum()
            r = act @ w_noisy
            s = (np.mean(r) * 252 - rf) / (np.std(r) * np.sqrt(252) + 1e-8)
            noisy.append(s)

        rows.append({
            'Model':           name,
            'Sharpe (σ=0)':    round(base_sharpe, 3),
            'Sharpe (σ=0.10)': round(float(np.mean(noisy)), 3),
            'Degradation':     round(base_sharpe - float(np.mean(noisy)), 3),
        })
    return pd.DataFrame(rows).set_index('Model')

def diversification_analysis(mvo_w_map, cvar_w_map):
    rows = []
    all_models = set(mvo_w_map) | set(cvar_w_map)
    for name in all_models:
        mw = mvo_w_map.get(name)
        cw = cvar_w_map.get(name)
        rows.append({
            'Model':              name,
            'MVO Eff. Assets':    round(1.0 / np.sum(mw**2), 2) if mw is not None else 0,
            'CVaR Eff. Assets':   round(1.0 / np.sum(cw**2), 2) if cw is not None else 0,
            'MVO Top-3 Weight':   round(float(np.sort(mw)[-3:].sum()), 4) if mw is not None else 0,
            'CVaR Top-3 Weight':  round(float(np.sort(cw)[-3:].sum()), 4) if cw is not None else 0,
        })
    return pd.DataFrame(rows).set_index('Model')
