import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.cluster.hierarchy as sch

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.titleweight":  "bold",
    "figure.facecolor":  "#F8F9FA",
    "axes.facecolor":    "#FFFFFF",
})

PALETTE_BLUE  = "#2C3E50"
PALETTE_RED   = "#E74C3C"
PALETTE_GREEN = "#27AE60"
PALETTE_GOLD  = "#F39C12"

def plot_prediction_quality(eval_df, save_path="1_prediction_quality.png"):
    #Bar chart comparing Val IC and Test IC across all models.
    models  = eval_df.index
    val_ic  = eval_df['Val IC']
    test_ic = eval_df['Test IC']
    x, w    = np.arange(len(models)), 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, val_ic,  w, label='Val IC',  color='#4C72B0')
    ax.bar(x + w / 2, test_ic, w, label='Test IC', color='#DD8452')
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')

    ax.set_ylabel('Information Coefficient (IC)')
    ax.set_title('Model Prediction Quality: Validation vs. Test IC')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved → {save_path}")
#sharpe comparison
def plot_sharpe_comparison(comparison_df, gspc_sharpe,
                            save_path="2_sharpe_comparison.png"):
    """Side-by-side MVO vs CVaR Sharpe bars with S&P 500 reference line."""
    models = comparison_df.index
    x, w   = np.arange(len(models)), 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, comparison_df['MVO Sharpe'],  w, label='MVO Sharpe',  color=PALETTE_GREEN)
    ax.bar(x + w / 2, comparison_df['CVaR Sharpe'], w, label='CVaR Sharpe', color=PALETTE_RED)
    ax.axhline(gspc_sharpe, color='gray', linestyle='-.', linewidth=2,
               label=f'S&P 500 ({gspc_sharpe:.2f})')

    ax.set_ylabel('Ex-post Sharpe Ratio')
    ax.set_title('Risk-Adjusted Walk-Forward Performance (Sharpe)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved → {save_path}")
# risk vs reward scatter
def plot_risk_reward(wf_df, wf_cvar_df, gspc_ann_vol, gspc_ann_ret,
                     save_path="3_risk_reward_scatter.png"):
    """Volatility / Return scatter for MVO and CVaR portfolios."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        x='Ann. Vol (%)', y='Ann. Return (%)', data=wf_df,
        s=200, marker='o', label='MVO', color=PALETTE_GREEN,
        edgecolor='black', ax=ax
    )
    sns.scatterplot(
        x='Ann. Vol (%)', y='Ann. Return (%)', data=wf_cvar_df,
        s=200, marker='s', label='CVaR', color=PALETTE_RED,
        edgecolor='black', ax=ax
    )
    ax.scatter(
        gspc_ann_vol * 100, gspc_ann_ret * 100,
        color='gold', s=300, marker='*', edgecolor='black', label='S&P 500'
    )

    for i, model in enumerate(wf_df.index):
        ax.annotate(model, (wf_df['Ann. Vol (%)'].iloc[i], wf_df['Ann. Return (%)'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.set_title('Risk vs. Reward: Walk-Forward Performance')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved → {save_path}")


#robustness linkage
def plot_robustness_linkage(linkage_df, rho_mvo, p_mvo,
                             save_path="4_robustness_linkage.png"):
    """Scatter + regression line: Test IC vs MVO Sharpe degradation."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.regplot(
        x='Test IC', y='MVO Degradation', data=linkage_df, ax=ax,
        scatter_kws={'s': 150, 'color': '#4C72B0', 'edgecolor': 'black'},
        line_kws={'color': PALETTE_RED, 'linestyle': '--'}
    )
    for i, model in enumerate(linkage_df.index):
        ax.annotate(
            model,
            (linkage_df['Test IC'].iloc[i], linkage_df['MVO Degradation'].iloc[i]),
            textcoords="offset points", xytext=(10, 0), ha='left', fontsize=9
        )

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Test Information Coefficient (IC)')
    ax.set_ylabel('MVO Degradation (Sharpe Drop)')
    ax.set_title('Robustness Paradox: Test IC vs. Noise Degradation')
    ax.text(
        0.05, 0.05,
        f'Spearman ρ: {rho_mvo:.3f}\n(p-value: {p_mvo:.3f})',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved → {save_path}")

# allocation dashboard
def _plot_donut(ax, weights, title, colormap):
    top    = weights.sort_values(ascending=False).head(4)
    others = max(0.0, 1.0 - top.sum())
    vals   = list(top.values) + ([others] if others > 0.001 else [])
    labels = list(top.index)  + (['Others'] if others > 0.001 else [])
    colors = sns.color_palette(colormap, len(vals))

    _, _, autotexts = ax.pie(
        vals, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.4},
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10)
    ax.set_title(title, fontsize=13, fontweight='bold')


def plot_allocation_dashboard(model_name, tickers, mvo_w, cvar_w,
                               save_path="5_allocation_dashboard.png"):
    mvo_s  = pd.Series(mvo_w,  index=tickers)
    cvar_s = pd.Series(cvar_w, index=tickers)
    df     = pd.DataFrame({'MVO': mvo_s, 'CVaR': cvar_s}).sort_values('MVO', ascending=False)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Portfolio Allocation Comparison: {model_name}',
                 fontsize=20, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, :])
    x, w = np.arange(len(df)), 0.35
    ax1.bar(x - w / 2, df['MVO']  * 100, width=w, label='MVO',  color=PALETTE_BLUE,  alpha=0.9)
    ax1.bar(x + w / 2, df['CVaR'] * 100, width=w, label='CVaR', color=PALETTE_RED,   alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=0, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Capital Allocation (%)', fontsize=12)
    ax1.set_title('Absolute Weight Distribution', fontsize=15)
    ax1.legend(fontsize=12)

    ax2    = fig.add_subplot(gs[1, 0])
    diff   = (cvar_s - mvo_s).sort_values() * 100
    colors = [PALETTE_RED if v < 0 else PALETTE_GREEN for v in diff.values]
    bars   = ax2.bar(diff.index, diff.values, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=1.5)
    ax2.set_xticklabels(diff.index, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Weight Change (%)', fontsize=12)
    ax2.set_title('Allocation Shift (CVaR vs. MVO)', fontsize=15)
    for bar in bars:
        yv = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yv + (0.5 if yv >= 0 else -1.5),
                 f"{yv:+.1f}%", ha='center', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    gs_d   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.1)
    ax_p1  = fig.add_subplot(gs_d[0, 0])
    ax_p2  = fig.add_subplot(gs_d[0, 1])
    _plot_donut(ax_p1, mvo_s,  'MVO Concentration',  'Blues_r')
    _plot_donut(ax_p2, cvar_s, 'CVaR Concentration', 'Reds_r')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved → {save_path}")

#hrp comparison
def plot_hrp_comparison(linkage_matrix, tickers, df_compare,
                         target_model, save_path="6_hrp_comparison.png"):
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor('#F8F9FA')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#FFFFFF')
    sch.dendrogram(linkage_matrix, labels=tickers, ax=ax1,
                   leaf_rotation=90, leaf_font_size=12, color_threshold=0.5)
    ax1.set_title('HRP: Asset Hierarchical Clustering', fontsize=15)
    ax1.set_ylabel('Distance (Correlation-based)', fontsize=12)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#FFFFFF')
    df_compare.plot(
        kind='bar', ax=ax2,
        color=[PALETTE_BLUE, PALETTE_RED, PALETTE_GREEN],
        width=0.75, edgecolor='none', alpha=0.9
    )
    ax2.set_title(f'Strategy Comparison: MVO vs CVaR vs HRP ({target_model})', fontsize=15)
    ax2.set_ylabel('Capital Allocation', fontsize=12)
    ax2.set_xticklabels(df_compare.index, rotation=0, fontweight='bold', fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='x', alpha=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved → {save_path}")
