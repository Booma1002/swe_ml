import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, hamming_loss, precision_recall_curve, \
    average_precision_score
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, hamming_loss
import pandas as pd


class Visualizer:
    @staticmethod
    def plot_grid(classifiers, plot_func, X_test, y_test, shape=(1, 1), figsize=(15, 10), title=None, save_path=None):
        fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
        if title: fig.suptitle(title, fontsize=20, fontweight='bold')
        axes_flat = axes.flatten() if shape[0] * shape[1] > 1 else [axes]

        for (name, clf), ax in zip(classifiers.items(), axes_flat):
            if plot_func == plot_training_history:
                plot_func(clf, ax=ax, title=f"{name} History")
            elif plot_func == conf:
                y_pred = clf.predict(X_test)
                plot_func(clf, y_test, y_pred, ax=ax)
                ax.set_title(f"{name} Confusion Matrix", fontsize=16)
            elif plot_func == ROC or plot_func == plot_precision_recall:
                plot_func(clf, y_test, X_test, ax=ax)
                ax.set_title(f"{name} {plot_func.__name__}", fontsize=16)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()


def plot_training_history(clf, scope=50, title="Model Training History", ax_loss=None):
    if ax_loss is None:
        fig, ax_loss = plt.subplots(figsize=(12, 7))
        should_show = True
    else:
        should_show = False

    def get_series(attribute_name):
        if not hasattr(clf, attribute_name): return None
        raw_data = getattr(clf, attribute_name)
        if not raw_data or len(raw_data) == 0: return None
        clean_data = [x.item() if hasattr(x, 'item') else x for x in raw_data]
        y_full = np.array(clean_data)
        idx = np.arange(0, len(y_full), scope)
        if idx[-1] != len(y_full) - 1: idx = np.append(idx, len(y_full) - 1)
        return idx + 1, y_full[idx]

    ax_score = ax_loss.twinx()

    if title:
        ax_loss.set_title(title, fontsize=16, fontweight='bold')

    lines = []
    res = get_series('losses_') or get_series('loss_curve_')
    if res:
        x, y = res
        ln, = ax_loss.plot(x, y, color='crimson', lw=4, linestyle='-', alpha=0.8, label='Train Loss')
        lines.append(ln)

    res = get_series('val_losses_')
    if res:
        x, y = res
        if hasattr(clf, 'val_jump'): x = x * clf.val_jump
        min_val = np.min(y)
        ln, = ax_loss.plot(x, y, color='red', lw=4, linestyle='--', marker='o', markersize=7,
                           label=f'Val Loss (Min: {min_val:.4f})')
        lines.append(ln)

    res = get_series('scores_')
    if res:
        x, y = res
        ln, = ax_score.plot(x, y, color='dodgerblue', linestyle='-', linewidth=4, alpha=0.6, label='Train Score')
        lines.append(ln)

    res = get_series('val_scores_') or get_series('validation_scores_')
    if res:
        x, y = res
        if hasattr(clf, 'val_jump'): x = x * clf.val_jump
        max_val = np.max(y)
        ln, = ax_score.plot(x, y, color='navy', linewidth=4, linestyle='--', marker='s', markersize=7,
                            label=f'Val Score (Max: {max_val:.4f})')
        lines.append(ln)

    if hasattr(clf, "best_epoch_") and clf.best_epoch_:
        vline = ax_loss.axvline(clf.best_epoch_, color='green', linestyle='-.', linewidth=2,
                                label=f'Best Epoch: {clf.best_epoch_}')
        lines.append(vline)

    ax_loss.set_xlabel("Epoch", fontsize=14)
    ax_loss.tick_params(axis='x', labelsize=14)
    ax_loss.set_ylabel("Loss (Cross Entropy)", fontsize=16, color='crimson')
    ax_loss.tick_params(axis='y', labelcolor='crimson', labelsize=14)
    ax_loss.grid(True, linestyle=':', alpha=0.6)

    ax_score.set_ylabel("Score", fontsize=16, color='navy')
    ax_score.tick_params(axis='y', labelcolor='navy', labelsize=14)
    ax_score.set_ylim(0, 1.05)

    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc="right", frameon=True, shadow=True)

    if should_show:
        plt.tight_layout()
        plt.show()

def conf(clf, y_test, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        should_show = True
    else:
        should_show = False

    if hasattr(clf, 'classes_'):
        classes = clf.classes_
    else:
        classes = np.unique(y_test)

    cm = confusion_matrix(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    specs = []
    n_classes = y_test_bin.shape[1]
    if n_classes == 1 and len(classes) == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specs.append(tn / (tn + fp))
    else:
        for i in range(n_classes):
            tn, fp, fn, tp = confusion_matrix(y_test_bin[:, i], y_pred_bin[:, i]).ravel()
            specs.append(tn / (tn + fp))

    spec = np.mean(specs)
    hamming = hamming_loss(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', cbar=False, ax=ax)

    ax.set_title(f'Confusion Matrix\nMacro-Avg Specificity: [{spec:.4f}]  -  Hamming-loss: [{hamming:.4f}]',
                 fontsize=18)
    ax.set_ylabel('True Label', fontsize=14, color='crimson')
    ax.set_xlabel('Predicted Label', fontsize=14, color='navy')
    ax.tick_params(axis='y', labelcolor='crimson', labelsize=14)
    ax.tick_params(axis='x', labelcolor='navy', labelsize=14)

    if should_show:
        plt.show()


def ROC(clf, y_test, X_test_proc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        should_show = True
    else:
        should_show = False

    y_test_bin = label_binarize(y_test, classes=clf.classes_)
    y_score = clf.predict_proba(X_test_proc)
    n_classes = y_test_bin.shape[1]

    if n_classes == 1:
        n_classes = 2
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    colors = plt.get_cmap('coolwarm', n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors(i), lw=1, label=f'Class {i} (AUC = {roc_auc:.6f})')

    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, 1.0)
    ax.plot([0, 1], [0, 1], 'g--', lw=2)
    ax.set_title('ROC Curve', fontsize=20)
    ax.set_xlabel('False Positive Rate', fontsize=18, color='crimson')
    ax.set_ylabel('True Positive Rate', fontsize=18, color='navy')
    ax.tick_params(axis='x', labelcolor='crimson', labelsize=14)
    ax.tick_params(axis='y', labelcolor='navy', labelsize=14)
    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=8)
    ax.grid(alpha=0.4, axis="both", color='k', lw=2)

    if should_show:
        plt.show()


def plot_precision_recall(clf, y_test, X_test_proc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        should_show = True
    else:
        should_show = False

    y_test_bin = label_binarize(y_test, classes=clf.classes_)
    y_score = clf.predict_proba(X_test_proc)
    n_classes = y_test_bin.shape[1]

    if n_classes == 1:
        n_classes = 2
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    colors = plt.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        prec = average_precision_score(y_test_bin[:, i], y_score[:, i])

        ax.plot(recall, precision, color=colors(i), lw=2,
                label=f'Class {i} (AP = {prec:.4f})')

    ax.set_xlabel('Recall', fontsize=24)
    ax.set_ylabel('Precision', fontsize=24)
    ax.set_title('Precision-Recall Curve', fontsize=18)
    ax.tick_params(labelsize=12)
    ax.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    ax.grid(alpha=0.8)

    if should_show:
        plt.tight_layout()
        plt.show()

def plot_benchmark_metrics(classifiers, X_test, y_test, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        should_show = True
    else:
        fig= ax.get_figure()
        should_show = False

    metrics_names = ['Precision', 'Recall', 'F1 Score', 'MCC', 'Hamming Loss', 'Specificity']
    results = {name: [] for name in classifiers.keys()}
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)


        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        hamming = hamming_loss(y_test, y_pred)
        classes = np.unique(y_test)
        specs = []
        y_test_bin = label_binarize(y_test, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)

        for i in range(len(classes)):
            if y_test_bin.shape[1] == 1:
                t_true, t_pred = y_test_bin.ravel(), y_pred_bin.ravel()
            else:
                t_true, t_pred = y_test_bin[:, i], y_pred_bin[:, i]
            tn, fp, fn, tp = confusion_matrix(t_true, t_pred).ravel()
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        spec = np.mean(specs)
        results[name] = [prec, rec, f1, mcc, hamming, spec]

    x = np.arange(len(metrics_names))
    width = 0.35
    multiplier = 0

    for attribute, measurement in results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=10, fontweight='bold')
        multiplier += 1

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Benchmark Metrics Comparison', fontsize=18, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(loc='lower left', fontsize=12, frameon=True, shadow=True)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    if should_show:
        plt.tight_layout()
        plt.show()


def plot_clinical_forecast(history_dates, history_burnout, future_dates, actual_future, ridge_preds, arimax_preds,
                           conf_int, patient_id):
    """Renders the clinical 25-day horizon forecast for a specific patient."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))

    plt.plot(history_dates, history_burnout, label='Historical Burnout (Train)', color='black', linewidth=2, marker='o')

    plt.plot(future_dates, actual_future, label='Actual Future Burnout (Test)', color='green', linewidth=2,
             linestyle='--', marker='s')

    plt.plot(future_dates, ridge_preds, label=f'Ridge Baseline Forecast', color='red', linewidth=2, linestyle='-.')

    plt.plot(future_dates, arimax_preds, label=f'Tuned ARIMAX Forecast', color='blue', linewidth=2)

    plt.fill_between(future_dates,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='blue', alpha=0.15, label='ARIMAX 95% Confidence Interval')

    plt.axvline(x=history_dates.iloc[-1], color='gray', linestyle=':', linewidth=2, label='Forecast Horizon Start')

    plt.title(f'Clinical Forecasting: 25-Day Burnout Trajectory for Student {patient_id}', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Burnout Index (1-10)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(1, 10)

    plt.tight_layout()
    plt.show()



def plot_vif_bars(vif_input, threshold=5.0, cmap_name="viridis", figsize=(10, 6),
                  title="Variance Inflation Factor (VIF) by Feature",
                  font_family=("DejaVu Sans", "Arial"),
                  input_name_color="black", input_val_color="black"):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(font_family),
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })

    if isinstance(vif_input, pd.DataFrame):
        vif_df = vif_input.copy()
    else:
        vif_df = pd.DataFrame(vif_input)

    if "Feature" not in vif_df.columns or "VIF_Score" not in vif_df.columns:
        if vif_df.shape[1] == 2:
            vif_df.columns = ["Feature", "VIF_Score"]
        else:
            raise ValueError("vif_input must be a DataFrame with columns ['Feature','VIF_Score'].")

    vif_plot = vif_df[vif_df["Feature"] != "const"].sort_values("VIF_Score", ascending=True).reset_index(drop=True)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.15, 0.85, len(vif_plot))) if len(vif_plot) > 0 else []

    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.5 * len(vif_plot))))
    bars = ax.barh(range(len(vif_plot)), vif_plot["VIF_Score"], color=colors, edgecolor="k", linewidth=0.4)
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=3.5, label=f"VIF Threshold = {threshold}")

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("VIF Score (Lower is better)", fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')

    max_v = vif_plot["VIF_Score"].max() if len(vif_plot) > 0 else 1.0
    left_text_x = 0.02 * max_v

    for i, (bar, feat, v) in enumerate(zip(bars, vif_plot["Feature"], vif_plot["VIF_Score"])):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        clean_feat = str(feat).replace('_', ' ').title()

        ax.text(left_text_x, y, clean_feat, va="center", ha="left", color=input_name_color, fontsize=14,
                weight="semibold")
        ax.text(w - 0.02 * max_v, y, f"{v:.2f}", va="center", ha="right", color=input_val_color, fontsize=14,
                weight="bold")

    ax.set_ylim(-0.5, len(vif_plot) - 0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_feature_dist(data):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=data, x='Z-Score', hue='Feature', fill=True,
        common_norm=False, alpha=0.3, palette='viridis', linewidth=2
    )
    plt.title("Engine 4: Standardized Feature Distributions (Latent Priors)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Z-Score (Standard Deviation from Mean)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.xlim(-4, 4)
    plt.grid(True, alpha=0.2)
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1), title="Behavioral Metrics")
    plt.tight_layout()
    plt.show()


def plot_champion_leaderboard(ledger_df, cmap_name="viridis", figsize=(10, 6),
                              title="Grand Champion Leaderboard (TPE Optimized)",
                              font_family=("DejaVu Sans", "Arial"),
                              input_name_color="black", input_val_color="black"):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(font_family),
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })
    plot_df = ledger_df.sort_values(by='Best Silhouette (↑)', ascending=True).reset_index(drop=True)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.15, 0.85, len(plot_df))) if len(plot_df) > 0 else []

    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], 0.5 * len(plot_df))))
    bars = ax.barh(range(len(plot_df)), plot_df['Best Silhouette (↑)'], color=colors, edgecolor="k", linewidth=0.4)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Maximized Silhouette Score (Approaching 1.0 is Better)", fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')

    max_v = plot_df['Best Silhouette (↑)'].max() if len(plot_df) > 0 else 1.0
    left_text_x = 0.02 * max_v

    champion_name = plot_df.iloc[-1]['Algorithm']

    for i, (bar, algo, score, k_val, db_score) in enumerate(
            zip(bars, plot_df['Algorithm'], plot_df['Best Silhouette (↑)'], plot_df['Clusters Formed (K)'],
                plot_df['Davies-Bouldin (↓)'])):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        ax.text(left_text_x, y, str(algo), va="center", ha="left", color=input_name_color, fontsize=16,
                weight="semibold")

        metrics_text = f"S: {score:.4f}  |  K: {k_val}  |  DB: {db_score:.2f}"
        ax.text(w - 0.02 * max_v, y, metrics_text, va="center", ha="right", color=input_val_color, fontsize=16,
                weight="bold")

    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    plt.tight_layout()
    plt.show()

    return champion_name


def plot_cohort_comparison(profiles_df, title="Engine 4: Behavioral Centroids (Raw Domain)",
                           font_family=("DejaVu Sans", "Arial")):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(font_family),
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })

    # Normalize row-wise so bar lengths are relative (0.0 to 1.0)
    # This solves the issue of plotting 8000 XP next to 0.90 Attendance
    normalized_df = profiles_df.div(profiles_df.max(axis=1), axis=0)

    metrics = profiles_df.index.tolist()
    cohorts = profiles_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    y = np.arange(len(metrics))

    # Dynamic color mapping: Green for High Performers, Crimson for At-Risk
    color_map = {cohorts[0]: '#2ca02c', cohorts[1]: '#d62728'}

    for i, cohort in enumerate(cohorts):
        bar_lengths = normalized_df[cohort]
        raw_values = profiles_df[cohort]

        bars = ax.barh(y - bar_width / 2 + i * bar_width, bar_lengths, bar_width,
                       label=cohort, color=color_map[cohort], edgecolor='black', linewidth=1.2)

        # Annotate with the exact RAW values needed for the database
        for bar, raw_val, metric in zip(bars, raw_values, metrics):
            width = bar.get_width()

            # Smart formatting based on the metric type
            if 'xp' in metric.lower():
                text_val = f"{int(raw_val):,} XP"
            elif 'burnout' in metric.lower():
                text_val = f"{raw_val:.1f} / 10"
            else:
                text_val = f"{raw_val * 100:.1f}%"

            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, text_val,
                    va='center', ha='left', color='black', fontsize=13, weight='bold')

    ax.set_yticks(y)
    clean_metrics = [m.replace('_', ' ').title() for m in metrics]
    ax.set_yticklabels(clean_metrics, weight='bold', color='#333333')

    ax.set_xlim(0, 1.25)  # Buffer for text labels
    ax.set_xticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_streaming_benchmark(benchmark_df, title="Streaming Architectures: Topology vs. Latency",
                             font_family=("DejaVu Sans", "Arial")):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(font_family),
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })

    plot_df = benchmark_df.sort_values(by='Silhouette Score', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.2, 0.8, len(plot_df)))

    bars = ax.barh(range(len(plot_df)), plot_df['Silhouette Score'], color=colors, edgecolor="k", linewidth=1.2)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Topological Integrity (Maximized Silhouette Score)", fontsize=14, labelpad=15)
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')

    max_v = plot_df['Silhouette Score'].max()
    left_text_x = 0.02 * max_v

    for i, (bar, algo, s_score, db_score, latency) in enumerate(
            zip(bars, plot_df['Algorithm'], plot_df['Silhouette Score'], plot_df['Davies-Bouldin'],
                plot_df['Latency (ms)'])):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        ax.text(left_text_x, y, str(algo), va="center", ha="left", color="black", fontsize=15, weight="semibold")
        metrics_text = f"S: {s_score:.4f}  |  DB: {db_score:.2f}  |  Latency: {latency:.1f} ms"
        text_color = "black" if w < (0.85 * max_v) else "white"

        ax.text(w - 0.015 * max_v, y, metrics_text, va="center", ha="right", color=text_color, fontsize=14,
                weight="bold")

    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    plt.tight_layout()
    plt.show()

def plot_cohort_centroids(profiles_df, title="Behavioral Centroids (Raw Domain Extraction)",
                          font_family=("DejaVu Sans", "Arial")):
    if profiles_df.empty:
        print("[!] Execution Halted: Input dataframe is empty.")
        return

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(font_family),
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
    })

    row_max = profiles_df.max(axis=1)
    normalized_df = profiles_df.div(row_max.replace(0, 1), axis=0)

    metrics = profiles_df.index.tolist()
    cohorts = profiles_df.columns.tolist()
    n_cohorts = len(cohorts)
    n_metrics = len(metrics)

    fig_height = max(5, 1.2 * n_metrics + 0.5 * n_cohorts)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    total_group_width = 0.8
    bar_height = total_group_width / n_cohorts
    y_indices = np.arange(n_metrics)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.8, 0.2, n_cohorts))

    for i, cohort in enumerate(cohorts):
        bar_lengths = normalized_df[cohort]
        raw_values = profiles_df[cohort]
        offset = (i - (n_cohorts - 1) / 2) * bar_height
        bars = ax.barh(y_indices + offset, bar_lengths, bar_height,
                       label=cohort, color=colors[i], edgecolor='black', linewidth=0.8)
        for bar, raw_val, metric in zip(bars, raw_values, metrics):
            width = bar.get_width()

            if 'xp' in metric.lower():
                text_val = f"{int(raw_val):,} XP"
            elif 'burnout' in metric.lower():
                text_val = f"{raw_val:.1f} / 10"
            else:
                text_val = f"{raw_val * 100:.1f}%"
            text_x = width + 0.02
            ax.text(text_x, bar.get_y() + bar.get_height() / 2, text_val,
                    va='center', ha='left', color='black', fontsize=12, weight='bold')

    ax.set_yticks(y_indices)
    clean_metrics = [m.replace('_', ' ').title() for m in metrics]
    ax.set_yticklabels(clean_metrics, weight='bold', color='#333333')
    ax.set_xlim(0, 1.4)
    ax.set_xticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_title(title, fontweight='bold', pad=30, fontsize=18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(3, n_cohorts), frameon=True, fontsize=11, title="Discovered Tiers")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_xgbranker(data):
    df_importance = data
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_importance['Feature'], df_importance['Information Gain'],
                   color='#2ca02c', edgecolor='black', linewidth=1.2)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                va='center', ha='left', fontsize=11, fontweight='bold')

    ax.set_title("XGBRanker Feature Contribution (Pairwise Loss Reduction)", fontweight='bold', pad=20)
    ax.set_xlabel("Relative Information Gain", fontweight='bold')
    ax.set_ylabel("")
    ax.set_xlim(0, df_importance['Information Gain'].max() * 1.15)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_triage(data, sample_session_id):
    # 6. Render the Native Matplotlib Table Architecture
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')
    top_5_df = data
    table = ax.table(cellText=top_5_df.values,
                     colLabels=top_5_df.columns,
                     cellLoc='center',
                     loc='center')

    # Structural Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Conditional Gradient Application
    cmap = plt.get_cmap('viridis')
    min_util = top_5_df['Predicted_Utility'].min()
    max_util = top_5_df['Predicted_Utility'].max()
    range_util = max_util - min_util if max_util != min_util else 1.0

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header formatting
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2b2b2b')
        else:
            # Highlight True Engagement (Ground Truth Positive)
            if col == 6 and top_5_df.iloc[row - 1, col] == 1:
                cell.set_facecolor('#b5e7a0')  # Mint Green indicator
                cell.set_text_props(weight='bold')

            # Heatmap Gradient for Predicted Utility
            elif col == 5:
                val = top_5_df.iloc[row - 1, col]
                norm_val = (val - min_util) / range_util
                rgba = cmap(norm_val)
                cell.set_facecolor(rgba)

                # Dynamic text color to maintain contrast against the Viridis gradient
                text_color = 'white' if norm_val < 0.6 else 'black'
                cell.set_text_props(color=text_color, weight='bold')

    ax.set_title(f"Next-Best-Action Triage Engine (Session QID: {sample_session_id})",
                 fontweight='bold', fontsize=14, pad=15)

    plt.tight_layout()
    plt.show()