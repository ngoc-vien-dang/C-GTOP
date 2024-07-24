import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
np.seterr(invalid='ignore')
import warnings
warnings.filterwarnings('ignore')

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1

    bin_true_probs = []
    bin_pred_probs = []
    for bin_index in range(n_bins):
        bin_indices_i = (bin_indices == bin_index)
        bin_count = bin_indices_i.sum()
        if bin_count > 0:
            bin_true_probs.append(y_true[bin_indices_i].mean())
            bin_pred_probs.append(y_prob[bin_indices_i].mean())
        else:
            bin_true_probs.append(0)
            bin_pred_probs.append(0)

    ece = np.abs(np.array(bin_true_probs) - np.array(bin_pred_probs)).mean()
    return ece

def bootstrap_ci(data, metric_func, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence intervals for a given metric function."""
    np.random.seed(42)  # Set a fixed seed for reproducibility

    bootstrap_samples = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_samples.append(metric_func(sample))

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    lower_bound = np.percentile(bootstrap_samples, lower_percentile)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile)

    return lower_bound, np.mean(bootstrap_samples), upper_bound

def process_baseline_data(baseline, proba_col, true_label_col, pred_label_col, method_name):
    # List of split DataFrames
    split_dfs = [baseline[(baseline['Split'] == i) & (baseline['Train_Val_Test'] == 'Test')] for i in range(5)]

    # Initialize an empty list for results
    results_df_list = []

    for i, df in enumerate(split_dfs):
        # Initialize a dictionary to store results
        results_dict = {}

        # Get unique age groups
        age_groups = df['age_group'].unique()

        # Initialize the results_dict with keys for each metric and columns for each age group
        for age_group in age_groups:
            results_dict[f'AUROC_{age_group}'] = []
            results_dict[f'BAcc_{age_group}'] = []
            results_dict[f'BCE_{age_group}'] = []
            results_dict[f'ECE_{age_group}'] = []
            results_dict[f'FPR_{age_group}'] = []
            results_dict[f'FNR_{age_group}'] = []

        # Calculate metrics for each age subgroup and store in the dictionary
        for age_group, group_data in df.groupby('age_group'):
            proba0 = group_data[proba_col]
            y_true = group_data[true_label_col]
            y_pred = group_data[pred_label_col]

            if y_true.isnull().any() or y_pred.isnull().any() or proba0.isnull().any():
                auc_roc = balanced_acc = bce = ece = fpr = fnr = np.nan
            else:
                try:
                    auc_roc = roc_auc_score(y_true, proba0)
                except ValueError:
                    auc_roc = np.nan
                try:
                    balanced_acc = balanced_accuracy_score(y_true, y_pred)
                except ValueError:
                    balanced_acc = np.nan
                try:
                    bce = log_loss(y_true, proba0)
                except ValueError:
                    bce = np.nan
                try:
                    ece = expected_calibration_error(y_true, proba0)
                except ValueError:
                    ece = np.nan

                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
                except ValueError:
                    fpr = fnr = np.nan

            results_dict[f'AUROC_{age_group}'].append(auc_roc)
            results_dict[f'BAcc_{age_group}'].append(balanced_acc)
            results_dict[f'BCE_{age_group}'].append(bce)
            results_dict[f'ECE_{age_group}'].append(ece)
            results_dict[f'FPR_{age_group}'].append(fpr)
            results_dict[f'FNR_{age_group}'].append(fnr)

        # Ensure each key has only one value in the list
        for key in results_dict:
            if len(results_dict[key]) == 1:
                results_dict[key] = results_dict[key][0]

        # Convert the results dictionary to a DataFrame and add to the results list
        split_results_df = pd.DataFrame([results_dict])
        split_results_df['split'] = i  # Add split index
        results_df_list.append(split_results_df)

    # Concatenate all results DataFrames
    baseline_df = pd.concat(results_df_list, ignore_index=True)
    # Add the method column
    baseline_df['method'] = method_name

    # Calculate Bootstrap CIs
    groups = ['<65', '65-74', '75-84', '85+']
    metrics = ['AUROC', 'BCE', 'ECE', 'BAcc', 'FPR', 'FNR']
    all_stats = []
    methods = baseline_df['method'].unique()

    for method in methods:
        for group in groups:
            for metric in metrics:
                column_name = f"{metric}_{group}"
                data = baseline_df[baseline_df['method'] == method][column_name].dropna()

                if len(data) > 0:
                    ci_lower, mean, ci_upper = bootstrap_ci(data, np.mean, n_bootstrap=100, ci=95)
                else:
                    ci_lower, mean, ci_upper = np.nan, np.nan, np.nan

                # Append stats for each quantile
                all_stats += [
                    {'metric': metric.upper(), 'grp_val': group, 'CI_quantile_95': 'lower', 'values': ci_lower, 'exp_name': method},
                    {'metric': metric.upper(), 'grp_val': group, 'CI_quantile_95': 'mid', 'values': mean, 'exp_name': method},
                    {'metric': metric.upper(), 'grp_val': group, 'CI_quantile_95': 'upper', 'values': ci_upper, 'exp_name': method},
                ]

    # Convert the all_stats list of dictionaries to a DataFrame
    stats_df = pd.DataFrame(all_stats)
    # Add eval_group column
    stats_df['eval_group'] = 'Age'
    # Final DataFrame
    stats_df_baseline = stats_df
    return stats_df_baseline

class smart_dict(dict):
    def __missing__(self, key):
        return key
grp_name_mapping = smart_dict({
    '<65': '<65',
    '65-74': '65-74',
    '75-84': '75-84',
    '85+': '85+'
})

def insert_missing_bins(fraction_of_positives, mean_predicted_value, n_bins):
    all_bins = np.linspace(0, 1, n_bins, endpoint=False) + 1 / (2 * n_bins)
    new_fraction_of_positives = np.full(n_bins, np.nan)
    new_mean_predicted_value = all_bins.copy()
    indices = np.clip(np.round(mean_predicted_value * n_bins).astype(int), 0, n_bins-1)
    new_fraction_of_positives[indices] = fraction_of_positives
    new_mean_predicted_value[indices] = mean_predicted_value
    return new_fraction_of_positives, new_mean_predicted_value

def count_samples_per_bin(y_preds, n_bins):
    bin_limits = np.linspace(0, 1, n_bins+1)
    sample_count, _ = np.histogram(y_preds, bins=bin_limits)
    return sample_count

# Define the combined plotting function
def plot_combined_figure(calibration_df, stats_df, true_column, pred_column, saved_path, n_bins=10):
    methods =['Baseline', 'CPP', 'C-GTOP']
    fig, axs = plt.subplots(nrows=3, ncols=len(methods), figsize=(10, 8), constrained_layout=False)
    titles = ['Baseline', 'CPP', 'C-GTOP']
    groups = ["<65", "65-74", "75-84", "85+"]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # First row: Calibration Curves
    #methods =['Baseline', 'CPP', 'C-GTOP']
    for idx, col in enumerate(methods):
        ax = axs[0, idx]
        method_df = calibration_df[calibration_df['method'] == col]

        for g, group in enumerate(groups):
            group_df = method_df[method_df['age_group'] == group]

            if group_df.empty:
                continue

            fractions_of_positives = []
            mean_predicted_values = []
            sample_counts = []

            model_splits = np.array_split(group_df, 5)
            for split in model_splits:
                if len(split) == 0:
                    continue

                y_test = split[true_column].values
                y_preds = split[pred_column].values

                fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_preds, n_bins=n_bins)
                fraction_of_positives, mean_predicted_value = insert_missing_bins(fraction_of_positives, mean_predicted_value, n_bins)
                sample_count = count_samples_per_bin(y_preds, n_bins)

                fractions_of_positives.append(fraction_of_positives)
                mean_predicted_values.append(mean_predicted_value)
                sample_counts.append(sample_count)

            mean_fraction_of_positives = np.nanmean(np.array(fractions_of_positives, dtype=float), axis=0)
            std_fraction_of_positives = np.nanstd(np.array(fractions_of_positives, dtype=float), axis=0)
            mean_predicted_value = np.nanmean(np.array(mean_predicted_values, dtype=float), axis=0)
            std_predicted_value = np.nanstd(np.array(mean_predicted_values, dtype=float), axis=0)
            mean_sample_count = np.nanmean(np.array(sample_counts, dtype=float), axis=0)

            non_nan_indices = np.logical_not(np.isnan(mean_fraction_of_positives))
            color = colors[g % len(colors)]
            ax.scatter(mean_predicted_value[non_nan_indices], mean_fraction_of_positives[non_nan_indices], s=10*mean_sample_count[non_nan_indices], color=color, alpha=0.7)
            ax.plot(mean_predicted_value[non_nan_indices], mean_fraction_of_positives[non_nan_indices], "-", color=color, label=f"{group}")
            lower_bound = np.maximum(mean_fraction_of_positives - std_fraction_of_positives, 0)
            upper_bound = mean_fraction_of_positives + std_fraction_of_positives
            ax.fill_between(mean_predicted_value[non_nan_indices], lower_bound[non_nan_indices], upper_bound[non_nan_indices], alpha=0.2, color=color)

        ax.set_title(titles[idx], fontsize=12)
        ax.plot([0, 1], [0, 1], "k:")
        ax.set(xlabel='Mean Predicted AD Probability')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect(0.75)

        if idx == 0:
            ax.set(ylabel='Fraction of True AD')
    methods = ['C-GTOP', 'CPP', 'Baseline']
    # Second row: AUROC, BCE, ECE
    for d, t in enumerate(['Age']):
        grps = list({"Age": ['<65', '65-74', '75-84', '85+']}[t])
        method_mapping_int = {i: c for c, i in enumerate(methods)}
        offsets = np.flip(np.linspace(-0.2, 0.2, len(grps)))

        for c, m in enumerate(['AUROC', 'BCE', 'ECE']):
            ax = axs[1, c] if len(['Age']) == 1 else axs[1, c]
            subset = stats_df[(stats_df.metric == m) & (stats_df.eval_group == t)]
            for e, grp in enumerate(grps):
                subset2 = subset[subset['grp_val'] == grp]
                if not len(subset2):
                    continue
                subset2 = subset2.copy()
                subset2.loc[:, 'x'] = subset2['exp_name'].map(method_mapping_int) + offsets[e]
                subset2 = subset2.sort_values(by='x', ascending=True)
                subset2 = subset2[['CI_quantile_95', 'values', 'x']].pivot_table(values='values', index='x', columns='CI_quantile_95', aggfunc=lambda x: x)
                subset2['lower_err'] = subset2['mid'] - subset2['lower']
                subset2['upper_err'] = subset2['upper'] - subset2['mid']
                yerrs = subset2[['lower_err', 'upper_err']].values.T

                ax.errorbar(y=subset2.index.values, x=subset2['mid'].values, fmt='o', xerr=yerrs, color=colors[e], markersize=5, elinewidth=3, capsize=4, capthick=2)

            ax.set_yticks(range(len(methods)))
            if c == 0:
                ax.set_yticklabels(methods)
            else:
                ax.set_yticklabels([])

            if d == 0:
                ax.set_title(m)

            for f in np.arange(0, len(methods) - 1):
                ax.axhline(f + 0.5, linestyle='--', color='k', linewidth=0.3)

    # Third row: BACC, FPR, FNR
    for d, t in enumerate(['Age']):
        grps = list({"Age": ['<65', '65-74', '75-84', '85+']}[t])
        method_mapping_int = {i: c for c, i in enumerate(methods)}
        offsets = np.flip(np.linspace(-0.2, 0.2, len(grps)))

        for c, m in enumerate(['BACC', 'FPR', 'FNR']):
            ax = axs[2, c] if len(['Age']) == 1 else axs[2, c]
            subset = stats_df[(stats_df.metric == m) & (stats_df.eval_group == t)]
            for e, grp in enumerate(grps):
                subset2 = subset[subset['grp_val'] == grp]
                if not len(subset2):
                    continue
                subset2 = subset2.copy()
                subset2.loc[:, 'x'] = subset2['exp_name'].map(method_mapping_int) + offsets[e]
                subset2 = subset2.sort_values(by='x', ascending=True)
                subset2 = subset2[['CI_quantile_95', 'values', 'x']].pivot_table(values='values', index='x', columns='CI_quantile_95', aggfunc=lambda x: x)
                subset2['lower_err'] = subset2['mid'] - subset2['lower']
                subset2['upper_err'] = subset2['upper'] - subset2['mid']
                yerrs = subset2[['lower_err', 'upper_err']].values.T

                ax.errorbar(y=subset2.index.values, x=subset2['mid'].values, fmt='o', xerr=yerrs, color=colors[e], markersize=5, elinewidth=3, capsize=4, capthick=2)

            ax.set_yticks(range(len(methods)))
            if c == 0:
                ax.set_yticklabels(methods)
            else:
                ax.set_yticklabels([])

            if d == 0:
                ax.set_title(m)

            for f in np.arange(0, len(methods) - 1):
                ax.axhline(f + 0.5, linestyle='--', color='k', linewidth=0.3)

    # Add a single legend to the right side
    legend = [
        mlines.Line2D([], [], linestyle='-', color=colors[c], marker='o',
                      label=grp_name_mapping[grp], markersize=12, linewidth=3) for c, grp in list(zip(range(len(grps)), grps))
    ]

    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.52, -0.03), ncol=4, frameon=False)
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(wspace=0.2, hspace=0.4)  # Adjust the space between subplots
    plt.savefig(saved_path, bbox_inches='tight')
    plt.show()