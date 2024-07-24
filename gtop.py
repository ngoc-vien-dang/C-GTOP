import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import gaussian_kde, genextreme as gev
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class OptimalThresholdFinder:
    def __init__(self, n_components=3):
        self.n_components = n_components

    # Sigmoid function to convert logit to probability
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Function to fit KDE to data
    def fit_kde(data):
        kde = gaussian_kde(data)
        return kde

    # Function to fit GMM to data
    def fit_gmm(self, data):
        gmm = GaussianMixture(n_components=self.n_components)
        gmm.fit(data.reshape(-1, 1))
        return gmm

    # Function to fit EVT to data
    def fit_evt(data):
        params = gev.fit(data)
        return params

    # Function to fit gamma distributions
    def fit_gamma(data):
        params = stats.gamma.fit(data, floc=0)
        return params
        
    # Function to fit normal distributions
    def fit_norm(data):
        params = stats.norm.fit(data)
        return params

    # Function to fit Student’s t distributions
    def fit_t(data):
        params = stats.t.fit(data)
        return params

    # Function to fit the best distribution to data using log-likelihood
    def fit_distribution(self, data):
        best_fit_name = None
        best_fit_params = None
        best_log_likelihood = -np.inf
        distributions = ['kde', 'gmm', 'gamma', 'norm', 't']
        
        for distribution in distributions:
            if distribution == 'kde':
                kde = self.fit_kde(data)
                log_likelihood = np.sum(np.log(kde.evaluate(data)))
                params = kde
            elif distribution == 'gmm':
                gmm = self.fit_gmm(data)
                log_likelihood = np.sum(gmm.score_samples(data.reshape(-1, 1)))
                params = gmm
            elif distribution == 'gamma':
                params = self.fit_gamma(data)
                log_likelihood = np.sum(stats.gamma.logpdf(data, *params))
            elif distribution == 'norm':
                params = self.fit_norm(data)
                log_likelihood = np.sum(stats.norm.logpdf(data, *params))
            elif distribution == 't':
                params = self.fit_t(data)
                log_likelihood = np.sum(stats.t.logpdf(data, *params))
            else:
                continue

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_fit_name = distribution
                best_fit_params = params

        return best_fit_name, best_fit_params

    # Function to get the cumulative distribution function (CDF) of the fitted distribution
    def get_distribution_cdf(name, params):
        if name == 'kde':
            return lambda x: OptimalThresholdFinder.kde_cdf(params, x)
        elif name == 'gmm':
            return lambda x: OptimalThresholdFinder.gmm_cdf(params, x)
        elif name == 'gamma':
            return lambda x: stats.gamma.cdf(x, *params)
        elif name == 'norm':
            return lambda x: stats.norm.cdf(x, *params)
        elif name == 't':
            return lambda x: stats.t.cdf(x, *params)
        else:
            raise ValueError("Unsupported distribution")

    # Function to get the CDF from KDE
    def kde_cdf(kde, x):
        return kde.integrate_box_1d(-np.inf, x)

    # Function to get the CDF from GMM
    def gmm_cdf(gmm, x):
        weights = gmm.weights_
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        cdf = 0
        for weight, mean, covariance in zip(weights, means, covariances):
            cdf += weight * stats.norm.cdf(x, mean, np.sqrt(covariance))
        return cdf

    # Function to calculate FPR and FNR for the general population
    def calculate_general_population_metrics(probs_all, y_all, threshold=0.5):
        tn_all, fp_all, fn_all, tp_all = confusion_matrix(y_all, probs_all >= threshold).ravel()
        fpr_all = fp_all / (fp_all + tn_all)
        fnr_all = fn_all / (fn_all + tp_all)
        return fpr_all, fnr_all

    # Function to calculate the confusion matrix elements based on the CDFs and threshold
    def confusion(threshold, sample_config, pos_cdf, neg_cdf):
        lst = [
            sample_config['n11'] * (1 - pos_cdf(threshold)),
            sample_config['n11'] * pos_cdf(threshold),
            sample_config['n01'] * (1 - neg_cdf(threshold)),
            sample_config['n01'] * neg_cdf(threshold)
        ]
        return np.array(lst).reshape(-1, 1) / sum(sample_config.values())

    # Define the objective function 
    def objective(threshold, pos_cdf, neg_cdf, sample_config, fpr_all, fnr_threshold):
        z = OptimalThresholdFinder.confusion(threshold, sample_config, pos_cdf, neg_cdf)
        fpr_g = z[2] / (z[2] + z[3])
        fnr_g = z[1] / (z[0] + z[1])

        # Only consider thresholds where FNR is less than or equal to the general population's FNR
        if fnr_g <= fnr_threshold:
            return abs(fpr_g - fpr_all)
        else:
            return np.inf

    # Grid search to find the optimal threshold
    def grid_search(self, pos_cdf, neg_cdf, sample_config, fpr_all, fnr_threshold, thresholds=np.linspace(0.5, 1, 100)):
        best_threshold = 0.5
        best_loss = np.inf
        for threshold in thresholds:
            loss = self.objective(threshold, pos_cdf, neg_cdf, sample_config, fpr_all, fnr_threshold)
            if loss < best_loss:
                best_loss = loss
                best_threshold = threshold
        return best_threshold

    # Function to calculate the sample configuration based on the fitted CDFs
    def calculate_sample_config(y_group, probs_group, pos_cdf, neg_cdf):
        thresholds = np.linspace(0.5, 1, 100)
        sample_config = {'n11': 0, 'n01': 0, 'n10': 0, 'n00': 0}
        for threshold in thresholds:
            n11 = sum((y_group == 1) & (probs_group >= threshold))
            n01 = sum((y_group == 0) & (probs_group >= threshold))
            n10 = sum((y_group == 1) & (probs_group < threshold))
            n00 = sum((y_group == 0) & (probs_group < threshold))
            if n11 + n01 + n10 + n00 == len(y_group):
                sample_config = {'n11': n11, 'n01': n01, 'n10': n10, 'n00': n00}
                break
        return sample_config

    # Function to process and evaluate a single dataframe
    def process_dataframe(self, df, df_name):
        # Define a mapping for the age groups to numerical values
        age_group_mapping = {
            '<65': 0,
            '65-74': 1,
            '75-84': 2,
            '85+': 3
        }
        df['age_group_num'] = df['age_group'].map(age_group_mapping)

        # Split the dataframe into Train, Val, and Test sets
        df_train = df[df['Train_Val_Test'] == 'Train'][['proba0', 'true_label_AD', 'age_group_num']]
        df_val = df[df['Train_Val_Test'] == 'Val'][['proba0', 'true_label_AD', 'age_group_num']]
        df_test = df[df['Train_Val_Test'] == 'Test'][['proba0', 'true_label_AD', 'age_group_num']]

        # Get validation data for general population and specific age groups
        probs_all_val = df_val['proba0']
        y_all_val = df_val['true_label_AD']
        probs_g_val_65_74 = df_val[df_val['age_group_num'] == 1]['proba0']
        y_g_val_65_74 = df_val[df_val['age_group_num'] == 1]['true_label_AD']
        probs_g_val_75_84 = df_val[df_val['age_group_num'] == 2]['proba0']
        y_g_val_75_84 = df_val[df_val['age_group_num'] == 2]['true_label_AD']
        probs_g_val_85 = df_val[df_val['age_group_num'] == 3]['proba0']
        y_g_val_85 = df_val[df_val['age_group_num'] == 3]['true_label_AD']

        # Calculate initial metrics for general population
        fpr_all_val, fnr_all_val = self.calculate_general_population_metrics(probs_all_val, y_all_val)

        # Convert Series to numpy arrays before reshaping
        probs_g_val_75_84_AD = probs_g_val_75_84[y_g_val_75_84 == 1].values
        probs_g_val_75_84_CN = probs_g_val_75_84[y_g_val_75_84 == 0].values
        probs_g_val_85_AD = probs_g_val_85[y_g_val_85 == 1].values
        probs_g_val_85_CN = probs_g_val_85[y_g_val_85 == 0].values

        # Fit the best distribution for each group (AD and CN)
        dist_75_84_AD_name, dist_75_84_AD_params = self.fit_distribution(probs_g_val_75_84_AD)
        dist_75_84_CN_name, dist_75_84_CN_params = self.fit_distribution(probs_g_val_75_84_CN)
        dist_85_AD_name, dist_85_AD_params = self.fit_distribution(probs_g_val_85_AD)
        dist_85_CN_name, dist_85_CN_params = self.fit_distribution(probs_g_val_85_CN)

        # Get the CDF functions for each fitted distribution
        cdf_75_84_AD = self.get_distribution_cdf(dist_75_84_AD_name, dist_75_84_AD_params)
        cdf_75_84_CN = self.get_distribution_cdf(dist_75_84_CN_name, dist_75_84_CN_params)
        cdf_85_AD = self.get_distribution_cdf(dist_85_AD_name, dist_85_AD_params)
        cdf_85_CN = self.get_distribution_cdf(dist_85_CN_name, dist_85_CN_params)

        # Define the sample configurations for each group
        sample_config_75_84 = self.calculate_sample_config(y_g_val_75_84, probs_g_val_75_84, cdf_75_84_AD, cdf_75_84_CN)
        sample_config_85 = self.calculate_sample_config(y_g_val_85, probs_g_val_85, cdf_85_AD, cdf_85_CN)

        # Find optimal thresholds using the best fitted distributions
        optimal_threshold_75_84 = self.grid_search(cdf_75_84_AD, cdf_75_84_CN, sample_config_75_84, fpr_all_val, fnr_all_val)
        optimal_threshold_85 = self.grid_search(cdf_85_AD, cdf_85_CN, sample_config_85, fpr_all_val, fnr_all_val)

        # Initialize result dictionaries
        result_75_84 = {
            'Dataframe': df_name,
            'Optimal Threshold': optimal_threshold_75_84,
            'TN (Default)': 0, 'FP (Default)': 0, 'FN (Default)': 0, 'TP (Default)': 0,
            'Accuracy (Default)': 0, 'Precision (Default)': 0, 'Recall (Default)': 0,
            'Balanced Accuracy (Default)': 0, 'FPR (Default)': 0, 'FNR (Default)': 0,
            'TN (Optimal)': 0, 'FP (Optimal)': 0, 'FN (Optimal)': 0, 'TP (Optimal)': 0,
            'Accuracy (Optimal)': 0, 'Precision (Optimal)': 0, 'Recall (Optimal)': 0,
            'Balanced Accuracy (Optimal)': 0, 'FPR (Optimal)': 0, 'FNR (Optimal)': 0
        }

        result_85 = {
            'Dataframe': df_name,
            'Optimal Threshold': optimal_threshold_85,
            'TN (Default)': 0, 'FP (Default)': 0, 'FN (Default)': 0, 'TP (Default)': 0,
            'Accuracy (Default)': 0, 'Precision (Default)': 0, 'Recall (Default)': 0,
            'Balanced Accuracy (Default)': 0, 'FPR (Default)': 0, 'FNR (Default)': 0,
            'TN (Optimal)': 0, 'FP (Optimal)': 0, 'FN (Optimal)': 0, 'TP (Optimal)': 0,
            'Accuracy (Optimal)': 0, 'Precision (Optimal)': 0, 'Recall (Optimal)': 0,
            'Balanced Accuracy (Optimal)': 0, 'FPR (Optimal)': 0, 'FNR (Optimal)': 0
        }

        # Apply the optimal thresholds to the test set and evaluate
        def evaluate(df_test, threshold_75_84, threshold_85, result_75_84, result_85, description):
            df_test_copy = df_test.copy()
            df_test_copy.loc[df_test_copy['age_group_num'].isin([0, 1]), 'transformed_AD_label'] = df_test_copy.loc[df_test_copy['age_group_num'].isin([0, 1]), 'proba0'].apply(lambda x: 1 if x >= 0.5 else 0)
            df_test_copy.loc[df_test_copy['age_group_num'] == 2, 'transformed_AD_label'] = df_test_copy.loc[df_test_copy['age_group_num'] == 2, 'proba0'].apply(lambda x: 1 if x >= threshold_75_84 else 0)
            df_test_copy.loc[df_test_copy['age_group_num'] == 3, 'transformed_AD_label'] = df_test_copy.loc[df_test_copy['age_group_num'] == 3, 'proba0'].apply(lambda x: 1 if x >= threshold_85 else 0)

            # Evaluate separately for 75-84 and 85+ age groups
            df_test_75_84 = df_test_copy[df_test_copy['age_group_num'] == 2]
            df_test_85 = df_test_copy[df_test_copy['age_group_num'] == 3]

            def store_confusion_matrix_info(df_test_group, result_dict, description):
                conf_matrix_test = confusion_matrix(df_test_group['true_label_AD'], df_test_group['transformed_AD_label'])
                accuracy_test = accuracy_score(df_test_group['true_label_AD'], df_test_group['transformed_AD_label'])
                precision_test = precision_score(df_test_group['true_label_AD'], df_test_group['transformed_AD_label'])
                recall_test = recall_score(df_test_group['true_label_AD'], df_test_group['transformed_AD_label'])
                balanced_accuracy_test = (recall_test + (conf_matrix_test[0, 0] / (conf_matrix_test[0, 0] + conf_matrix_test[0, 1]))) / 2
                fpr_test = conf_matrix_test[0, 1] / (conf_matrix_test[0, 0] + conf_matrix_test[0, 1])
                fnr_test = conf_matrix_test[1, 0] / (conf_matrix_test[1, 0] + conf_matrix_test[1, 1])

                if description == "default threshold":
                    result_dict['TN (Default)'] = conf_matrix_test[0, 0]
                    result_dict['FP (Default)'] = conf_matrix_test[0, 1]
                    result_dict['FN (Default)'] = conf_matrix_test[1, 0]
                    result_dict['TP (Default)'] = conf_matrix_test[1, 1]
                    result_dict['Accuracy (Default)'] = accuracy_test
                    result_dict['Precision (Default)'] = precision_test
                    result_dict['Recall (Default)'] = recall_test
                    result_dict['Balanced Accuracy (Default)'] = balanced_accuracy_test
                    result_dict['FPR (Default)'] = fpr_test
                    result_dict['FNR (Default)'] = fnr_test
                else:
                    result_dict['TN (Optimal)'] = conf_matrix_test[0, 0]
                    result_dict['FP (Optimal)'] = conf_matrix_test[0, 1]
                    result_dict['FN (Optimal)'] = conf_matrix_test[1, 0]
                    result_dict['TP (Optimal)'] = conf_matrix_test[1, 1]
                    result_dict['Accuracy (Optimal)'] = accuracy_test
                    result_dict['Precision (Optimal)'] = precision_test
                    result_dict['Recall (Optimal)'] = recall_test
                    result_dict['Balanced Accuracy (Optimal)'] = balanced_accuracy_test
                    result_dict['FPR (Optimal)'] = fpr_test
                    result_dict['FNR (Optimal)'] = fnr_test

            store_confusion_matrix_info(df_test_75_84, result_75_84, description)
            store_confusion_matrix_info(df_test_85, result_85, description)

        # Evaluate with default threshold
        evaluate(df_test, 0.5, 0.5, result_75_84, result_85, "default threshold")

        # Evaluate with optimal thresholds
        evaluate(df_test, optimal_threshold_75_84, optimal_threshold_85, result_75_84, result_85, "optimal threshold")

        # Save the transformed labels to the original DataFrame
        df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 0), 'transformed_AD_label'] = df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 0), 'proba0'].apply(lambda x: 1 if x >= 0.5 else 0)
        df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 1), 'transformed_AD_label'] = df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 1), 'proba0'].apply(lambda x: 1 if x >= 0.5 else 0)
        df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 2), 'transformed_AD_label'] = df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 2), 'proba0'].apply(lambda x: 1 if x >= optimal_threshold_75_84 else 0)
        df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 3), 'transformed_AD_label'] = df.loc[(df['Train_Val_Test'] == 'Test') & (df['age_group_num'] == 3), 'proba0'].apply(lambda x: 1 if x >= optimal_threshold_85 else 0)

        return df, result_75_84, result_85