import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
import argparse
warnings.filterwarnings("ignore")

class Calibration:
    def __init__(self, baseline, save_path, lam=0.001, lr=0.1, nepoches=2000):
        self.baseline = baseline
        self.save_path = save_path
        self.lam = lam
        self.lr = lr
        self.nepoches = nepoches

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def prob(w, X):
        return Calibration.sigmoid(X.dot(w))

    def loss(w, X, y, lam):
        z = Calibration.prob(w, X)
        return -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z)) + 0.5 * lam / X.shape[0] * np.sum(w * w)

    def logistic_regression(self, w_init, X, y):
        N, d = X.shape
        w = w_init
        w_old = w_init
        loss_hist = [Calibration.loss(w_init, X, y, self.lam)]
        ep = 0
        while ep < self.nepoches:
            ep += 1
            mix_ids = np.random.permutation(N)
            for i in mix_ids:
                xi = X[i]
                yi = y[i]
                zi = Calibration.sigmoid(xi.dot(w))
                w = w - self.lr * ((zi - yi) * xi + self.lam * w)
            loss_hist.append(Calibration.loss(w, X, y, self.lam))
            if np.linalg.norm(w - w_old) / d < 1e-6:
                break
            w_old = w
        return w, loss_hist

    def preprocess_baseline(self):
        # Create the new columns with inverse values because of in the baseline dataset labels of CN: 1 and AD: 0
        self.baseline['predicted_label_AD'] = self.baseline['predicted_label'].apply(lambda x: 1 if x == 0 else 0)
        self.baseline['true_label_AD'] = self.baseline['true_label'].apply(lambda x: 1 if x == 0 else 0)
        # Map age group
        self.baseline['age_group'] = self.baseline['age_group'].map({'65 to 74': '65-74', '75 to 84': '75-84', 'Less than 65': '<65', 'Greater than 85': '85+'})

    def apply_logistic_regression(self):
        for split in range(5):
            train_data = self.baseline[(self.baseline['Split'] == split) & (self.baseline['Train_Val_Test'] == 'Train')]
            val_data = self.baseline[(self.baseline['Split'] == split) & (self.baseline['Train_Val_Test'] == 'Val')]
            test_data = self.baseline[(self.baseline['Split'] == split) & (self.baseline['Train_Val_Test'] == 'Test')]
            
            # Perform logistic regression on Train set to get initial parameters
            w_init = np.zeros(train_data[['proba0']].shape[1])
            w, loss_hist = self.logistic_regression(w_init, train_data[['proba0']].values, train_data['true_label_AD'].values)
            
            # Perform Platt Scaling on Val set using initial parameters from Train set
            log_reg = LogisticRegression()
            log_reg.coef_ = np.array([w])
            log_reg.intercept_ = np.array([0])
            log_reg.classes_ = np.array([0, 1])
            log_reg.fit(val_data[['proba0']], val_data['true_label_AD'])
            val_proba_platt = log_reg.predict_proba(val_data[['proba0']])[:, 1]
            test_proba_platt = log_reg.predict_proba(test_data[['proba0']])[:, 1]
            
            # Store the calibrated probabilities back to the main dataframe
            self.baseline.loc[val_data.index, 'platt_proba'] = val_proba_platt
            self.baseline.loc[test_data.index, 'platt_proba'] = test_proba_platt

    def run(self):
        self.preprocess_baseline()
        self.apply_logistic_regression()
        self.baseline.to_csv(self.save_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Run Calibration on Baseline Dataset')
    parser.add_argument('baseline_path', type=str, help='Path to the baseline CSV file')
    parser.add_argument('save_path', type=str, help='Path to save the calibrated CSV file')

    args = parser.parse_args()

    # Load the dataset
    baseline = pd.read_csv(args.baseline_path)
    
    # Create an instance of the Calibration class and run the calibration process
    calibration = Calibration(baseline, args.save_path)
    calibration.run()

if __name__ == "__main__":
    main()
