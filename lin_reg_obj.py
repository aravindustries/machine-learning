import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

class data:
    def __init__(self, df):
        rows, col = df.shape
        df = df.sample(frac=1, replace=False, random_state=42)
        self.train = df.iloc[0:int(np.ceil(rows*0.8))]
        self.valid = df.iloc[int(np.ceil(rows*0.8)):int(np.ceil(rows*0.9))]
        self.test = df.iloc[int(np.ceil(rows*0.9)):rows]


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.standard_errors = None
        self.z_scores = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.intercept = theta_best[0]
        self.coefficients = theta_best[1:]
    
        y_pred = X_b @ theta_best
        residuals = y - y_pred
        
        n = X.shape[0]
        p = X.shape[1]
        residual_variance = (residuals.T @ residuals) / (n - p - 1)
        
        cov_matrix = np.linalg.inv(X_b.T @ X_b) * residual_variance
        
        self.standard_errors = np.sqrt(np.diag(cov_matrix))
        
        full_params = np.r_[self.intercept, self.coefficients]
        self.z_scores = full_params / self.standard_errors
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        return X_b @ np.r_[self.intercept, self.coefficients]
    
    def get_standard_errors(self):
        return self.standard_errors
    
    def get_z_scores(self):
        return self.z_scores

class RidgeRegression:
    def __init__(self, alpha=1.0, feature_names=None):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
        self.feature_names = feature_names

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  
        
        n_features = X_b.shape[1]
        A = X_b.T @ X_b + self.alpha * np.eye(n_features)  
        b = X_b.T @ y
        theta = np.linalg.inv(A) @ b
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def predict(self, X):
        return X @ self.coefficients + self.intercept

    def tune_alpha(self, X_train, y_train, X_val, y_val, alphas):
        best_alpha = alphas[0]
        best_mse = float('inf')
        coefficients_history = []
        
        for alpha in alphas:
            self.alpha = alpha
            self.fit(X_train, y_train)
            y_pred = self.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            
            coefficients_history.append(self.coefficients)
            
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        
        self.alpha = best_alpha
        self.plot_coefficients(alphas, coefficients_history, best_alpha)
        
        return best_alpha

    def plot_coefficients(self, alphas, coefficients_history, best_alpha):
        plt.figure(figsize=(10, 6))
        for i in range(coefficients_history[0].shape[0]):
            plt.plot(alphas, [coef[i] for coef in coefficients_history], label=self.feature_names[i])
        
        plt.xscale('log')
        plt.axvline(x=best_alpha, color='r', linestyle='--', label='Optimal Alpha')  # Dashed vertical line
        
        plt.xlabel('Alpha (lambda)')
        plt.ylabel('Coefficient Value')
        plt.title('Ridge Regression Coefficients vs Alpha')
        plt.legend()
        plt.grid()
        plt.show()


class LassoRegression:
    def __init__(self, X_train, y_train, feature_names, alphas=None):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.alphas = alphas if alphas is not None else np.logspace(-4, 2, 100)
        self.optimal_alpha = None
        self.coefs = None
        self.model = None

    def fit(self):
        lasso_cv = LassoCV(alphas=self.alphas, cv=5, random_state=42)
        lasso_cv.fit(self.X_train, self.y_train)
        self.optimal_alpha = lasso_cv.alpha_
        self.coefs = lasso_cv.coef_
        self.model = lasso_cv

    def predict(self, X):
        return self.model.predict(X)

    def get_selected_features(self):
        selected_features = np.array(self.feature_names)[self.coefs > 0.0001]
        return selected_features

    def plot_coefficients(self):
       
        coefs = []
        for alpha in self.alphas:
            lasso = LassoCV(alphas=[alpha], cv=5, random_state=42)
            lasso.fit(self.X_train, self.y_train)
            coefs.append(lasso.coef_)

        coefs = np.array(coefs)

        plt.figure(figsize=(10, 6))
        for i in range(coefs.shape[1]):
            plt.plot(self.alphas, coefs[:, i], label=self.feature_names[i])
        
        plt.xscale('log')
        plt.axvline(x=self.optimal_alpha, color='r', linestyle='--', label='Optimal Alpha')
        plt.title('Lasso Coefficients vs Alpha (Lambda)')
        plt.xlabel('Alpha (lambda)')
        plt.ylabel('Coefficient Value')
        plt.legend()
        plt.grid()
        plt.show()

    def get_optimal_alpha(self):
        return self.optimal_alpha
