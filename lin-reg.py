import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from bs4 import BeautifulSoup

from lin_reg_obj import data, LinearRegression, RidgeRegression, LassoRegression

def fetchProstateCancerData(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        lines = soup.get_text().splitlines()
        with open('data.csv', 'w', newline='') as file:
            for line in lines:
                stripped_line = line.strip()
                data = stripped_line.replace('\t', ',')
                file.write(data + '\n')
        df = pd.read_csv('data.csv')
        return df
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return pd.DataFrame()


def concatenate_square_and_product(data):
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    
    squared = data ** 2
    product = np.multiply(data, data[:, ::-1])
    result = np.concatenate((data, squared, product), axis=1)

    return result

url = 'https://hastie.su.domains/ElemStatLearn/datasets/prostate.data'

pcdata = data(fetchProstateCancerData(url))

edge = 8

X_train = pcdata.train.iloc[:, :edge].to_numpy()
y_train = pcdata.train.iloc[:, edge].to_numpy()

X_val = pcdata.valid.iloc[:, :edge].to_numpy()
y_val = pcdata.valid.iloc[:, edge].to_numpy()

X_test = pcdata.test.iloc[:, :edge].to_numpy()
y_test = pcdata.test.iloc[:, edge].to_numpy()

feature_names = pcdata.train.iloc[:, :edge].columns

print()
print('Table 3.1:')

print(pcdata.train.iloc[:, 0:edge].corr())

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

coeffs = np.r_[linear_model.intercept, linear_model.coefficients]

table2 = pd.DataFrame({
    'Coefficient': coeffs,
    'Std. Error': linear_model.get_standard_errors(),
    'Z Score': linear_model.get_z_scores()})

table_index2 = feature_names

table2.index =  table_index2.insert(0, 'intercept') 

print()
print('Table 3.2:')

print(table2)

y_pred_lin = linear_model.predict(X_test)

mse_lin = np.mean((y_test - y_pred_lin) ** 2)

print()
print(f'Linear Regression MSE: {mse_lin}')

ridge_model = RidgeRegression(feature_names=feature_names)

alphas = np.logspace(-3, 3, 100)

optimal_alpha = ridge_model.tune_alpha(X_train, y_train, X_val, y_val, alphas)

print()
print(f'Optimal Lambda (Ridge): {optimal_alpha}')

ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

mse_ridge = np.mean((y_test - y_pred_ridge) ** 2)

print(f'Ridge Regression MSE: {mse_ridge}')

lasso_model = LassoRegression(X_train, y_train, feature_names)
lasso_model.fit()

print()
print("Optimal Lambda (Lasso):", lasso_model.get_optimal_alpha())
lasso_model.plot_coefficients()

y_pred_lasso = lasso_model.predict(X_test)

mse_lasso = np.mean((y_test - y_pred_lasso) ** 2)

print(f'Lasso Regression MSE: {mse_lasso}')
print(f'Selected Features: {lasso_model.get_selected_features()}')

X_nonlinear_train = concatenate_square_and_product(X_train)
X_nonlinear_test = concatenate_square_and_product(X_test)

linear_model2 = LinearRegression()

linear_model2.fit(X_nonlinear_train, y_train)

y_pred_lin2 = linear_model2.predict(X_nonlinear_test)

mse_lin2 = np.mean((y_test - y_pred_lin2) ** 2)

print()
print(f'Linear Regression on Nonlinear Dataset MSE: {mse_lin2}')

# plt.plot(y_test, label='y')
# plt.plot(y_pred_lin, label='y_lin')
# plt.plot(y_pred_ridge, label='y_ridge')
# plt.plot(y_pred_lasso, label='y_lasso')
# plt.legend()
# plt.show()