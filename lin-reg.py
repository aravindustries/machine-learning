import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from bs4 import BeautifulSoup


class data:
    def __init__(self, df):
        rows, col = df.shape
        df = df.sample(frac=1, replace=False, random_state=42)
        self.train = df.iloc[0:int(np.ceil(rows*0.8))]
        self.valid = df.iloc[int(np.ceil(rows*0.8)):int(np.ceil(rows*0.9))]
        self.test = df.iloc[int(np.ceil(rows*0.9)):rows]


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


def getCorrMatrix(X):
    return np.corrcoef(X, rowvar=False)
    


url = 'https://hastie.su.domains/ElemStatLearn/datasets/prostate.data'

df = fetchProstateCancerData(url)

pcdata = data(df)

X = pcdata.train.iloc[:, 0:8].to_numpy()

print(getCorrMatrix(X))

# X = df.iloc[0:count, 0:8].to_numpy()

# y = df['lpsa']

# X_b = np.c_[np.ones((count, 1)), X]

# print(y)

# beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# print(beta)

# y_pred =  X_b.dot(beta)

# # y_pred_binary = (y_pred >= 0.5).astype(int)
# print(y_pred)

# rss = np.sum((y - y_pred) ** 2)

# print(rss)

# plt.plot(y, label='y')  # Plot the first array (y1)
# plt.plot(y_pred, label='y_hat')  # Plot the second array (y2)

# # Add labels and title

# # Add a legend to distinguish between the two plots
# plt.legend()

# # Display the plot
# plt.show()

# print(df['lcavol'].corr(df['age']))