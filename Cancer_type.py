import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_excel('cancer_Data.xlsx')
print(data.head())

sns.jointplot(x='Column3', y='Column4', data = data)
pt.show()
# sns.heatmap(data.corr())
# pt.show()
# print(data.isnull().sum())
X = data[['Column23', 'Column24', 'Column25', 'Column26', 'Column27']]
Y = data['Column2']
# print(X.head())
# print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)

y_pred = log_model.predict(X_train)
print(y_pred)