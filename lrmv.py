import numpy as np
import pandas as pd
from sklearn import linear_model
import math
from matplotlib import pyplot as pt

data = pd.read_csv('area,bedrooms,age,price.csv')
# print(data.head())
mean_bedrooms = math.floor(data.bedrooms.median())
data.bedrooms = data.bedrooms.fillna(mean_bedrooms)
# print(data.head())

reg = linear_model.LinearRegression()
reg.fit(data[['area', 'bedrooms', 'age']], data.price)
print(reg.predict([[3000, 3, 40]]))

# print(reg.coef_)
# print(reg.intercept_)

# print(112.06244194*3000 + 23388.88007794*3 - 3231.71790863*40 + 221323.00186540437 )