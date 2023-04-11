import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as pt

data = pd.read_csv('area_price.csv')
# print(data.head())

pt.xlabel('area(sq m)')
pt.ylabel('price(rupees)')
pt.scatter(data.area, data.price, color='red')
# pt.show()

reg = linear_model.LinearRegression()
reg.fit(X=data[['area']], y=data.price)
# print(reg.predict([[3300]]))

data1 = pd.read_csv('percapita_income.csv')
# print(data1.head())
data1.columns.values[1] = 'pci'
# print(data1.columns)
reg1 = linear_model.LinearRegression()
reg1.fit(X=data1[['year']], y=data1.pci)
print(reg1.predict([[2020]]))