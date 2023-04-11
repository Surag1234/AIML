import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as pt
import seaborn as sns;sns.set(font_scale = 1.2)

recipies = pd.read_excel('data_cake.xlsx')
print(recipies.head())
sns.lmplot(x='Flour',y='Sugar',data = recipies, hue = 'Type', palette = 'Set1',fit_reg = False,scatter_kws={"s":70}) 

type_lable = np.where(recipies['Type'] == 'Muffin',0,1)
recipie_features = recipies.columns.values[1:].tolist()
print(recipie_features)
ingredients = recipies[['Flour','Sugar']].values
print(ingredients)
model = svm.SVC(kernel='linear')
model.fit(ingredients,type_lable)

w = model.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(30,60)
yy = a*xx - (model.intercept_[0])/w[1]

b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a*b[0])
b= model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

sns.lmplot(x='Flour',y='Sugar',data = recipies, hue = 'Type', palette = 'Set1',fit_reg = False,scatter_kws={"s":70}) 
pt.plot(xx ,yy, linewidth = 2, color = 'black')
pt.plot(xx,yy_down, 'k--')
pt.plot(xx,yy_up, 'k--')
# pt.show()

def MorC(flour, sugar):
    if(model.predict([[flour,sugar]])) == 0:
        print("You are looking at Muffin recipie")
    else:
        print("You are looking at Cup cake recipie")

A = int(input("Enter the amount of flour added : "))
B = int(input("Enter the amount of sugar added : "))

MorC(A,B)

sns.lmplot(x='Flour',y='Sugar',data = recipies, hue = 'Type', palette = 'Set1',fit_reg = False,scatter_kws={"s":70}) 
pt.plot(xx ,yy, linewidth = 2, color = 'black')
pt.plot(A, B, 'yo', markersize = '10')
pt.show()