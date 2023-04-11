import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

data = pd.read_csv('Diabities_prediction.csv')

Zero_not_accepted =['Glucose','BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for c in Zero_not_accepted:
    data[c] = data[c].replace(0, np.NaN)
    mean = int(data[c].mean(skipna = True))
    data[c] = data[c].replace(np.NaN, mean)

X = data.iloc[:, 0 : 8]
Y = data.iloc[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=.23)

Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=11, p = 2, metric='euclidean')

classifier.fit(X_train, Y_train)

pred = classifier.predict(X_test)
# print(pred)

cm = confusion_matrix(Y_test, pred)
print(cm)

print(f1_score(Y_test, pred))
print(accuracy_score(Y_test, pred))