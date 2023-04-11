import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('heart_attack.csv')
# print(data.head())

Y = data['target']
X = data.drop('target', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=43)

scaler = StandardScaler()
scale = scaler.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)

score = accuracy_score(Y_test,pred)
print(score)

print(confusion_matrix(Y_test, pred))

print(classification_report(Y_test, pred))