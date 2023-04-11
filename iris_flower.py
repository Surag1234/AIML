from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head())

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_train'] = np.random.uniform(0,1, len(df)) <= 0.75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]

features  = df.columns[:4]
y = pd.factorize(train['species'])[0]

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features],y)
print(clf.predict(test[features]))
