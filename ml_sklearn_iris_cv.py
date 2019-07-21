import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import SCORERS

data = pd.read_csv('IRIS.csv')
print(data.describe())
print(data.head())

y = data['species']
X = data.drop(['species'], axis=1)
print(y.head())
print(X.head())

dt_model = DecisionTreeClassifier()
print(SCORERS.keys())
scores = cross_val_score(dt_model, X, y, cv=5, n_jobs=4, scoring='accuracy')
print(scores)
print(scores.mean())
print(scores.std())
