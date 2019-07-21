'''

main steps:
1. load
2. prepare data
4. fit
5. predict
6. evaluate
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('IRIS.csv')
print(data.describe())
print(data.head())

y = data['species']
X = data.drop(['species'], axis=1)
print(y.head())
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
predictions = dt_model.predict(X_test)
accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
print(accuracy)