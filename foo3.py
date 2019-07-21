'''
steps ML
1. load data
2. prepare data
3. training fit
4. predict
5. evaluate
'''

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# step 1
iris_data = pd.read_csv('IRIS.csv')
print(iris_data.head())

# step 2
y = iris_data['species']
print(y.head())
X = iris_data.drop(['species'], axis=1)
print(X.head())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print('encoded y', y_encoded[-10:])


# CV
model = DecisionTreeRegressor()

scores = -1 * cross_val_score(estimator=model,
                         X=X, y=y_encoded,
                         cv=5, n_jobs=4,
                        scoring='neg_mean_absolute_error')
print('result', scores)
print('mean', scores.mean())
