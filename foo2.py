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
from sklearn.model_selection import train_test_split
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

# split data

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,
                                                    test_size=0.33,
                                                    random_state=3)

# step 3
model = DecisionTreeRegressor()
# model.fit(X, y)
model.fit(X_train, y_train)

# steps 4
# predictions = model.predict(X)
predictions = model.predict(X_test)

# step 5 evaluation
# result = accuracy_score(y_true=y, y_pred=predictions)

result = mean_absolute_error(y_true=y_test, y_pred=predictions)

print('result', result)
