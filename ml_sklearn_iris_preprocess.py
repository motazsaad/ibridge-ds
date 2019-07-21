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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('IRIS.csv')
print(data.describe())
print(data.head())

y = data['species']
X = data.drop(['species'], axis=1)
print(y.head())
print(X.head())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print('encoded y ')
print(y_encoded[:10])
print(y_encoded[-10:])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.33, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
result = accuracy_score(y_true=y_test, y_pred=predictions)
print(result)