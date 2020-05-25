#!/usr/bin/env python
#akhkdfafuidayf
#lkaj;lskldslakfld
#kj;adklfjdsafkldadasd
import joblib
# coding: utf-8
import pandas as pd

dataset = pd.read_csv("/programs/Social_Network_Ads.csv")

X = dataset[['Age', 'EstimatedSalary' ] ] 

y = dataset['Purchased']

X = X.values

y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=4)

model.fit(X_train, y_train)

model.predict([[ 61, 200000 ]] )

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "pyfiles/KNN_model.h5")

file = open("pyfiles/accuracy_score.txt", "w") 

file.write(accuracy)

file.close()












