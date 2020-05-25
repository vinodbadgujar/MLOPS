#!/usr/bin/env python
import joblib
import sys 

#sys package is import for passing command line arguement

import pandas as pd

dataset = pd.read_csv("/programs/Social_Network_Ads.csv")

X = dataset[['Age', 'EstimatedSalary' ] ] 

y = dataset['Purchased']

X = X.values

y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
#it take a arguement from command line
model = KNeighborsClassifier(n_neighbors= int(sys.argv[1]))

model.fit(X_train, y_train)

model.predict([[ 61, 200000 ]] )

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
acc = acc*100

#Here I create file for storing accuracy of model

joblib.dump(model, "pyfiles/KNN_model.h5")

file = open("pyfiles/accuracy_score.txt", "w") 
accuracy = str(acc)

file.write(accuracy)

file.close()