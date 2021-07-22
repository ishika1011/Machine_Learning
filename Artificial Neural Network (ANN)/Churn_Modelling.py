import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values

X

Y

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

transform = ColumnTransformer([("Cs",OneHotEncoder(),[1])],remainder='passthrough')
X = transform.fit_transform(X)
X = X[:,1:]

transform = ColumnTransformer([("Gender",OneHotEncoder(),[3])],remainder='passthrough')
X = transform.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)
dataset.max()


#feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train


X_test





#let's create ANN
# Importing the keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initilising the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_shape= (11,)))

#adding 2nd hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))

#adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# if multi then categoricl_crossentrophy,adwdata


#fitting the ANN to the training dataset
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)


#predicting the test set results
Y_pred  = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

Y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

cm


print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)

#using ann perform project for house price prediction
#integrate the model with flask webApp











