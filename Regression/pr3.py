import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('dataset3.csv')

#just for check

dataset.shape
dataset.info()

#-------------------
#split data set

# y = mx + c - simple linear regreesiom
# y = m1x1 + m2x2 + m3x3 + ...+mnxn + c - multi linear regression
# salary = m * Yearsofexperience + c - we have to implement this 
# X  <-  YearsExperience 
# Y  <-  Salary

x = dataset.iloc[:, :-1].values
x
y = dataset.iloc[:,1].values
y


#-----------------------
#split for training the model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#by adding below library we are implementing
#salary = m * yearsofexperiance + c
#regression start here 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#prediction the test result
y_pred = regressor.predict(x_test)
print(y_pred)

y_test

y_pred1 = regressor.predict([[7]])
print(y_pred1)

#testing accuracy
regressor.score(x_test,y_test)
#training accuracy
regressor.score(x_train,y_train)

#ploting

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salary vs Experience  (Training set)' )
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#visulizing the Test set results
plt.scatter(x_test,y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('salary vs Experience  (Test set)' )
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()


import pickle
with open('source_object_name.pkl', 'wb') as f:
    pickle.dump(regressor, f)
with open('source_object_name.pkl', 'rb') as f1:
    dd = pickle.load(f1)
ans = dd.predict([[7]])
ans

#MULTILINEAR REGRESSION
#y = m1x1 + m2x2 + .... + mnxn + c

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset4 = pd.read_csv('dataset4.csv')
dataset4.head()


X = dataset4.iloc[:,:-1].values
Y = dataset4.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X

transform = ColumnTransformer([("State data",OneHotEncoder(),[3])],remainder='passthrough')
X = transform.fit_transform(X)
X


X = X[:,1:]
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,
                                                 random_state=0)
# multi linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


y_pred = regressor.predict(X_test)
y_pred

# trainning accurecy
regressor.score(X_train,Y_train)

# testing accurecy
regressor.score(X_test,Y_test)

#pred accurecy
regressor.score(X_test,y_pred)









