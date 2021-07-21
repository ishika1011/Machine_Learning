

import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
#logistic Regression
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

#predicting the test set results

y_pred = classifier.predict(x_test)

#making  the confusin matrix

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test,y_pred)
print((cm[0][0]+cm[1][1]/cm[0][0]+cm[1][0]+cm[1][1])*100)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


########KNN

import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np

dataset = pd.read_csv('Social_Network_Ads.csv')

#just for check

#dataset.shape
#dataset.info()
print("K-NN k nearest neighbourhood")
#split data set

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,4].values

#spliting dataset in training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


from sklearn.neighbors import  KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)#p=2 therefore euclidien distance, if p=1 manhatten distance
classifier.fit(x_train, y_train)

#predicting the Test set result
y_pred = classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


##### SVM

import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np

dataset = pd.read_csv('Social_Network_Ads.csv')

print("SVM")
#split data set

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,4].values

#spliting dataset in training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

#predicting the Test set result
y_pred = classifier.predict(x_test)


#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

