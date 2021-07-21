# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:07:54 2021

@author: dhruv
"""
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np

dataset = pd.read_csv('Wine.csv')
dataset.corr()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##### feature selection - FFS, BFE, ....

##### feature extrection -PCA -principal component Analysis 
##### LDA - Linear Discremeinet Analysis 
##### k-pca -kernel pca

##### Apply PCA
from sklearn.decomposition import PCA
pca=PCA(n_components = 2)
x_train=pca.fit_transform(x_train)
x_test= pca.transform(x_test)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



import pandas as pd
#import numpy as np

dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Wine.csv')
dataset.corr()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##### Apply LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components = 2)

x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#pca unsupervised
#lda U\supervised
