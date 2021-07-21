#PCA
#(UN-Supervised Feature Extration dimensionality reduction technique)

import pandas as pd
dataset = pd.read_csv('Wine.csv')
dataset.corr()



X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
Feature Selection
FFS(Forward Feature Selection)
BFE(Backward Feature Elimination)
Feature Extraction
PCA(Principal Component Analysis)
LDA(Linear Discreminent Analysis)
K-PCA(Kernal-PCA)
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)



y_pred = model.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



"""
LDA
(Supervised Feature Extration dimensionality reduction technique)

LDA is a Supervised Feature Extration dimensionality reduction technique.

why? while transforming dataset we takes input(X) and Output(Y) both into considerations.
"""
import pandas as pd
dataset = pd.read_csv('Wine.csv')
dataset.corr()



X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test) 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train,Y_train)



y_pred = model.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



"""
Exercise
n_components = 4 and 2

KNN

SVM

Decision Tree

Random Forest

Naive Bayes
"""

#PCA
import pandas as pd
# fetching dataset into X and Y
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
# training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train) # only we are providing input(X_train) not Y_train
X_test = pca.transform(X_test)



#KNN
#n-components=2
#(PCA - Components = 2)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
knn.fit(X_train,Y_train)



y_pred = knn.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#SVM
#(PCA - Components = 2)
from sklearn.svm import SVC
svm = SVC(kernel = 'linear',random_state=0)
svm.fit(X_train,Y_train)



y_pred = svm.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Decision Tree
#(PCA - Components = 2)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)



y_pred = dt.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Random Forest
#(PCA - Components = 2)



from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
RF = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
RF.fit(X_train, Y_train)



y_pred = RF.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Naive Bayes
#(PCA - Components = 2)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,Y_train)



y_pred = NB.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#N_Components = 4
# training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train) # only we are providing input(X_train) not Y_train
X_test = pca.transform(X_test)
#KNN
#(PCA - Components = 4)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
knn.fit(X_train,Y_train)



y_pred = knn.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#SVM
#(PCA - Components = 4)



from sklearn.svm import SVC
svm = SVC(kernel = 'linear',random_state=0)
svm.fit(X_train,Y_train)



y_pred = svm.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Decision Tree
#(PCA - Components = 4)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)



y_pred = dt.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Random Forest
#(PCA - Components = 4)
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
RF = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
RF.fit(X_train, Y_train)



y_pred = RF.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Naive Bayes
#(PCA - Components = 4)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,Y_train)



y_pred = NB.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#LDA
import pandas as pd
# fetching dataset into X and Y
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
#N_Components = 2
# training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test) 



#KNN
#(LDA - Components = 2)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
knn.fit(X_train,Y_train)



y_pred = knn.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#SVM
#(LDA - Components = 2)
from sklearn.svm import SVC
svm = SVC(kernel = 'linear',random_state=0)
svm.fit(X_train,Y_train)




y_pred = svm.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Decision Tree
#(LDA - Components = 2)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)



y_pred = dt.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Random Forest
#(LDA - Components = 2)
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
RF = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
RF.fit(X_train, Y_train)



y_pred = RF.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Naive Bayes
#(LDA - Components = 2)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,Y_train)



y_pred = NB.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#N_Components = 4
# training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=4)
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test) 



#KNN
#(LDA - Components = 4)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
knn.fit(X_train,Y_train)



y_pred = knn.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#SVM
#(LDA - Components = 4)
from sklearn.svm import SVC
svm = SVC(kernel = 'linear',random_state=0)
svm.fit(X_train,Y_train)



y_pred = svm.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Decision Tree
#(LDA - Components = 4)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)



y_pred = dt.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Random Forest
#(LDA - Components = 4)
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
RF = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
RF.fit(X_train, Y_train)



y_pred = RF.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



#Naive Bayes
#(LDA - Components = 4)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,Y_train)



y_pred = NB.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm











