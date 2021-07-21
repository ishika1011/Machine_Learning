import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Social_Network_Ads.csv')
# logistic regression
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_train



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(X_train,Y_train)



Y_pred = classifier.predict(X_test)
Y_pred



Y_test



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))







#K-Nearest Neighbors
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
classifier.fit(X_train,Y_train)



y_pred = classifier.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))



#Support Vector Machine
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state=0)
classifier.fit(X_train,Y_train)



y_pred = classifier.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))






#Exercise
#Apply all below techniques in Social Network Ads dataset
#Naive Bayes
#Decision tree
#Random Forest


#Naive Bayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)



model.score(X_train,Y_train)



model.score(X_test,Y_test)



Y_pred = model.predict(X_test)
Y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))




#Decision Tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)




from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



y_pred = dt.predict(X_test)
y_pred



dt.score(X_train,Y_train)



dt.score(X_test,Y_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))




#Random Forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
model.fit(X_train, Y_train)




y_pred = model.predict(X_test)
y_pred



model.score(X_train,Y_train)



model.score(X_test,Y_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))




#GridSearchCV in KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
classifier.fit(X_train,Y_train)




from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_neighbors':[2,3,5,7,9,11],
     'weights':['uniform','distance'],
     'metric':['euclidean','manhattan']
}

gs = GridSearchCV(
    KNeighborsClassifier(),
    grid_params,
    verbose = 1,
    cv = 3,
    n_jobs = -1
)

gs_results = gs.fit(X_train,Y_train)


"""
n_neighbors = 6,
weights = 2,
metric = 2,
cross validations=3 
total = 6 * 2 * 2 * 3 = 72

"""
gs_results.best_score_




gs_results.best_estimator_




gs_results.best_params_



y_pred = gs.predict(X_test)
y_pred



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
cm



print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100)



from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))
                      