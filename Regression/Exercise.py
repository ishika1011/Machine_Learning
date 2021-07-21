#Multilinear Regression in dataset 3.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset3 = pd.read_csv('dataset3.csv')
dataset3.head()


X = dataset3.iloc[:,:-1].values
X



Y = dataset3.iloc[:,-1].values
Y


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


y_pred = regressor.predict(X_test)
y_pred


Y_test


# trainning accurecy
regressor.score(X_train,Y_train)
mult_lin_train_3 = []
mult_lin_train_3.append(regressor.score(X_train,Y_train))


# testing accurecy
regressor.score(X_test,Y_test)
mult_lin_test_3 = []
mult_lin_test_3.append(regressor.score(X_test,Y_test))









#Multilinear Regression in dataset 4.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset4 = pd.read_csv('dataset4.csv')
dataset4.head()


X = dataset4.iloc[:,:-1].values
X


Y = dataset4.iloc[:,-1].values
Y


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X


transform = ColumnTransformer([("State data",OneHotEncoder(),[3])], remainder='passthrough')
X = transform.fit_transform(X)
X


X = X[:,1:]
X


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)
# multi linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



y_pred = regressor.predict(X_test)
y_pred



# trainning accurecy
regressor.score(X_train,Y_train)
mult_lin_train_4 = []
mult_lin_train_4.append(regressor.score(X_train,Y_train))
# testing accurecy
regressor.score(X_test,Y_test)
mult_lin_test_4 = []
mult_lin_test_4.append(regressor.score(X_test,Y_test))












#Polynomial Regression in dataset3.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset3 = pd.read_csv('dataset3.csv')
X = dataset3.iloc[:,0:1].values # Years Experience column
Y = dataset3.iloc[:,-1].values  # salary column
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('(Linear Regression)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)



X_poly



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-2(Polynomial Regression)')
plt.xlabel('Year of Experience')
plt.ylabel('salary')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly,Y)


plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_3.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-3(Polynomial Regression)')
plt.xlabel('Year Of Experience')
plt.ylabel('salary')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_4.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-4(Polynomial Regression)')
plt.xlabel('Year Of Experience')
plt.ylabel('salary')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_5 = LinearRegression()
lin_reg_5.fit(X_poly,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_5.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-5(Polynomial Regression)')
plt.xlabel('Year Of Experience')
plt.ylabel('salary')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
lin_reg_6 = LinearRegression()
lin_reg_6.fit(X_poly,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_6.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-6(Polynomial Regression)')
plt.xlabel('Year Of Experience')
plt.ylabel('salary')
plt.show()



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size = 0.2, random_state=0)
#Polynomial regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



y_pred = regressor.predict(X_test)
y_pred



Y_test



# trainning accurecy
regressor.score(X_train,Y_train)
poly_train_3 = []
poly_train_3.append(regressor.score(X_train,Y_train))

# testing accurecy
regressor.score(X_test,Y_test)
poly_test_3 = []
poly_test_3.append(regressor.score(X_train,Y_train))











#Polynomial Regression in dataset4.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset4 = pd.read_csv('dataset4.csv')
dataset4.head()


X = dataset4.iloc[:,:-1].values
X



Y = dataset4.iloc[:,-1].values
Y


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transform = ColumnTransformer([("State data",OneHotEncoder(),[3])],remainder='passthrough')
X = transform.fit_transform(X)
X



X = X[:,-3:-2]  # R & D Spend
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('(Linear Regression)')
plt.xlabel('R & D Spend')
plt.ylabel('Profit')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)


plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-2(Polynomial Regression)')
plt.xlabel('R & D Spend')
plt.ylabel('Profit')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_3.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-3(Polynomial Regression)')
plt.xlabel('R & D Spend')
plt.ylabel('Profit')
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_4.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Degree-4(Polynomial Regression)')
plt.xlabel('R & D Spend')
plt.ylabel('Profit')
plt.show()



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size = 0.2,random_state=0)
# multi linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



y_pred = regressor.predict(X_test)
y_pred



# trainning accurecy
regressor.score(X_train,Y_train)
poly_train_4 = []
poly_train_4.append(regressor.score(X_train,Y_train))


# testing accurecy
regressor.score(X_test,Y_test)
poly_test_4 = []
poly_test_4.append(regressor.score(X_test,Y_test))




#Support Vector Regression in dataset3.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset3 = pd.read_csv('dataset3.csv')
X = dataset3.iloc[:,0:1].values
Y = dataset3.iloc[:,-1:].values
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('(Linear Regression)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()



#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_X.transform(Y)
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'linear', C=100, gamma='auto')
svr_reg.fit(X, Y)



plt.scatter(X,Y,color='red')
plt.plot(X,svr_reg.predict(X),color='blue')
plt.title('(Support Vector Regression)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()



y_pred = svr_reg.predict([[20000]])
y_pred



Y_test


regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print(regressor.score(X_train,Y_train))
print(regressor.score(X_test,Y_test))
SVR_train_3 = []
SVR_train_3.append(regressor.score(X_train,Y_train))
SVR_test_3 = []
SVR_test_3.append(regressor.score(X_test,Y_test))



#Support Vector Regression in dataset4.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset4 = pd.read_csv('dataset4.csv')
X = dataset4.iloc[:,:-1].values
Y = dataset4.iloc[:,-1:].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transform = ColumnTransformer([("State data",OneHotEncoder(),[3])],remainder='passthrough')
X = transform.fit_transform(X)
X



X = X[:,-3:-2]  # R & D Spend
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)




plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('(Linear Regression)')
plt.xlabel('R & D Spend')
plt.ylabel('Salary')
plt.show()



#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_X.transform(Y)
X
Y



from sklearn.svm import SVR
svr_reg = SVR(kernel='linear', C=100, gamma='auto')
svr_reg.fit(X, Y)



plt.scatter(X,Y,color='red')
plt.plot(X,svr_reg.predict(X),color='blue')
plt.title('(Support Vector Regression)')
plt.xlabel('R & D Spend')
plt.ylabel('Salary')
plt.show()



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print(regressor.score(X_train,Y_train))
print(regressor.score(X_test,Y_test))
SVR_train_4 = []
SVR_train_4.append(regressor.score(X_train,Y_train))
SVR_test_4 = []
SVR_test_4.append(regressor.score(X_test,Y_test))



#Decision Tree in dataset3.csv
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
dataset3 = pd.read_csv('dataset3.csv')
X = dataset3.iloc[:,0:1].values
X




Y = dataset3.iloc[:,-1].values
Y



from sklearn.model_selection import train_test_split # Import train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)



from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
from pydot import graph_from_dot_data
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data)#), feature_names=iris.feature_names)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



y_pred = dt.predict(X_test)
y_pred



dt.score(X_train,y_train)



dt.score(X_test,y_test)



dt_train_3 = []
dt_train_3.append(dt.score(X_train,y_train))
dt_test_3 = []
dt_test_3.append(dt.score(X_test,y_test))



plt.scatter(X,Y,color='red')
plt.plot(X,dt.predict(X),color='blue')
plt.title('(decision Tree)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()



#Decision Tree in dataset4.csv
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
dataset4 = pd.read_csv('dataset4.csv')
X = dataset4.iloc[:,:-1].values
Y = dataset4.iloc[:,-2].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transform = ColumnTransformer([("State data",OneHotEncoder(),[3])],remainder='passthrough')
X = transform.fit_transform(X)
X



Y



X= X[:,3:4]  # R & D Spend
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)
Y



Y_train



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


dt_train_4 = []
dt_train_4.append(dt.score(X_train,Y_train))
dt_test_4 = []
dt_test_4.append(dt.score(X_test,Y_test))



plt.scatter(X,Y,color='red')
plt.plot(X,dt.predict(X),color='blue')
plt.title('(decision Tree)')
plt.xlabel('Salary')
plt.ylabel('City')
plt.show()












#Random Forest in dataset3.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset3 = pd.read_csv('dataset3.csv')
X = dataset3.iloc[:,:-1].values
Y = dataset3.iloc[:,-1].values
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,bootstrap = True, max_features = 'sqrt')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Fit on training data
model.fit(X_train, Y_train)



y_pred = model.predict(X_test)
y_pred



Y_test



model.score(X_train,Y_train)



model.score(X_test,Y_test)



RF_train_3 = []
RF_train_3.append(model.score(X_train,Y_train))
RF_test_3 = []
RF_test_3.append(model.score(X_test,Y_test))



plt.scatter(X,Y,color='red')
plt.plot(X,model.predict(X),color='blue')
plt.title('(Random Forest Classification)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()






#Random Forest in dataset4.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset4 = pd.read_csv('dataset4.csv')
X = dataset4.iloc[:,:-1].values
Y = dataset4.iloc[:,-2].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transform = ColumnTransformer([("State data",OneHotEncoder(),[3])],remainder='passthrough')
X = transform.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Fit on training data
model.fit(X_train, Y_train)




y_pred = model.predict(X_test)
y_pred



Y_test



model.score(X_train,Y_train)



model.score(X_test,Y_test)



RF_train_4 = []
RF_train_4.append(model.score(X_train,Y_train))
RF_test_4 = []
RF_test_4.append(model.score(X_test,Y_test))
plt.scatter(X[:,3:4],Y,color='red')
plt.plot(X[:,3:4],model.predict(X),color='blue')
plt.title('(Random Forest Classification)')
plt.xlabel('City')
plt.ylabel('Salary')
plt.show()






print("==============dataset-3===============")
print("-----------Training Accurecy----------")
print("dataset-3(Multi Linear Regression)(Training)",mult_lin_train_3)
print("dataset-3(Polynomial Regression)(Training)",poly_train_3)
print("dataset-3(Support Vector Regression)(Training)",SVR_train_3)
print("dataset-3(Decision Tree)(Training)",dt_train_3)
print("dataset-3(Random Forest)(Training)",RF_train_3)
print("-----------Testing Accurecy----------")
print("dataset-3(Multi Linear Regression)(Testing)",mult_lin_test_3)
print("dataset-3(Polynomial Regression)(Testing)",poly_test_3)
print("dataset-3(Support Vector Regression)(Testing)",SVR_test_3)
print("dataset-3(Decision Tree)(Testing)",dt_test_3)
print("dataset-3(Random Forest)(Testing)",RF_test_3)
print("==============dataset-4===============")
print("-----------Training Accurecy----------")
print("dataset-4(Multi Linear Regression)(Training)",mult_lin_train_4)
print("dataset-4(Polynomial Regression)(Training)",poly_train_4)
print("dataset-4(Support Vector Regression)(Training)",SVR_train_4)
print("dataset-4(Decision Tree)(Training)",dt_train_4)
print("dataset-4(Random Forest)(Training)",RF_train_4)
print("-----------Testing Accurecy----------")
print("dataset-4(Multi Linear Regression)(Testing)",mult_lin_test_4)
print("dataset-4(Polynomial Regression)(Testing)",poly_test_4)
print("dataset-4(Support Vector Regression)(Testing)",SVR_test_4)
print("dataset-4(Decision Tree)(Testing)",dt_test_4)
print("dataset-4(Random Forest)(Testing)",RF_test_4)



