# Understanding of Data Pre-processing for given dataset 1 using Spyder (Python)
import numpy as np
import pandas as pd

dataset= pd.read_csv('dataset1.csv')


dataset.shape

dataset.info()
X=dataset.iloc[:,:-1].values
X

Y=dataset.iloc[:,-1].values
Y



# 2. Replace Missing values by below imputation strategy.

from sklearn.impute import SimpleImputer
X = dataset.iloc[:,:-1].values 
imputer= SimpleImputer(missing_values = np.nan , strategy = 'mean') #mean
imputer= imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
X


X = dataset.iloc[:,:-1].values 
imputer= SimpleImputer(missing_values = np.nan , strategy = 'median') #median
imputer= imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
X


X = dataset.iloc[:,:-1].values 
imputer= SimpleImputer(missing_values = np.nan , strategy = 'most_frequent') #mostfrequent
imputer= imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])
X


X = dataset.iloc[:,:-1].values 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant',fill_value='IshikaShah') # using constant
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])
X

X = dataset.iloc[:,:-1].values  # again storing a value dataset to X
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # using mean
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])
X
Y

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X

"""
suppose, data = 5, 2, 1, 6, 3

here MinMax Scaler is

Min_Max_Scaler = x - minx / (maxx - minx)
here in our case = (5-1)/(6-1)=4/5 ~= 0.8 and like wise for other data! 0.8, 0.2, 0, 1, 0.4

similarly for standard scaler,

Standard_Scaler = (x-u) / s
u = mean = 17/5 = 3.4

s = standard daviation = sqrt ((1.6^2 + 1.4^2 + ....)/5)"""


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2,random_state= 0)

# Feature Scaling by Standard Scaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X_train

X_test



# Feature Scaling by MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X_train

X_test




#DATASET2

dataset2 = 'dataset2.csv'
ds = pd.read_csv(dataset2)
ds

x = ds.iloc[:,:].values
x

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transform =  ColumnTransformer([("Col 0",OneHotEncoder(),[0])],remainder = 'passthrough')
x = transform.fit_transform(x)
x





transform =  ColumnTransformer([("Column 4 converted",OneHotEncoder(),[4])],remainder = 'passthrough')
x = transform.fit_transform(x)
x



transform =  ColumnTransformer([("Outlook_OL0_OL1",OneHotEncoder(),[6])],remainder = 'passthrough')
x = transform.fit_transform(x)
print(x.astype(int))



