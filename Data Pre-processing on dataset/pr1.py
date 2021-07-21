#our need is to preprocess the dataset to handle data in ds&ML
#1st need => NaN  : fill up (how to imput missing value)
#2nd need => how to handle catagorical dataset (ex: India + 2)
#we have seen fillna(method=ffill /bfill)
#fillna(value = 20/ df.mean()) 
import numpy as np
import pandas as pd
dataset = pd.read_csv('dataset1.csv')
dataset.shape #(10,4)
dataset.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Country    10 non-null     object 
 1   Age        9 non-null      float64
 2   Salary     9 non-null      float64
 3   Purchased  10 non-null     object 
dtypes: float64(2), object(2)
memory usage: 448.0+ bytes
"""
#left to right : 0 1 2 3 
#right to left : -4 -3 -2 -1 

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
#after from libarary and after import it is class
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
#all strategy have to apply
imputer = imputer.fit(x [: , 1:3])
x[:,1:3] = imputer.transform(x [: , 1:3])
x

imputer = SimpleImputer(missing_values = np.nan,strategy = 'median')
imputer = imputer.fit(x [: , 1:3])
x[:,1:3] = imputer.transform(x [: , 1:3])
x

imputer = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
imputer = imputer.fit(x [: , 1:3])
x[:,1:3] = imputer.transform(x [: , 1:3])
x

imputer = SimpleImputer(missing_values = np.nan,strategy = 'constant')
imputer = imputer.fit(x [: , 1:3])
x[:,1:3] = imputer.transform(x [: , 1:3])
x


#-------------------------------------------------------------------------------

"""
=>    Data PreProccessing : 
there are three way to convert catagorical data into numerical data:
1.LabelEncoder(comparable)
2.OneHotEncoder(nonComparable)
3.CountVectorizer
-----------------------------
Example :
1...
Small < Med < Large < ELarge <EELarge
0<1<2<3<4
this is call Label Encoder
2....
India USA Canada AUS
   1   2   3       0
not Camparable OneHot Encoder
3...
900000 - students data is their
we use counterVectorizer
"""
from sklearn.preprocessing import LabelEncoder
#creating reference varable
LabelEncoder_x = LabelEncoder()
x[:,0] = LabelEncoder_x.fit_transform(x[:,0])
print(x)
"""
Output:
 [[0 44.0 72000.0]
 [2 27.0 48000.0]
 [1 30.0 54000.0]
 [2 38.0 61000.0]
 [1 40.0 nan]
 [0 35.0 58000.0]
 [2 nan 52000.0]
 [0 48.0 79000.0]
 [1 50.0 83000.0]
 [0 37.0 67000.0]]
"""
from sklearn.model_selection import train_test_split
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state= 0)
#random state = 0 ,.....but why

"""♣
100000000000 + 1
100000000000 >>>> 1 (we neglate)
= 100000000000

very very large value compare to other column with small value
means we have to apply feature scaling to
convert large value to samll vale

2 way are there:
    
1..MinMaxScaler
example: 2  4   1   6 
    min_max = (x - xmin) / (xmax - xmin)
2..Standard Scaler
StandardScaler = (x-u)/s
               = (2-3.25)/___
where u = mean value
      s = standard deviation

"""

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)




x = dataset.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
x = dataset.iloc[:,:].values
transform =  ColumnTransformer([("Column 0 converted",OneHotEncoder(),[0])], remainder = 'passthrough')
x = transform.fit_transform(x)
transform =  ColumnTransformer([("Column 4 converted",OneHotEncoder(),[4])], remainder = 'passthrough')
x = transform.fit_transform(x)
transform =  ColumnTransformer([("Outlook_OL0_OL1",OneHotEncoder(),[6])], remainder = 'passthrough')
x = transform.fit_transform(x)
print(x.astype(int))


