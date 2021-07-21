import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset5.csv')

#just for check

dataset.shape
dataset.info()

#-------------------
#split data set

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,-1].values


#regression start here
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluffn(Linear Regression')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()


#polynomial

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
print(x_poly)
print(x)

#graph of simple linear reg

plt.scatter(x , y, color = 'red')
plt.plot(x, lin_reg.predict(x),color= 'blue')
plt.title('Truth opf Bluffn(Linear Regression')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()

## run degree - 2 cell and then run
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
plt.scatter(x , y, color = 'red')

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(y)),color='blue')
plt.title('Truth or Bluffn(Polynomial Regression')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()






from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_4.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or Bluffn(Polynomial Regression')
plt.xlabel('position label')
plt.ylabel('salary')
plt.show()



print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))



