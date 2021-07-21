import pandas as pd
dataset = pd.read_csv('zoo.csv')
print(type(dataset))
dataset.dtypes

#by defalut value for head and tail is 5
print(dataset.head())
print(dataset.head(7))
print(dataset.tail())
print(dataset.tail(3))

#sample gives random value
print(dataset.sample())
print(dataset.sample(3))

#info gives detail about columns
print(dataset.info())

#Pandas describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values.
print(dataset.describe())


dataset1 = pd.DataFrame([['elephant','vegetable'],['tiger','meat'],['kangaroo','vegetables'],['zebra','vegetable'],['dog','meal']],columns=['animal','food'])
dataset[['animal','water_need']]

dataset.merge(dataset1)

dataset.merge(dataset1, how='outer')
dataset.merge(dataset1, how='left')
dataset.merge(dataset1, how='inner')


dataset.sort_values('water_need')
dataset.merge(dataset1, how='outer')

dataset.sort_values('water_need',ascending=False)
dataset.sort_values('water_need',ascending=False, inplace=True)
print(dataset)
dataset.sort_values('water_need',ascending=False).reset_index()
dataset.merge(dataset1, how='outer').fillna('MISSING')

"""Aggregation
We have looked at some aggregation functions in the article so far,
 such as mean, mode, and sum. 
 These perform statistical operations on a set of data. 
 Have a glance at all the aggregate functions in the Pandas package:
     print(dataset.mean())
     print(dataset.groupby('animal').count())
     """
#count() – Number of non-null observations
print(dataset.mean())
print(dataset.groupby('animal').count())

print(dataset.agg({"animal": "count"}))
print(dataset.groupby("animal").agg({"animal": "count"}))
#sum() – Sum of values
print(dataset.agg({"water_need": "sum"}))
#mean() – Mean of values
print(dataset.agg({"water_need": "mean"}))
#median() – Arithmetic median of values
print(dataset.agg({"water_need": "median"}))
#min() – Minimum
print(dataset.agg({"water_need": "min"}))
#max() – Maximum
print(dataset.agg({"water_need": "max"}))
#mode() – Mode
print(dataset.agg({"water_need": "mode"}))
#std() – Standard deviation
print(dataset.agg({"water_need": "std"}))
#var() – Variance
print("variance of water_need : ")
print(dataset.agg({"water_need": "var"}))

