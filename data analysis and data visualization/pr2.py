

import pandas as pd

df= pd.read_excel('Covid cases in India.xlsx')

df1= pd.read_excel('Covid cases in India.xlsx','Sheet2')


columns = df.columns
columns


df.columns[1:3]

df.TotalConfirmedCases

df[['TotalConfirmedCases','Active']]


df.columns[1:3]
df
new_df= df[df.columns[1:3]]

new_df

df[['Name of State / UT','TotalConfirmedCases']]

new_df.head()

new_df = df
new_df
new_df.shape


new_df = df[10:16]
new_df
new_df.shape

df['Active']

df['Active'] > 500

df.loc[df['Active'] >500, ['Name of State / UT','Active','Deaths']]

df.loc[df['Active'] >500] [['Name of State / UT','Active','Deaths']]
df['Active'] > 500
df.iloc[(df['Active']> 500).values & (df['Deaths']> 100).values]

df.iloc[(df['Active']> 500).values & (df['Deaths']> 100).values,[1,3,5]]

#exercise : 
#1. diplay 5 state details with maximun deaths
#print(dataset.agg({"water_need": "max"})) 
df1.sort_values('Death',ascending=False, inplace=True)
df1[['Name of State / UT','Death']].head()


#2. diplay 5 state details with minimum deaths
df1.sort_values('Death', inplace=True)
df1[['Name of State / UT','Death']].head()
#another way
df1.nsmallest[5,'Deaths']

#3. find out total confrimrd cases accross all the states
print(df.groupby("TotalConfirmedCases").agg({"TotalConfirmedCases": "count"}))
print(df.agg({"TotalConfirmedCases": "sum"}))

df['Total_Cases']= df['Active']+df['Recovered']+df['Deaths']
df


ndf = df
ndf.head()
ndf['Active'] = ndf['Active'].astype(int)
ndf.head()
ndf = df.fillna(value=df.median())
ndf.head()
ndf.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
df.reset_index(drop=True, inplace=True)
df
sns.barplot(x="Name of State / UT", y="Active", data=df)
df = df[:20]
df
plt.figure(figsize=(16,8))
sns.barplot(x="Name of State / UT", y="Active", data=df)
plt.xticks(rotation=70, horizontalalignment='right', fontweight='light', fontsize='x-large')
plt.show()
df.columns
plt.figure(figsize=(16,8))
sns.stripplot(x="Name of State / UT", y="Active", data=df)
plt.xticks(rotation=70, horizontalalignment='right', fontweight='light', fontsize='x-large')
plt.show()
plt.figure(figsize=(24,12))
sns.barplot( x=df["Active"], y=df["Name of State / UT"], color="pink", label="Total Cases")
sns.barplot( x=df["Recovered"], y=df["Name of State / UT"], color="lightgreen", label="Recovered")
plt.legend()
