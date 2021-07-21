import pandas as pd

df= pd.read_excel('pokemon_data.xlsx')

columns = df.columns
columns

df.Speed

df[['Name','Speed']]


df.columns[1:3]
new_df= df[df.columns[1:3]]

new_df.head()

new_df = df

new_df
new_df.shape


new_df = df[10:16]
new_df
new_df.shape

df['Sp. Atk'] > 100


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df= pd.read_excel('pokemon_data.xlsx')
df.reset_index(drop=True,inplace=True)

sns.barplot(x="Name",y="Sp. Atk",data=df)
df = df[:20]
df

plt.figure(figsize=(16,8))
sns.barplot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()

plt.figure(figsize=(16,8))
sns.stripplot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()


plt.figure(figsize=(16,8))
sns.relplot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()

#error
plt.figure(figsize=(16,8))
sns.distlot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()


plt.figure(figsize=(16,8))
sns.catplot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()


plt.figure(figsize=(16,8))
sns.jointplot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()

plt.figure(figsize=(16,8))
sns.pairplot(df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()

plt.figure(figsize=(16,8))
sns.swarmplot(x="Name",y="Sp. Atk",data=df)
plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()

plt.figure(figsize=(16,8))
#plt.pie(x="Name",data=df)
plt.pie(x="Sp. Atk",colors="101010",data=df)
#plt.xticks(rotation=70,horizontalalignment='right',fontweight='light',fontsize='x-large')
plt.show()
