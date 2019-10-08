#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matrixprofile import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
                                     MinMaxScaler, 
                                     StandardScaler, 
                                     OneHotEncoder, 
                                     LabelEncoder
                                  )

#%%
new_cols = ['date', 'avg_temp', 'min_temp', 'max_temp', 'precipitation_mm', 'weekend', 'total_consumption']
beer_df = pd.read_csv('.\consumo_cerveja.csv', skiprows=1, names=new_cols, index_col=['date'], parse_dates=['date'])

#%%
beer_df['day_of_year'] = beer_df.index.dayofyear
beer_df['day'] = beer_df.index.day
beer_df['week_day'] = beer_df.index.weekday
beer_df['month'] = beer_df.index.month
beer_df['year'] = beer_df.index.year

#%%
beer_df.head()

#%%
beer_df.tail()

#%%
beer_df.info()

#%%
beer_df = beer_df.dropna(how='all')
cols_to_float = ['avg_temp', 'min_temp', 'max_temp', 'precipitation_mm']
beer_df[cols_to_float] = beer_df[cols_to_float].applymap(lambda x: str(x).replace(',', '.')).astype(float)
beer_df.describe()

#%%
beer_df.info()

#%%
beer_df[['avg_temp']].hist()

#%%
beer_df[['min_temp']].hist()

#%%
beer_df[['max_temp']].hist()

#%%
beer_df[['precipitation_mm']].hist()

#%%
avg_consumption_df = beer_df.groupby('weekend')['total_consumption'].mean().reset_index(name='avg_consumption')
total_consumption_df = beer_df.groupby('weekend')['total_consumption'].sum().reset_index(name='sum_consumption')

fig, axes = plt.subplots(1, 2)

sns.barplot('weekend', 'avg_consumption', data=avg_consumption_df, ax=axes[0])
sns.barplot('weekend', 'sum_consumption', data=total_consumption_df, ax=axes[1])

#%%
total_cons_df = beer_df.sort_values('total_consumption', ascending=False).head(92)

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
sns.barplot(x="month", y="total_consumption", data=beer_df, estimator=sum, ax=ax1)
sns.barplot(x='month', y='total_consumption', hue='weekend', data=beer_df, estimator=sum, ax=ax2)

#%%
fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20,10))
sns.barplot(x="week_day", y="total_consumption", data=beer_df, estimator=sum, ax=ax1[0])
sns.barplot(x="week_day", y="total_consumption", data=beer_df, estimator=np.mean, ax=ax1[1])

#%%
liters_per_mm = sum(beer_df.total_consumption)/sum(beer_df.precipitation_mm)
print("{0:.2f}L for every mm of rain during the year.".format(liters_per_mm))

#%%
data = beer_df[['avg_temp','min_temp','max_temp','precipitation_mm', 'weekend']]

#%%
# sns.pairplot(data, hue='weekend')

#%%
fig, axes = plt.subplots(1, 3, figsize=(20,10))
sns.regplot('precipitation_mm', 'total_consumption', beer_df, ax=axes[0])
sns.regplot('min_temp', 'total_consumption', beer_df, ax=axes[1])
sns.regplot('weekend', 'total_consumption', beer_df, ax=axes[2])

#%%
sns.regplot('max_temp', 'total_consumption', data=beer_df)

#%%
weekly_consumption = beer_df.resample('W').sum()
monthly_consumption = beer_df.resample('M').sum()
trimester_consumption = beer_df.resample('3M').sum()

#%%
fig, (ax1,ax2,ax3,ax4) =plt.subplots(4, 1, figsize=(20,10), sharex=True)

ax1.set_ylabel('Total daily consumption')
ax1.plot(beer_df['total_consumption'])

ax2.set_ylabel('Total weekly consumption')
ax2.plot(weekly_consumption['total_consumption'])

ax3.set_ylabel('Total monthly consumption')
ax3.plot(monthly_consumption['total_consumption'])

ax4.set_ylabel('Total trimester consumption')
ax4.plot(trimester_consumption['total_consumption'])

#%%
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(25,10), sharex=True)

fig.suptitle('Moving averages total consumption', fontsize=16)

ax0.plot(beer_df.total_consumption.rolling(window=7).mean())
ax1.plot(beer_df.total_consumption.rolling(window=14).mean())
ax2.plot(beer_df.total_consumption.rolling(window=30).mean())
ax3.plot(beer_df.total_consumption.rolling(window=90).mean())

ylabels = ['Weekly', 'BiWeekly', 'Monthly', 'Trimester']
ax0.set_ylabel(ylabels[0])
ax1.set_ylabel(ylabels[1])
ax2.set_ylabel(ylabels[2])
ax3.set_ylabel(ylabels[3])

#%%
from matrixprofile import matrixProfile
m = 32
mp = matrixProfile.stomp(beer_df['total_consumption'].values,m)
mp_adj = np.append(mp[0],np.zeros(m-1)+np.nan)

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))

ax1.set_ylabel('Total monthly consumption', size=22)
ax1.set_xlabel('Date', size=22)
ax1.plot(beer_df.total_consumption)

ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color='red')
ax2.set_ylabel('Matrix Profile', size=22)
ax2.set_xlabel('Sample', size=22)

#%%
def high_temp(val: float) -> str:
  if val > 17.9:
    return 'high'
  return 'low'

#%%
temps = pd.get_dummies(beer_df.min_temp.apply(high_temp), prefix='min_temp_bucket')
beer_df = beer_df.join(temps)

#%%
one_hot = pd.get_dummies(beer_df.weekend, prefix='weekend')
beer_df = beer_df.join(one_hot)

#%%
lin_reg_df_min_max = beer_df.copy(deep=True)
lin_reg_df_std = beer_df.copy(deep=True)

#%%
cols_to_normalize = ['avg_temp', 'min_temp', 'max_temp', 'precipitation_mm']

#%%
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

min_max_scaler.fit(lin_reg_df_min_max[cols_to_normalize])
lin_reg_df_min_max[cols_to_normalize] = min_max_scaler.transform(lin_reg_df_min_max[cols_to_normalize])

standard_scaler.fit(lin_reg_df_std[cols_to_normalize])
lin_reg_df_std[cols_to_normalize] = standard_scaler.transform(lin_reg_df_std[cols_to_normalize])

#%%
lin_reg_df_std.head()

#%%
lin_reg_df_min_max.head()

#%%
# X, y = lin_reg_df_min_max[['avg_temp', 'max_temp', 'precipitation_mm', 'weekend_0.0', 'weekend_1.0', 'min_temp_bucket_high', 'min_temp_bucket_low']], lin_reg_df_min_max[['total_consumption']]

X, y = lin_reg_df_std[
  [
    'avg_temp', 
    'max_temp', 
    'precipitation_mm', 
    'weekend_0.0', 
    'weekend_1.0', 
    'min_temp_bucket_high', 
    'min_temp_bucket_low']
  ], lin_reg_df_std[['total_consumption']]

#%%
reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)

#%%
from sklearn.metrics import mean_squared_error

y_predict = reg.predict(X_test)

reg_mse = mean_squared_error(y_predict, y_test)

import math

math.sqrt(reg_mse)

#%%
coefs = np.array(reg.coef_).tolist()[0]
pd.DataFrame(list(zip(list(X.columns), coefs)), columns=['features', 'coefs'])

#%%
fig, (ax1) = plt.subplots()
sns.regplot(lin_reg_df_min_max.max_temp, lin_reg_df_min_max.total_consumption, ax=ax1)

#%%
from sklearn.cluster import KMeans

data = np.array(list(zip(beer_df.total_consumption, beer_df.max_temp)))

#%%
fig, axes = plt.subplots()
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(data)
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('WCSS')

#%%
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
y_km = kmeans.fit_predict(data)

#%%
fig, axes = plt.subplots()

plt.scatter(data[y_km ==0,0], data[y_km == 0,1], s=50, c='yellow', marker='s', label='winter')
plt.scatter(data[y_km ==1,0], data[y_km == 1,1], s=50, c='green', marker='v', label='summer')
plt.scatter(data[y_km ==2,0], data[y_km == 2,1], s=50, c='blue',  marker='+', label='fall, spring?')
# plt.scatter(data[y_km ==3,0], data[y_km == 3,1], s=50, c='red',  marker='o', label='summer')

plt.title('Beer consumption by Season')
plt.xlabel('Total Consumption (L)')
plt.ylabel('Temp')
plt.legend()