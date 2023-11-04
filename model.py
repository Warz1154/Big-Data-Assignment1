#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Housing.csv")


# In[2]:


df


# In[3]:


df.info()


# In[4]:


one_hot_encode = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
convert = ['furnishingstatus', 'bathrooms', 'stories', 'parking', 'bedrooms']
df = pd.get_dummies(df, columns=convert)
for col in one_hot_encode:
    df[col] = df[col].map({'yes': 1, 'no': 0})


# In[5]:


df


# In[6]:


#from sklearn.preprocessing import OneHotEncoder

#one_hot_encode = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
#convert = ['furnishingstatus', 'bathrooms', 'stories', 'parking', 'bedrooms']
#encoder = OneHotEncoder(sparse=False)
#encoded_features = encoder.fit_transform(df[one_hot_encode])
#encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(one_hot_encode))
#df = df.drop(columns=one_hot_encode)
#df = pd.get_dummies(df, columns=convert)
#df = pd.concat([df, encoded_df], axis=1)
#print(df)


# In[7]:


df


# In[ ]:





# In[8]:


scaler = StandardScaler()

columns_to_scale = ['area', 'price']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# In[ ]:





# In[ ]:





# In[9]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


from scipy import stats
z_threshold = 3
newdf = df.copy()
for column in df.columns:
    if df[column].dtype in [int, float]:
        z_scores = stats.zscore(df[column])
        outlier_indices = (z_scores > z_threshold) | (z_scores < -z_threshold)
        newdf = newdf[~outlier_indices]
newdf.reset_index(drop=True, inplace=True)
print(newdf)


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10,6]
print('\033[1mCorrelation Matrix'.center(100))
plt.figure(figsize=[25,20])
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center=0)
plt.show()


# In[12]:


import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


target_column = 'price'
features = df.drop(target_column, axis=1)
target = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

select_k_best = SelectKBest(score_func=mutual_info_regression, k=5)

X_train_new = select_k_best.fit_transform(X_train_scaled, y_train)

selected_features = X_train.columns[select_k_best.get_support()]
print("Selected Features: ", selected_features)


# In[13]:


df


# In[14]:


import pandas as pd
from sklearn.cluster import KMeans

df = df.drop('price', axis=1)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df)

cluster_labels = kmeans.labels_

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

cluster_counts.to_csv('k.txt', header=False, index=False, sep='\t')

print("Cluster Counts:")
print(cluster_counts)


# In[ ]:





# In[ ]:




