#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv("Housing.csv")


# In[29]:


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


columns_to_scale = ['area', 'price']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# In[30]:


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


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

numeric_columns = ['price', 'area']

for col in numeric_columns:
    sns.boxplot(x=df[col])
    plt.title(f'Box plot for {col} (Original Data)')
    plt.show()

for col in numeric_columns:
    sns.boxplot(x=newdf[col])
    plt.title(f'Box plot for {col} (Data without Outliers)')
    plt.show()


# In[ ]:





# In[33]:


feauture_selction = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'price', 'parking', 'airconditioning']
df = df[feauture_selction]


# In[34]:


bedroom_counts = df['bedrooms'].value_counts()
plt.figure(figsize=(8, 6))
bedroom_counts.plot(kind='bar', color='green', alpha=0.7)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.title('Number of Bedrooms Distribution')
plt.savefig('vis.png')
plt.show()


# In[35]:


bedroom_counts = df['bathrooms'].value_counts()
plt.figure(figsize=(8, 6))
bedroom_counts.plot(kind='bar', color='green', alpha=0.7)
plt.xlabel('Number of bathrooms')
plt.ylabel('Count')
plt.title('Number of bathrooms Distribution')
plt.show()


# In[36]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['mainroad'].value_counts().plot(kind='bar', ax=axes[0], color='blue', alpha=0.7)
df['guestroom'].value_counts().plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
axes[0].set_xlabel('Main Road')
axes[1].set_xlabel('Guest Room')
axes[0].set_ylabel('Count')
axes[1].set_ylabel('Count')
axes[0].set_title('Main Road Distribution')
axes[1].set_title('Guest Room Distribution')
plt.show()


# In[37]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['parking'].value_counts().plot(kind='bar', ax=axes[0], color='purple', alpha=0.7)
df['airconditioning'].value_counts().plot(kind='bar', ax=axes[1], color='cyan', alpha=0.7)
axes[0].set_xlabel('Parking')
axes[1].set_xlabel('Air Conditioning')
axes[0].set_ylabel('Count')
axes[1].set_ylabel('Count')
axes[0].set_title('Parking Distribution')
axes[1].set_title('Air Conditioning Distribution')
plt.show()


# In[38]:


grouped = df.groupby(['bedrooms', 'bathrooms']).size().unstack()
grouped.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.title('Number of Bedrooms and Bathrooms Distribution')
plt.show()


# In[39]:


grouped = df.groupby(['stories', 'mainroad']).size().unstack()
grouped.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Number of Stories')
plt.ylabel('Count')
plt.title('Number of Stories and Main Road Distribution')
plt.show()


# In[ ]:




