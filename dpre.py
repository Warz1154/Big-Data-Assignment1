#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv("Housing.csv")


# In[12]:


df


# In[13]:


df.info()


# In[14]:


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


# In[ ]:





# In[15]:


one_hot_encode = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
convert = ['furnishingstatus', 'bathrooms', 'stories', 'parking', 'bedrooms']
df = pd.get_dummies(df, columns=convert)
for col in one_hot_encode:
    df[col] = df[col].map({'yes': 1, 'no': 0})


# In[16]:


scaler = StandardScaler()

columns_to_scale = ['area', 'price']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# In[17]:


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


# In[18]:


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


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10,6]
print('\033[1mCorrelation Matrix'.center(100))
plt.figure(figsize=[25,20])
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center=0)
plt.show()


# In[20]:


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


# In[22]:


new_df = df[selected_features]
new_df.to_csv(' res_dpre.csv', index=False)


# In[ ]:




