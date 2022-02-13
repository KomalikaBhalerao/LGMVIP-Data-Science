#!/usr/bin/env python
# coding: utf-8

# # Name: Komalika Bhalerao
# # LGMVIP Task-3
# # Prediction Using Decision Tree Algorithm

# In[1]:


import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# In[4]:


Iris = pd.read_csv(r'C:\Users\sushmita\Downloads\Iris.csv')


# In[5]:


Iris.head()


# In[6]:


Iris.info()


# In[7]:


Iris.describe()


# ### Visualizing the data

# In[16]:


plt.scatter(Iris['SepalLengthCm'],Iris['SepalWidthCm'])


# In[39]:


sns.pairplot(Iris, hue='Species')


# In[42]:


sns.pairplot(Iris, hue="Species", diag_kind ="hist")


# ## Correlation Matrix

# In[34]:


Iris.corr()


# ## Heat Map

# In[36]:


sns.heatmap(Iris.corr(), cmap="RdPu")


# ## Data Processing

# In[44]:


target=Iris['Species']
df=Iris.copy()
df=df.drop('Species', axis=1)
df.shape


# In[48]:


x=Iris.iloc[:, [0,1,2,3]].values
LaEn=LabelEncoder()
Iris['Species']=LaEn.fit_transform(Iris['Species'])
y=Iris['Species'].values
Iris.shape


# ## Training the model
# ## Splitting the data into Training and Testing set

# In[49]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training srt:",x_train.shape)
print("Testing set:",x_test.shape)


# ## Defining Decision Tree Algorithm

# In[51]:


d_tree=DecisionTreeClassifier()
d_tree.fit(x_train,y_train)
print("Decision Tree Classifier created!")


# ## Visualization of Trained Model
