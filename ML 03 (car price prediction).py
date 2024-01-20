#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df = pd.read_excel(r'E:\ML PROJECTS\carprices dummy variables.xlsx')
df


# In[3]:


dummies = pd.get_dummies(df.Car)
dummies


# In[4]:


merged = pd.concat([df,dummies],axis = 'columns')
merged


# In[5]:


final = merged.drop(['Car','Mercedez Benz C class'],axis = 'columns')
final


# In[6]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()

              


# In[7]:


x = final.drop('Price' ,axis= 'columns')
x


# In[8]:


y = final.Price
y


# In[9]:


model.fit(x,y)


# In[10]:


model.predict(x)


# In[11]:


model.score(x,y)


# In[12]:


model.predict([[69000,6,1,0]])


# In[13]:


model.predict([[79000,7,0,0]])


# In[ ]:




