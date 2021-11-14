#!/usr/bin/env python
# coding: utf-8

# # # GRIP: The Sparks Foundation
# 
# Data Science and business Analytics Intern
# 
# **Author** : KUMAR AYUSH
# 
# 
# **Task 1:**
# Prediction using Supervised ML
# 
# **Problem Statement:**Predict the percentage of an studentbased on the no. of study hours.

#  **Importing the required modules**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Reading the data**

# In[2]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)


# **Exploring Data**

# In[3]:


print(data.shape)
data.head()


# In[4]:


data.describe()


# In[5]:


data.describe()


# In[6]:


data.plot(kind='scatter' ,x='Hours',y='Scores')
plt.show()


# In[7]:


data.corr(method='pearson')


# In[8]:


data.corr(method='spearman')


# In[11]:


hours=data['Hours']
scores=data['Scores']


# In[12]:


sns.distplot(hours)


# In[13]:


sns.distplot(scores)


# **Linear Regression**

# In[24]:


X=data.iloc[:,:-1].values
y=data.iloc[:, 1].values


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=50)


# In[26]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, y_train)


# In[27]:


m=reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[28]:


y_pred=reg.predict(X_test)


# In[30]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[32]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# **What would be the predicted score if a student studies for 9.25 hours/day?** 

# In[34]:


h=9.25
s=reg.predict([[h]])
print("If a student studies for{} hours per day he/she will score {} % in exam." .format(h,s))


# **Model Evaluation**

# In[36]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('R2 Score:',r2_score(y_test,y_pred))


# In[ ]:




