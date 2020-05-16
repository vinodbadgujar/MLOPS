#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:
url = "https://raw.githubusercontent.com/vinodbadgujar/MLOPS/master/SalaryData.csv"
dataset = pd.read_csv(url, error_bad_lines=False)
# In[3]:

# In[5]:





# In[6]:





# In[7]:


y=dataset['Salary']


# In[8]:


x=dataset['YearsExperience']


# In[9]:


X=x.values.reshape(30,1)


# In[11]:





# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train ,x_test,y_train ,y_test = train_test_split(X,y,test_size=0.3, random_state=42)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


mind = LinearRegression()


# In[16]:


mind.fit(x_train,y_train)


# In[17]:


y_prediction = mind.predict(x_test)


# In[18]:





# In[19]:





# In[21]:





# In[23]:





# In[25]:




# In[26]:





# In[27]:





# In[29]:


from sklearn import metrics


# In[31]:


# In[32]:

acc = print(metrics.mean_squared_error(y_test,y_prediction))



# In[35]:


import joblib as jb


# In[36]:
jb.dump(mind,"/pyfiles/SalaryPredictor.pk1")
print("model train successfully with accuracy :",acc)


# In[ ]:




