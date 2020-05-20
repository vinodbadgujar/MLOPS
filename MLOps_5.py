#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# In[2]:
#dataset = pd.read_csv(url, error_bad_lines=False
dataset = pd.read_csv("/programs/SalaryData.csv")



y=dataset['Salary']


x=dataset['YearsExperience']




X=x.values.reshape(30,1)





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



from sklearn import metrics


# In[31]:


# In[32]:

metrics.mean_squared_error(y_test,y_prediction)
acc= np.mean((abs(y_test)-abs(y_prediction))/2)



# In[35]:


import joblib as jb


# In[36]:
jb.dump(mind,"/pyfiles/SalaryPredictor.pk1")
print("model train successfully with accuracy :", acc)


# In[ ]:




