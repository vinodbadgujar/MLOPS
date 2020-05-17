#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from io import StringIO

# In[2]:
url = requests.get("https://doc-10-5k-docs.googleusercontent.com/docs/securesc/9j0btqiv2g5nk6oitrfdltqohge73gib/efdsqd046vbhhhvimtiojaf2bfi29gvg/1589725875000/15171157401513724834/15171157401513724834/1mEAQNbPUkl646P3FH2EkRdD8Ygo0ijd1?e=download&authuser=0&nonce=05tlkou94b78i&user=15171157401513724834&hash=ip25ebqicqi2nuaaac7q6hm2l5kk43ng") 
#dataset = pd.read_csv(url, error_bad_lines=False)
csv_raw = StringIO(url.text)
dataset = pd.read_csv(csv_raw ,delim_whitespace=True)


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
acc= (abs(y_test)-abs(y_prediction))/2



# In[35]:


import joblib as jb


# In[36]:
jb.dump(mind,"/pyfiles/SalaryPredictor.pk1")
print("model train successfully with accuracy :", acc)


# In[ ]:




