#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


# In[14]:


time_data = pd.read_csv("delivery_time.csv")
time_data.columns= ["Delivery_Time","Sorting_Time"]
# time_data.set_index(keys="Delivery_Time",inplace = True)
time_data.head(5)


# Checking if data is linear
# 

# In[15]:


plt.scatter(x=time_data.Sorting_Time,y=time_data.Delivery_Time)
plt.xlabel("Sorting Data")
plt.ylabel("Delivery Time")
plt.show()


# In[16]:


sns.lmplot(x="Sorting_Time",y="Delivery_Time",data=time_data)


# Checking if data is normally distributed
# 

# In[17]:


sns.distplot(time_data)
plt.show()


# In[18]:


stats.probplot(x=time_data["Sorting_Time"],plot=plt)
plt.show()


# In[19]:


stats.probplot(x=time_data["Delivery_Time"],plot=plt)
plt.show()


# In[20]:


time_data.corr()


# Model building
# 

# In[21]:


import statsmodels.formula.api as smf
model1 = smf.ols("Delivery_Time~Sorting_Time",data=time_data).fit()
model1.params


# In[22]:


model1.summary()


# Since 1st model's RSquare is not satisfactory..we need to improve our model.
# 

# In[23]:


model2 = smf.ols("Delivery_Time~np.log(Sorting_Time)",data=time_data).fit()
model2.summary()


# In[24]:


model3 = smf.ols("np.log(Delivery_Time)~Sorting_Time",data=time_data).fit()
model2.summary()


# In[26]:


time_data["Squared_X"] = time_data.Sorting_Time ** 2
model4 = smf.ols("np.log(Delivery_Time)~Sorting_Time + Squared_X", data=time_data).fit()
model2.summary()


# RSquare is not improving for any of the model so we can stop here.
# 

# Predict for new data
# 

# In[27]:


newdata=pd.Series([30,40])
data_predict=pd.DataFrame(newdata,columns = ['Sorting_Time'])
data_predict


# In[28]:


model3.predict(data_predict)


# 2) Salary_hike -> Build a prediction model for Salary_hike
# 

# In[29]:


salary_data = pd.read_csv("Salary_Data.csv")
salary_data.head(5)


# In[30]:


salary_data.isna().sum()


# Checking correlation of features
# 

# In[31]:


salary_data.corr()


# In[32]:


##checking Linearity
sns.lmplot(x="YearsExperience",
    y="Salary",
    data=salary_data)


# In[33]:


plt.scatter(x=salary_data.YearsExperience,y=salary_data.Salary)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()


# Normality Test
# 

# In[34]:


sns.distplot(salary_data)


# In[35]:


salary_data.skew()


# In[36]:


stats.probplot(salary_data["YearsExperience"],plot = plt)
plt.xlabel("YearsExperience")
plt.show()


# In[37]:


stats.probplot(salary_data["Salary"],plot = plt)
plt.xlabel("Salary")
plt.show()


# Model Building for predicting Salary
# 

# In[39]:


import statsmodels.formula.api as smf
model = smf.ols("Salary~YearsExperience",data=salary_data).fit()
model.params


# In[40]:


model.summary()


# Predicting for new data
# 

# In[41]:


test_data = [3.5,4.8,10.5]
pred_data = pd.DataFrame(test_data,columns=["YearsExperience"])
pred_data


# In[42]:


pred_data = model.predict(pred_data)
pred_data


# In[ ]:




