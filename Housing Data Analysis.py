#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv("C:/Users/Neethu Santhosh/Desktop/decoder lectures/case study/train (1).csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


import seaborn as sns


# In[8]:


_=sns.displot(df["SalePrice"])


# In[9]:


total=df.isnull().sum().sort_values(ascending=False)


# In[10]:


total


# In[11]:


percent=df.isnull().sum()/df.shape[0]


# In[13]:


missing_data=pd.concat([total,percent],axis=1,keys=["Total","Percent"])


# In[14]:


missing_data.head(20).sort_index()


# In[15]:


null_has_meaning=["Alley","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
                 "BsmtQual","Fence","FireplaceQu","GarageCond","GarageFinish","GarageQual","GarageType","PoolQC","MiscFeature"]


# In[16]:


for i in null_has_meaning:
    df[i].fillna("None",inplace=True)
    


# In[17]:


total=df.isnull().sum().sort_values(ascending=False)
percent=df.isnull().sum()/df.shape[0]
missing_data=pd.concat([total,percent],axis=1,keys=["Total","Percent"])
missing_data.head(20)


# In[18]:


df.drop("LotFrontage",axis=1,inplace=True)


# In[19]:


df.dtypes


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


var="GarageYrBlt"
data=pd.concat([df["SalePrice"],df[var]],axis=1)
f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=var,y="SalePrice",data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90)


# In[22]:


df["GarageYrBlt"].fillna(df["GarageYrBlt"].median(),inplace=True)
df["MasVnrArea"].fillna(df["MasVnrArea"].median(),inplace=True)
df["MasVnrType"].fillna("None",inplace=True)


# In[23]:


total=df.isnull().sum().sort_values(ascending=False)
percent=df.isnull().sum()/df.shape[0]
missing_data=pd.concat([total,percent],axis=1,keys=["Total","Percent"])
missing_data.head(20)


# In[24]:


df.dropna(inplace=True)


# In[25]:


total=df.isnull().sum().sort_values(ascending=False)
percent=df.isnull().sum()/df.shape[0]
missing_data=pd.concat([total,percent],axis=1,keys=["Total","Percent"])
missing_data.head(20)


# In[26]:


types_train=df.dtypes
num_train=types_train[(types_train=="int64") | (types_train==float)]


# In[27]:


cat_train=types_train[(types_train==object)]


# In[28]:


pd.DataFrame(types_train)[0].value_counts()


# In[29]:


num_train


# In[30]:


numerical_values_train=list(num_train.index)


# In[31]:


num_train


# In[32]:


numerical_values_train


# In[33]:


categorical_values_train=list(cat_train.index)
categorical_values_train


# In[35]:


sns.displot(df["SalePrice"])


# In[36]:


import numpy as np


# In[38]:


sns.displot(np.log(df["SalePrice"]))


# In[39]:


df["TransformedPrice"]=np.log(df["SalePrice"])


# In[40]:


categorical_values_train


# In[41]:


df["MSZoning"].value_counts()


# In[42]:


df[categorical_values_train]


# In[43]:


set(df["MSZoning"])


# In[44]:


for i in categorical_values_train:
    feature_set=set(df[i])
    for j in feature_set:
        feature_list=list(feature_set)
        df.loc[df[i]==j,i]=feature_list.index(j)
        


# In[45]:


df.head()


# In[46]:


X=df.drop(["Id","SalePrice","TransformedPrice"],axis=1)


# In[47]:


y=df["TransformedPrice"]


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


# In[50]:


params={"alpha":[0.0001,0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2,3,4,5,6,7,8,9,10,20,50,100,500,1000]}


# In[51]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[52]:


lasso=Lasso()
folds=5
model_cv=GridSearchCV(estimator=lasso,
                     param_grid=params,scoring="neg_mean_absolute_error",
                      return_train_score=True,
                     cv=folds,verbose=1)
model_cv.fit(X_train,y_train)


# In[53]:


cv_results=pd.DataFrame(model_cv.cv_results_)


# In[54]:


cv_results["param_alpha"]=cv_results["param_alpha"].astype("float32")


# In[55]:


cv_results.dtypes


# In[56]:


plt.plot(cv_results["param_alpha"],cv_results["mean_train_score"])
plt.plot(cv_results["param_alpha"],cv_results["mean_test_score"])
plt.xlabel("alpha")
plt.ylabel("Negative Mean Absolute Error")
plt.legend(["train_score","test_score"],loc="upper left")
plt.show()


# In[57]:


alpha=50
lasso=Lasso(alpha=alpha)
lasso.fit(X_train,y_train)


# In[58]:


lasso.coef_


# In[59]:


ridge=Ridge()
folds=5
model_cv=GridSearchCV(estimator=ridge,
                     param_grid=params,scoring="neg_mean_absolute_error",
                      return_train_score=True,
                     cv=folds,verbose=1)
model_cv.fit(X_train,y_train)


# In[60]:


cv_results=pd.DataFrame(model_cv.cv_results_)


# In[61]:


cv_results.dtypes


# In[62]:


cv_results["param_alpha"]=cv_results["param_alpha"].astype("float32")


# In[63]:


plt.plot(cv_results["param_alpha"],cv_results["mean_train_score"])
plt.plot(cv_results["param_alpha"],cv_results["mean_test_score"])
plt.xlabel("alpha")
plt.ylabel("Negative Mean Absolute Error")
plt.legend(["train_score","test_score"],loc="upper left")
plt.show()


# In[64]:


alpha=10
ridge=Ridge(alpha=alpha)
ridge.fit(X_train,y_train)


# In[65]:


ridge.coef_


# In[ ]:





# In[ ]:




