#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import sklearn.linear_model as sk
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report


# In[3]:


data = pd.read_csv("D:\done\logistic\creditcard.csv")


# In[4]:


data.card.replace(('yes', 'no'), (1, 0), inplace=True)
data.owner.replace(('yes', 'no'), (1, 0), inplace=True)
data.selfemp.replace(('yes', 'no'), (1, 0), inplace=True)


# In[5]:


data.rename(columns={'Unnamed: 0':'Sln'},inplace=True)


# In[6]:


data.drop(["Sln"],inplace=True,axis=1)         #drops sln column
data.head()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe()


# In[9]:


data['card'].value_counts()


# In[10]:


sns.countplot(x='owner',data=data,palette='hls')
pd.crosstab(data.owner,data.card).plot(kind="bar")


# In[11]:


sns.countplot(x='selfemp',data=data,palette='hls')
pd.crosstab(data.selfemp,data.card).plot(kind="bar")


# In[12]:


sns.countplot(x='majorcards',data=data,palette='hls')
pd.crosstab(data.majorcards,data.card).plot(kind="bar")


# In[13]:


sns.countplot(x='active',data=data,palette='hls')
pd.crosstab(data.active,data.card).plot(kind="bar")


# In[14]:


sns.countplot(x='card',data=data,palette='hls')


# In[15]:


sns.boxplot(x="owner",y="card",data=data,palette='hls')


# In[16]:


sns.boxplot(x="selfemp",y="card",data=data,palette='hls')


# In[17]:


sns.boxplot(x="majorcards",y="card",data=data,palette='hls')


# In[18]:


sns.boxplot(x="active",y="card",data=data,palette='hls')


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


x=data.iloc[:,1:12]


# In[21]:


y=data.iloc[:,0]


# In[40]:


trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=2)


# In[41]:


classifier = LogisticRegression()


# In[42]:


classifier.fit(trainx,trainy)


# In[43]:


classifier.coef_


# In[45]:


probs=classifier.predict_proba(x)


# In[26]:


y_pred = classifier.predict(x)


# In[27]:


data["y_pred"]=y_pred


# In[28]:


y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))


# In[29]:


new_df = pd.concat([data,y_prob],axis=1)


# In[30]:


from sklearn.metrics import confusion_matrix


# In[31]:


confusion_matrix = confusion_matrix(y,y_pred) # confusion matrix is to evaluate the performance of the model


# In[32]:


print (confusion_matrix)
type(y_pred)


# In[33]:


accuracy = sum(y==y_pred)/data.shape[0]
pd.crosstab(y_pred,y)
# 1 and 23 are incorrect decisions, 295 and 1000 are correct decisions.
# 295 times error not identified cases are correctly predicted,1000 times error identified cases are correctly identified,1 time the error is identified but our model hasn't identified it,23 times error is identifed and model has identified it too 


# In[48]:


corr = data.corr() # .corr is used to find corelation
f,ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, cbar = True,  square = True, annot = False, fmt= '.1f', 
            xticklabels= True, yticklabels= True
            ,cmap="coolwarm", linewidths=.5, ax=ax)
plt.title('CORRELATION MATRIX - HEATMAP', size=18);


# In[ ]:




