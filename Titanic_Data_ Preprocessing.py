#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[78]:


titanic = pd.read_csv(r"D:\NIT\8 nov (ml)\8th- ML\overifitting, underfitting, pca\TASK-13\DATASET\train.csv")


# In[79]:


titanic


# In[80]:


titanic.tail()


# In[81]:


titanic.describe()


# In[82]:


#Name column can never decide survival of a person, hence we can safely delete it
del titanic["Name"]
titanic.head()


# In[83]:


del titanic["Ticket"]
titanic.head()


# In[84]:


del titanic["Fare"]
titanic.head()


# In[85]:


del titanic['Cabin']
titanic.head()


# In[86]:


# Changing Value for "Male, Female" string values to numeric values , male=1 and female=2
def getNumber(str):
    if str=="male":
        return 1
    else:
        return 2
titanic["Gender"]=titanic["Sex"].apply(getNumber)


# In[87]:


titanic.head()


# In[88]:


#Deleting Sex column, since no use of it now
del titanic["Sex"]
titanic.head()


# In[89]:


titanic.isnull().sum()


# In[90]:


titanic["Age"]=titanic["Age"].fillna(np.mean(pd.to_numeric(titanic["Age"])))


# In[91]:


titanic.isnull().sum()


# In[92]:


titanic["Embarked"]=titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])


# In[93]:


titanic.isnull().sum()


# In[94]:


titanic.head()


# In[95]:


titanic['Gender'].unique()


# In[96]:


titanic['Age'].unique()


# In[97]:


titanic.rename(columns={'Gender':'Sex'}, inplace=True)
titanic.head()


# In[98]:


def getEmb(str):
    if str=="S":
        return 1
    elif str=='Q':
        return 2
    else:
        return 3
titanic["Embark"]=titanic["Embarked"].apply(getEmb)
titanic.head()


# In[99]:


del titanic['Embarked']
titanic.rename(columns={'Embark':'Embarked'}, inplace=True)
titanic.head()


# In[118]:


#Drawing a pie chart for number of males and females aboard
import matplotlib.pyplot as plt
from matplotlib import style


# In[105]:


males = (titanic['Sex'] == 1).sum() 
#Summing up all the values of column gender with a 
#condition for male and similary for females


# In[106]:


males


# In[107]:


females = (titanic['Sex'] == 2).sum()


# In[108]:


females


# In[109]:


p = [males, females]
plt.pie(p,labels = ['Male', 'Female'])#Correspndingly giving labels


# In[110]:


p = [males, females]
plt.pie(p,labels = ['Male', 'Female'], colors = ['green', 'yellow']) # Corresponding colors


# In[111]:


plt.pie(p,labels = ['Male', 'Female'], colors = ['green', 'yellow'],explode = (0.15, 0),startangle = 0)
 #How much the gap should me there between the pies, what start angle should be given


# In[112]:


plt.pie(p,labels = ['Male', 'Female'], colors = ['green', 'yellow'],explode = (0.15, 0),startangle = 0)
plt.axis('equal') 


# In[113]:


# More Precise Pie Chart
MaleS=titanic[titanic.Sex==1][titanic.Survived==1].shape[0]
print(MaleS)
MaleN=titanic[titanic.Sex==1][titanic.Survived==0].shape[0]
print(MaleN)


# In[114]:


FemaleS=titanic[titanic.Sex==2][titanic.Survived==1].shape[0]
print(FemaleS)
FemaleN=titanic[titanic.Sex==2][titanic.Survived==0].shape[0]
print(FemaleN)


# In[115]:


chart=[MaleS,MaleN,FemaleS,FemaleN]
colors=['lightskyblue','yellowgreen','Yellow','Orange']
labels=["Survived Male","Not Survived Male","Survived Female","Not Survived Female"]
explode=[0,0.05,0,0.1]


# In[116]:


plt.pie(chart,labels=labels,colors=colors,explode=explode,counterclock=False)
plt.axis("equal")


# In[117]:


plt.pie(chart,labels=labels,colors=colors,explode=explode,startangle=100,counterclock=False,autopct="%.2f%%")
plt.axis("equal")

