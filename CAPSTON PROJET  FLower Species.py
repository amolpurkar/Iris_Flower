#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[136]:


df=pd.read_csv(r"D:\Imarticus\DATA SETS\IRIS.csv")


# In[137]:


df.head()


# In[81]:


df.shape


# In[82]:


#Let’s see some information about the dataset.
df.describe()


# In[83]:


df.isnull().sum()


# In[89]:


#visualize whole dataset
sns.pairplot(df,hue='species')


# In[133]:


#iris-setosa is well separated from the other two flowers.
# iris virginica is the longest flower and iris setosa is the shortest.


# In[93]:


Labels=df.species.value_counts()
Labels


# In[94]:


# importing packages
import seaborn as sns
import matplotlib.pyplot as plt


sns.countplot(x='species', data=df, )
plt.show()


# In[134]:


# target varaible are equal


# In[95]:


df.corr(method='pearson')


# In[98]:


# importing packages
import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(method='pearson'),annot = True);

plt.show()


# In[ ]:


Petal width and petal length have high correlations. 


# In[99]:


# importing packages
import seaborn as sns
import matplotlib.pyplot as plt

def graph(y):
	sns.boxplot(x="species", y=y, data=df)

plt.figure(figsize=(10,10))
	
# Adding the subplot at the specified
# grid position
plt.subplot(221)
graph('sepal_length')

plt.subplot(222)
graph('sepal_width')

plt.subplot(223)
graph('petal_length')

plt.subplot(224)
graph('petal_width')

plt.show()


# In[ ]:


Species Setosa has the smallest features and less distributed with some outliers.
Species Versicolor has the average features.
Species Virginica has the highest features


# In[100]:




sns.boxplot(x='sepal_width', data=df)


# In[101]:


# IQR
Q1 = np.percentile(df['sepal_width'], 25,
                interpolation = 'midpoint')
 
Q3 = np.percentile(df['sepal_width'], 75,
                interpolation = 'midpoint')
IQR = Q3 - Q1
 
print("Old Shape: ", df.shape)
 
# Upper bound
upper = np.where(df['sepal_width'] >= (Q3+1.5*IQR))
 
# Lower bound
lower = np.where(df['sepal_width'] <= (Q1-1.5*IQR))
 
# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)
 
print("New Shape: ", df.shape)
 
sns.boxplot(x='sepal_width', data=df)


# In[ ]:





# In[ ]:





# In[102]:


from sklearn.preprocessing import LabelEncoder


# In[103]:


le=LabelEncoder()


# In[107]:


df.Species=le.fit_transform(df.species)


# In[108]:


from sklearn.model_selection import train_test_split


# In[109]:


train,test=train_test_split(df,test_size=0.2)


# In[110]:


train_x=train.iloc[:,0:-1]
train_y=train.iloc[:,-1]


# In[111]:


test_x=test.iloc[:,0:-1]
test_y=test.iloc[:,-1]


# In[112]:


from sklearn.linear_model import LogisticRegression


# In[113]:


log=LogisticRegression()


# In[114]:


log.fit(train_x,train_y)


# In[115]:


pred=log.predict(test_x)


# In[116]:


from sklearn.metrics import confusion_matrix


# In[117]:


tab=confusion_matrix(test_y,pred)


# In[118]:


tab


# In[119]:


from sklearn.metrics import accuracy_score

accuracy_score(test_y,pred)*100


# In[121]:


from sklearn.svm import SVC
svc=SVC()


# In[123]:


svc.fit(train_x,train_y)


# In[124]:


pred_svc=svc.predict(test_x)


# In[126]:


svc_tab=confusion_matrix(test_y,pred_svc)
svc_tab


# In[129]:


accuracy_score(test_y,pred_svc)*100


# In[131]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines'],
    'Score': [96.66666666666667,96.66666666666667]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()


# In[ ]:




