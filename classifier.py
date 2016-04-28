
# coding: utf-8

# In[97]:

get_ipython().magic('matplotlib inline')


# In[134]:

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import metrics


# In[274]:

#data = pd.read_csv('train1p_with_header.csv')
data = pd.read_csv('train.csv')


# In[ ]:




# In[277]:

truck_inxs = np.where(data['class'] == 'truck')
cat_inxs = np.where(data['class'] =='cat')
truck_cat_inxs = np.append(truck_inxs,cat_inxs)
print cat_inxs


# In[278]:

data_truck_cat = data.iloc[truck_cat_inxs,:]


# In[279]:

data_truck_cat.shape


# In[100]:

pca = PCA(n_components=300)


# In[202]:

X=data.iloc[:,0:data.shape[1]-1].as_matrix()
y=data.iloc[:,data.shape[1]-1]


# In[102]:

pca.fit(X)


# In[103]:

data.head()


# In[104]:

print(pca.explained_variance_ratio_) 


# In[105]:

scores = np.dot(X,np.transpose(pca.components_))


# In[106]:

scores.shape


# In[107]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,oob_score=True)
clf.fit(scores,y)


# In[108]:

clf.oob_score_


# In[109]:

clf.predict(scores)


# # Just truck versus cat

# In[110]:

#truck_inxs = np.where(y == 'truck')


# In[111]:

#cat_inxs = np.where(y == 'cat')


# In[112]:

#truck_cat_inxs = np.append(cat_inxs,truck_inxs)


# In[207]:

#truck_cat_data = data.iloc[truck_cat_inxs,:]


# In[280]:

truck_cat_data = data_truck_cat


# In[281]:

truck_cat_data.shape


# In[282]:

truck_cat_pca = PCA(n_components=300)


# In[283]:

X=truck_cat_data.iloc[:,0:truck_cat_data.shape[1]-1].as_matrix()
y=truck_cat_data.iloc[:,truck_cat_data.shape[1]-1]


# In[284]:

truck_cat_pca.fit(X)


# In[285]:

X_pca = np.dot(X,np.transpose(pca.components_))


# In[333]:

inxs = np.random.permutation(X_pca.shape[0])
X_pca_train = X_pca[inxs[:7000],:]
X_pca_test = X_pca[inxs[7001:],:]
y_train = y.iloc[inxs[:7000]]
y_test = y.iloc[inxs[7001:]]


# In[335]:

y_test


# In[336]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,oob_score=True)
clf.fit(X_pca_train,y_train)


# In[337]:

clf.oob_score_


# In[121]:




# In[338]:

probs = clf.predict_proba(X_pca_test)

#fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


# In[339]:

probs


# In[340]:

clf.classes_


# In[300]:

import copy
y01 = np.zeros(np.size(y))
y01[np.where(y == 'cat')] = 1
print y
print y01


# In[341]:

print y_test


# In[342]:

fpr, tpr, thresholds = metrics.roc_curve(y_test,probs[:,0],pos_label='cat')


# In[343]:

fpr


# In[294]:

fpr = np.array(fpr)
tpr = np.array(tpr)
print fpr


# In[295]:

print tpr*1.,fpr*1.


# In[344]:

print metrics.classification_report(y_test,clf.predict(X_pca_test))


# In[ ]:




# In[346]:

#plt.plot(tpr, fpr)
#plt.plot([0, 1], [0, 1], 'k--')
#dt = np.c_[fpr,tpr]
#print dt
plt.plot(fpr, tpr,'r-',label='test')
plt.title('TRUCKS VS CATS')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# In[167]:

x = np.linspace(0, 3*np.pi, 500)
plt.plot(x, np.sin(x**2))
plt.title('A simple chirp')
plt.show()


# In[158]:

x


# In[ ]:



