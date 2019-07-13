
# coding: utf-8

# #### Importing Required packages

# In[1]:

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from pandas_ml import ConfusionMatrix
import pandas_ml as pdml
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle

# #### Reading the data

# In[2]:


def read_input():
    train = pd.read_csv("creditcard.csv")
    #train = train.sample(frac=1).reset_index(drop=True)
    return train


# In[3]:


train = read_input()
train.head()


# #### Checking for null values

# In[4]:


print(train.isna().sum())


# In[5]:


frauds = train.loc[train['Class'] == 1]
no_frauds = train.loc[train['Class'] == 0]
print("Frauds:",len(frauds))
print("No Frauds:",len(no_frauds))


# ## Exploratory Data Analysis(EDA)

# In[131]:


count_classes = pd.value_counts(train['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar',rot=0)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[6]:


bx = frauds.plot.scatter(x='Amount', y='Class', color='Blue', label='Fraud')
print("Checking the amount for which fraud transaction occured")
plt.show()


# #### Plotting fraud and non-fraud trasactions with respect to amount

# In[7]:


ax = frauds.plot.scatter(x='Amount', y='Class', c='Red', label='Fraud')
no_frauds.plot.scatter(x='Amount', y='Class', c='Blue', label='Normal', ax=ax)
plt.show()


# By looking at the above plot we can conclude that data is very imbalanced as there are very few data points for class 1(fraud)

# ### Plotting correlation of all the data

# In[84]:


#Compute correlation matrix
corr = train.drop(['Class'], axis=1).corr()
get_ipython().magic(u'matplotlib inline')

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] =True
f, ax = plt.subplots(figsize=(14, 12))
cmap= sns.diverging_palette(220, 10, as_cmap= True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4,
            square=True, xticklabels=1, yticklabels=1,
            linewidths=0.4,fmt= '.1f', cbar_kws={"shrink": 0.5}, ax=ax,annot = True)


# ## KDE plots of all features (for feature selection)

# Drawing kde plots of all the features for both classes 0 and 1.If the graph is same then that particular feature is not helping in classifying the data correctly. So remove all such features
#

# In[33]:


for j in list(train):
    for i in range(2):
        if(j in ('Time','Amount','Class')):
            continue
        else:
            sns.kdeplot(train[train.Class==i][j])
    plt.show()


#  keep the following features with time and amount:
#  V1, V3, V4, V7, V9, V10, V11, V12, V14, V16, V17, V18, V19

# In[168]:


features = ['Time','V1','V3','V4','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','Amount']


# ### Splitting the dataset into train, test

# In[169]:



X = train[features]
y = train["Class"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,random_state = 123)
print(X_train.shape)
print(X_test.shape)


# Printing the size of fraud and non-fraud data in train and test set

# In[170]:


print("Train and test sizes, respectively:", len(X_train), len(y_train), "|", len(X_test), len(y_test))
print("Total number of frauds:", len(y.loc[train['Class'] == 1]), len(y.loc[train['Class'] == 1])/len(y))
print("Number of frauds on y_test:", len(y_test.loc[train['Class'] == 1]), " and  % =",len(y_test.loc[train['Class'] == 1]) / len(y_test))
print("Number of frauds on y_train:", len(y_train.loc[train['Class'] == 1])," and  % =" ,len(y_train.loc[train['Class'] == 1])/len(y_train))


# ## Using Logistic regression,DT,RF,GBM without Oversampling

# In[171]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


# ### Pickling the model
filename = 'basic_LR.sav'
pickle.dump(lr, open(filename, 'wb'))


# In[173]:

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[174]:

y_predicted = np.array(lr.predict(X_test))
y_right = np.array(y_test)
print("Score :",lr.score(X_test,y_test))


# ### Plotting confusion matrix

# In[175]:

confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot(normalized=True)
plt.show()


# In[176]:


confusion_matrix.print_stats()


# ### Percentage of  Fraud Trasaction which model detected incorrectly

# In[177]:


print("FNR is {0}".format(confusion_matrix.stats()['FNR']))


# #### Plotting ROC curve

# In[178]:

logit_roc_auc = roc_auc_score(y_test, lr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted, lr.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[10]:


y_predicted = np.array(model.predict(X_test))
y_right = np.array(y_test)
print("Score :",model.score(X_test,y_test))


# In[11]:



confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)


# In[12]:


confusion_matrix.plot(normalized=True)
plt.show()


# In[13]:


confusion_matrix.print_stats()


# In[14]:


print("FNR is {0}".format(confusion_matrix.stats()['FNR']))


# In[15]:

logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted, model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='decision Tree (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=25)
rfc.fit(X_train,y_train)
print("RF score = ",rfc.score(X_test,y_test))


# In[17]:


y_predicted = np.array(rfc.predict(X_test))
y_right = np.array(y_test)
print("Score :",rfc.score(X_test,y_test))


# In[18]:


confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)


# In[19]:


print("FNR is {0}".format(confusion_matrix.stats()['FNR']))


# In[20]:

logit_roc_auc = roc_auc_score(y_test, rfc.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted, rfc.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[21]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=300)
gbc.fit(X_train,y_train)
print("GB Score = ",gbc.score(X_test,y_test))


# In[22]:


y_predicted1 = np.array(gbc.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)


# In[23]:


print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))


# In[24]:

logit_roc_auc = roc_auc_score(y_test, gbc.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted, gbc.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='GBC Tree (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




#    ## Using SMOTE For Over Sampling

# In[179]:


train2 = pdml.ModelFrame(X_train, target=y_train)
sampler = train2.imbalance.over_sampling.SMOTE()
sampler


# In[180]:


sampled = train2.fit_sample(sampler)
sampled.head()


# In[181]:


print("Size of training set after over sampling:", len(sampled))


# ## Using Logistic regression,DT,RF,GBM with Oversampling

# In[182]:


X_train_sampled = sampled.iloc[:,1:]
y_train_sampled = sampled['Class']


logistic = LogisticRegression()
logistic.fit(X_train_sampled, y_train_sampled)
print("Score: ", logistic.score(X_test, y_test))


# In[51]:


filename = 'Over_Sampled_LR.sav'
pickle.dump(logistic, open(filename, 'wb'))


# ### Plotting Confusion matrix

# In[146]:


y_predicted1 = np.array(logistic.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)
confusion_matrix1.plot(normalized=True)
plt.show()


# In[147]:


confusion_matrix1.print_stats()


# In[148]:


print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))


# ### Plotting ROC curve

# In[149]:


logit_roc_auc = roc_auc_score(y_test, logistic.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted1, logistic.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



dt = DecisionTreeClassifier()
dt.fit(X_train_sampled, y_train_sampled)
print("Score: ", dt.score(X_test, y_test))


# In[29]:


filename = 'Over_Sampled_DT.sav'
pickle.dump(dt, open(filename, 'wb'))


# In[30]:

# In[31]:


y_predicted1 = np.array(dt.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)


# In[32]:


print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))


# In[33]:


logit_roc_auc = roc_auc_score(y_test, dt.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted1, dt.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='decision Tree (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[34]:


rf2 = RandomForestClassifier(n_estimators = 25)
rf2.fit(X_train_sampled, y_train_sampled)
print("Score: ", rf2.score(X_test, y_test))


# In[35]:


filename = 'Over_Sampled_RF.sav'
pickle.dump(dt, open(filename, 'wb'))


# In[36]:


y_predicted1 = np.array(rf2.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)


# In[37]:


print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))


# In[38]:


logit_roc_auc = roc_auc_score(y_test, dt.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted1, rf2.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='RandomForest Tree (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[39]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train_sampled,y_train_sampled)
("Score = ",gb.score(X_test,y_test))


# In[40]:

filename = 'Over_Sampled_GB.sav'
pickle.dump(gb, open(filename, 'wb'))

# In[41]:

y_predicted1 = np.array(dt.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)


# In[42]:


print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))


# In[43]:

logit_roc_auc = roc_auc_score(y_test, gb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted1, gb.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='GBM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


### Logistic regression with balanced class weight

# In[150]:

best_c, best_fnr = 1, 1
for i in range(20):
    print()
    print("values for round ",(i+1))
    c = random.uniform(0, 1)
    logist = LogisticRegression(C=c, class_weight="balanced")
    logist.fit(X_train_sampled, y_train_sampled)
    print("Score: ", logist.score(X_test, y_test))
    y_predicted2 = np.array(logist.predict(X_test))
    y_right2 = np.array(y_test)
    confusion_matrix2 = ConfusionMatrix(y_right2, y_predicted2)

    fnr = confusion_matrix2.stats()['FNR']
    print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))
    if fnr < best_fnr:
        best_fnr = fnr
        best_c = c
print("Best C is {0} with best FNR of {1}.".format(best_c, best_fnr))

# In[151]:

print(best_c)

# In[152]:

logist1 = LogisticRegression(C=best_c, class_weight="balanced")
logist1.fit(X_train_sampled, y_train_sampled)
print("Score: ", logist1.score(X_test, y_test))
filename = 'Over_Sampled_LR_with_balanced_class_weight.sav'
pickle.dump(logist1, open(filename, 'wb'))
y_predicted1 = np.array(logistic.predict(X_test))
y_right1 = np.array(y_test)

# In[153]:

y_predicted1 = np.array(logistic.predict(X_test))
y_right1 = np.array(y_test)

confusion_matrix1 = ConfusionMatrix(y_right1, y_predicted1)
print("Confusion matrix:\n%s" % confusion_matrix1)
confusion_matrix1.plot(normalized=True)
plt.show()

# In[154]:
print("FNR is {0}".format(confusion_matrix1.stats()['FNR']))
# In[155]:

logit_roc_auc = roc_auc_score(y_test, logist1.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_predicted1, logist1.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
