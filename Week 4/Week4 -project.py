#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LinearRegression


import seaborn.apionly as sns
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

def leppard(source_data, prediction_data):
    false_negative = 0
    false_positive = 0
    correct_assessment = 0
    for result in range(0, len(prediction_data)):
        if int(prediction_data[result]) == 1 and int(source_data[result]) == 0:
            false_positive += 1
        if int(prediction_data[result]) == 0 and int(source_data[result]) == 1:
            false_negative += 1
        if (int(prediction_data[result]) == 1 and int(source_data[result]) == 1) or (int(prediction_data[result]) == 0 and int(source_data[result]) == 0):
            correct_assessment += 1
    print ()
    print ("False Positives: ", false_positive)
    print ("False Negatives: ", false_negative)
    print ("Correct Assessment: ", correct_assessment)

    print ("Classification Accuracy: ", 1 - (false_positive + false_negative) / len(source_data))


# In[2]:


np.random.seed(9001)
df = pd.read_csv('hw6_dataset.csv')
msk = np.random.rand(len(df)) < 0.75
data_train = df[msk]
data_test = df[~msk]
orig_columns = list(data_train.columns.values)
new_columns = []
for x in range (len(orig_columns) - 1):
    #print(orig_columns[x])
    index_of_e = orig_columns[x].index('e')
    revised_string = orig_columns[x][:index_of_e + 4]
    #print(revised_string)
    converted_string = float(revised_string)
    new_columns.append(str(converted_string))
new_columns.append('Class Label')
#print(new_columns)
data_train.columns = new_columns
data_test.columns = new_columns
data_train.head(10)

y_train = data_train['Class Label'].values
X_train = data_train.values
y_train = y_train.reshape(len(y_train), 1)

y_test = data_test['Class Label'].values
X_test = data_test.values
y_test = y_test.reshape(len(y_test), 1)


# In[3]:


df.head()


# In[4]:


clf = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,cv=10
        ,random_state=777
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10)
clf.fit(X_train, y_train)
print('\n')
print("The optimized L2 regularization paramater id:", clf.C_)

# The coefficients
print('Estimated beta1: \n', clf.coef_)
print('Estimated beta0: \n', clf.intercept_)

# Scoring
clf_y_pred_test = clf.predict(X_test)
clf_y_pred_test = clf_y_pred_test.reshape(len(clf_y_pred_test), 1)
test_df = pd.DataFrame(clf_y_pred_test)
Total = test_df[0].sum()
print('\n')
print("malignant: ", Total)

pd.set_option('display.max_rows', 1000)
test_df['All Normal'] = 0

# Reset indexes so copy will work
test_df = test_df.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)
test_df['Class Label'] = data_test['Class Label']

# Confusion Matrix
print('\n')
print('Classifier applied to Test Set:') 
leppard(test_df['Class Label'], test_df[0])
print(confusion_matrix(y_test, clf.predict(X_test)))


print('\n')
print('Classifier that predicts all normal:')
leppard(test_df['Class Label'], test_df['All Normal'])
print(confusion_matrix(y_test, test_df['All Normal']))


# In[5]:


def t_repredict(est, t, xtest):
    probs = est.predict_proba(xtest)
    p0 = probs[:,0]
    p1 = probs[:,1]
    ypred = (p1 > t)*1
    return ypred
print('Confusion matrix that predicts all patients to be negative:')
print(confusion_matrix(y_test,t_repredict(clf, 01.00, X_test)))


# In[6]:


from sklearn.metrics import roc_curve, auc

def make_roc(name, clf, ytest, xtest, ax=None, labe=5, proba=True, skip=0):
    initial=False
    if not ax:
        ax=plt.gca()
        initial=True
    if proba:#for stuff like logistic regression
        fpr, tpr, thresholds=roc_curve(ytest, clf.predict_proba(xtest)[:,1])
    else:#for stuff like SVM
        fpr, tpr, thresholds=roc_curve(ytest, clf.decision_function(xtest))
    roc_auc = auc(fpr, tpr)
    if skip:
        l=fpr.shape[0]
        ax.plot(fpr[0:l:skip], tpr[0:l:skip], '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    else:
        ax.plot(fpr, tpr, '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    label_kwargs = {}
    label_kwargs['bbox'] = dict(
        boxstyle='round,pad=0.3', alpha=0.2,
    )
    if labe!=None:    
        for k in range(0, fpr.shape[0],labe):
            #from https://gist.github.com/podshumok/c1d1c9394335d86255b8
            threshold = str(np.round(thresholds[k], 2))
            ax.annotate(threshold, (fpr[k], tpr[k]), **label_kwargs)
    if initial:
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
    fpr_0, tpr_0, thresholds_1 = metrics.roc_curve(y_test, t_repredict(clf, 01.00, X_test))
    roc_auc_0 = auc(fpr_0, tpr_0)
    plt.plot(fpr_0, tpr_0, '.-', alpha=0.3, label='ROC curve for all Negative Predictions (area = %0.2f)' % (roc_auc_0))
    ax.legend(loc="lower right")
    return ax

ax=make_roc("logistic",clf, y_test, X_test, labe=100, skip=2)


# In[7]:


len(clf.predict_proba(X_test)[:,1])


# In[8]:


fprs = [0,.1,.5,.9]
fpr, tpr, thresholds=roc_curve(y_test, clf.predict_proba(X_test)[:,1])       
for i in range(len(fpr)):        
    if int(fpr[i] > 0):
                print('FPR:', fpr[i], 'TPR', tpr[i], 'Threshold', thresholds[i] )
                break
for i in range(len(fpr)):             
    if int(fpr[i]) >= .1:
                print('FPR:', fpr[i], 'TPR', tpr[i], 'Threshold', thresholds[i] )
                
for i in range(len(fpr)):                 
    if int(fpr[i]) >= .5:
                print('FPR:', fpr[i], 'TPR', tpr[i], 'Threshold', thresholds[i] ) 
                
for i in range(len(fpr)):      
    if int(fpr[i]) >= .9:
                print('FPR:', fpr[i], 'TPR', tpr[i], 'Threshold', thresholds[i] )  
                
                f, ax = plt.subplots()
                ax.plot(fpr, tpr)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC')
                ax.legend(loc="lower right")
           


# In[9]:


print('fprs:',fpr)


# In[10]:


print('tprs:',tpr)


# In[11]:


print('thresholds:',thresholds)


# In[12]:


np.random.seed(9001)
df = pd.read_csv('HW6_dataset_missing.csv')
msk = np.random.rand(len(df)) < 0.75
data_train = df[msk]
data_test = df[~msk]
data_train = data_train.dropna()
data_test = data_test.dropna()


# In[13]:


df.head()


# In[14]:


y_train = data_train['type'].values
X_train = data_train.values
y_train = y_train.reshape(len(y_train), 1)

y_test = data_test['type'].values
X_test = data_test.values
y_test = y_test.reshape(len(y_test), 1)


# In[15]:


clf = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,cv=10
        ,random_state=777
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10)
clf.fit(X_train, y_train)

# L2 Regularization parameter
print('\n')
print("The optimized L2 regularization paramater id:", clf.C_)

# The coefficients
print('Estimated beta1: \n', clf.coef_)
print('Estimated beta0: \n', clf.intercept_)

# Metrics
print('\n')
print('Test Set Confusion matrix:') 
print(confusion_matrix(y_test, clf.predict(X_test)))

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
y_prediction = clf.predict(X_test)
test_precision = precision_score(y_test, y_prediction)
print('The training classification accuracy is: ', train_score)
print('The testing classification accuracy is: ', test_score)
print('The precision score on the test set is: ', test_precision)


# In[16]:


np.random.seed(9001)
df_2 = pd.read_csv('HW6_dataset_missing.csv')
msk = np.random.rand(len(df)) < 0.75
data_train_2 = df_2[msk]
data_test_2 = df_2[~msk]


# In[17]:


for column in data_train_2:
    data_train_2[column] = data_train_2[column].fillna(data_train_2[column].mean())
for column in data_test_2:
    data_test_2[column] = data_test_2[column].fillna(data_train_2[column].mean())
    
y_train = data_train_2['type'].values
X_train = data_train_2.values
y_train = y_train.reshape(len(y_train), 1)

y_test = data_test_2['type'].values
X_test = data_test_2.values
y_test = y_test.reshape(len(y_test), 1)


# Fit a logistic regression classifier to the training set and report the accuracy of the classifier on the test set
clf = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,cv=10
        ,random_state=777
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10)
clf.fit(X_train, y_train)

# L2 Regularization parameter
print('\n')
print("The optimized L2 regularization paramater id:", clf.C_)

# The coefficients
print('Estimated beta1: \n', clf.coef_)
print('Estimated beta0: \n', clf.intercept_)

# Metrics
print('\n')
print('Test Set Confusion matrix:') 
print(confusion_matrix(y_test, clf.predict(X_test)))

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
y_prediction = clf.predict(X_test)
test_precision = precision_score(y_test, y_prediction)
print('The training classification accuracy is: ', train_score)
print('The testing classification accuracy is: ', test_score)
print('The precision score on the test set is: ', test_precision)


# In[18]:


#Split the data set into a training set and a testing set
np.random.seed(9001)
df_imp = pd.read_csv('HW6_dataset_missing.csv')
msk = np.random.rand(len(df)) < 0.75
data_train_imp = df_imp[msk]
#print(data_train_imp)
data_test_imp = df_imp[~msk]
data_train_full = data_train_imp.dropna()

data_test_imp.iloc[:, 91]
#y_train_imp = data_train['type'].values
#X_train_imp = data_train.values
#y_train_imp = y_train.reshape(len(y_train), 1)

#y_test_imp = data_test_imp['type'].values
#X_test_imp = data_test_imp.values
#y_test_imp = y_test_imp.reshape(len(y_test), 1)


# In[19]:


for i in range(1,117,1):
    y_train_imp = data_train_full.iloc[:, i]
    #print(y_train_imp)
    #print(y_train_imp.shape())
    X_train_imp = data_train_full.loc[:, data_train_full.columns != i]
    #print(X_train_imp)
    #print(X_train_imp.shape())
    y_train_imp = y_train_imp.reshape(len(y_train_imp), 1)
    #print(y_train_imp)
    
    # regress column i on all other columns with randomness
    regress = LinearRegression()
    regress.fit(X_train_imp,y_train_imp)
    y_hat = regress.predict(X_train_imp)
    
    X_missing = data_test_imp[data_test_imp.iloc[:, i].isnull()]
   
    print (X_missing)
    if not X_missing:
        print("X ", i, "complete; nothing missing")
        continue
    else:
        print(X_missing)
        print(TEST_missing)
        

    y_missing = regress.predict(X_missing)
    y_missing_noise = y_missing+np.random.normal(loc=0,scale=np.sqrt(mean_squared_error(y_train_imp,y_hat)),size=y_missing.shape[0])

        
     
    missing_index = data_train_imp.i[data_train_imp.i.isnull()].index
    missing_series = pd.Series(data = y_missing_noise, index = missing_index)
    
    #back to the data set with missingness and impute the predictions
    data_train_imp2 = data_train_imp.copy()
    data_train_imp2[i] = data_train_imp2[i].fillna(missing_series)
    
    # regress on test set
    regress.fit(X_train_imp,y_train_imp)
    y_hat = regress.predict(X_train_imp)
    
    X_missing = data_train_imp[data_train_imp.i.isnull()]
    X_missing = X_missing.reshape(len(X_missing), 1)
    y_missing = regress.predict(X_missing)
    y_missing_noise = y_missing+np.random.normal(loc=0,scale=np.sqrt(mean_squared_error(y_train_imp,y_hat)),size=y_missing.shape[0])
    
    missing_index = data_train_imp.i[data_train_imp.i.isnull()].index
    missing_series = pd.Series(data = y_missing_noise, index = missing_index)
    
    #back to the data set with missingness and impute the predictions
    data_train_imp2 = data_train_imp.copy()
    data_train_imp2[i] = data_train_imp2[i].fillna(missing_series)
    
    # Fit a logistic regression classifier to the training set and report the accuracy of the classifier on the test set
    clf = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,cv=10
        ,random_state=777
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10)
    clf.fit(X_train, y_train)

    # L2 Regularization parameter
    print('\n')
    print("The optimized L2 regularization paramater id:", clf.C_)

    # The coefficients
    print('Estimated beta1: \n', clf.coef_)
    print('Estimated beta0: \n', clf.intercept_)

    # Metrics
    print('\n')
    print('Test Set Confusion matrix:') 
    print(confusion_matrix(y_test, clf.predict(X_test)))
    
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    y_prediction = clf.predict(X_test)
    test_precision = precision_score(y_test, y_prediction)
    print('The training classification accuracy is: ', train_score)
    print('The testing classification accuracy is: ', test_score)
    print('The precision score on the test set is: ', test_precision)

