#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams


# In[5]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[6]:


boston.keys()


# In[7]:


boston.data.shape


# In[8]:


print(boston.feature_names)


# In[9]:


bos = pd.DataFrame(boston.data)
bos.head()


# In[10]:


bos.columns = boston.feature_names
bos.head()


# In[11]:


print(boston.target.shape)


# In[12]:


bos['PRICE'] = boston.target
bos.head()


# In[13]:


bos.describe()


# In[14]:


plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")


# In[15]:


plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")


# In[16]:


sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)


# In[17]:


plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")


# In[18]:


plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequencey")
plt.show()


# In[19]:


plt.hist(bos.PRICE)
plt.title('Housing Prices: $Y_i$')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[20]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[21]:


m = ols('PRICE ~ RM',bos).fit()
print(m.summary())


# In[22]:


plt.scatter(bos['PRICE'], m.fittedvalues)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")


# In[23]:


from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)

# This creates a LinearRegression object
lm = LinearRegression()
lm


# In[24]:


lm.fit(X, bos.PRICE)


# In[25]:


print('Estimated intercept coefficient:', lm.intercept_)


# In[26]:


print('Number of coefficients:', len(lm.coef_))


# In[27]:


pd.DataFrame(zip(X.columns, lm.coef_), columns = ['features', 'estimatedCoefficients'])


# In[28]:


lm.predict(X)[0:5]


# In[29]:


plt.hist(lm.predict(X))
plt.title('Predicted Housing Prices (fitted values): $\hat{Y}_i$')
plt.xlabel('Price')
plt.ylabel('Frequency')


# In[30]:


print(np.sum((bos.PRICE - lm.predict(X)) ** 2))


# In[31]:


mseFull = np.mean((bos.PRICE - lm.predict(X)) ** 2)
print(mseFull)


# In[32]:


lm = LinearRegression()
lm.fit(X[['PTRATIO']], bos.PRICE)


# In[33]:


msePTRATIO = np.mean((bos.PRICE - lm.predict(X[['PTRATIO']])) ** 2)
print(msePTRATIO)


# In[34]:


plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")

plt.plot(bos.PTRATIO, lm.predict(X[['PTRATIO']]), color='blue', linewidth=3)
plt.show()


# In[36]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X[['CRIM']], bos.PRICE)


# In[37]:


mseCRIM = np.mean((bos.PRICE - lm.predict(X[['CRIM']])) ** 2)
print(mseCRIM)


# In[38]:


plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Crime rate")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")

plt.plot(bos.CRIM, lm.predict(X[['CRIM']]), color='blue', linewidth=3)
plt.show()


# In[ ]:





# In[39]:


X_train = X[:-50]
X_test = X[-50:]
Y_train = bos.PRICE[:-50]
Y_test = bos.PRICE[-50:]
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, bos.PRICE, test_size=0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[41]:


lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)


# In[42]:


print ("Fit a model X_train, and calculate MSE with Y_train:", np.mean((Y_train - lm.predict(X_train)) ** 2))
print ("Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((Y_test - lm.predict(X_test)) ** 2))


# In[43]:


plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')


# In[44]:


faithful = sm.datasets.get_rdataset("faithful")


# In[45]:


get_ipython().run_line_magic('pinfo', 'sm.datasets.get_rdataset')
get_ipython().run_line_magic('pinfo', 'faithful')


# In[46]:


faithful.title


# In[47]:


faithful = faithful.data
faithful.head()


# In[48]:


faithful.shape


# In[49]:


plt.hist(faithful.waiting)
plt.xlabel('Waiting time to next eruption (in mins)')
plt.ylabel('Frequency')
plt.title('Old Faithful Geyser time between eruption')
plt.show()


# In[50]:


plt.scatter(faithful.waiting, faithful.eruptions)
plt.xlabel('Waiting time to next eruption (in mins)')
plt.ylabel('Eruption time (in mins)')
plt.title('Old Faithful Geyser')
plt.show()


# In[51]:


X = faithful.waiting
y = faithful.eruptions
model = sm.OLS(y, X)


# In[52]:


results = model.fit()


# In[53]:


print(results.summary())


# In[54]:


results.params.values


# In[55]:


X = sm.add_constant(X)
X.head()


# In[56]:


modelW0 = sm.OLS(y, X)
resultsW0 = modelW0.fit()
print(resultsW0.summary())


# In[57]:


newX = np.array([1,75])
resultsW0.params[0]*newX[0] + resultsW0.params[1] * newX[1]


# In[58]:


resultsW0.predict(newX)


# In[59]:


plt.scatter(faithful.waiting, faithful.eruptions)
plt.xlabel('Waiting time to next eruption (in mins)')
plt.ylabel('Eruption time (in mins)')
plt.title('Old Faithful Geyser')

plt.plot(faithful.waiting, resultsW0.fittedvalues, color='blue', linewidth=3)
plt.show()


# In[60]:


resids = faithful.eruptions - resultsW0.predict(X)


# In[61]:


resids = resultsW0.resid


# In[62]:


plt.plot(faithful.waiting, resids, 'o')
plt.hlines(y = 0, xmin=40, xmax = 100)
plt.xlabel('Waiting time')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[63]:


print(np.sum((faithful.eruptions - resultsW0.predict(X)) ** 2))


# In[64]:


print(np.mean((faithful.eruptions - resultsW0.predict(X)) ** 2))


# In[65]:


X = sm.add_constant(faithful.waiting)
y = faithful.eruptions


# In[66]:


np.dot(X.T, X)


# In[67]:


np.linalg.inv(np.dot(X.T, X))


# In[68]:


beta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
print("Directly estimating beta:", beta)
print("Estimating beta using statmodels: ", resultsW0.params.values)


# In[70]:


from IPython.display import Image as Im
from IPython.display import display
Im('./images/shuttle.png')


# In[72]:


data=np.array([[float(j) for j in e.strip().split()] for e in open("./data/chall.txt")])
data


# In[73]:


# fit logistic regression model
import statsmodels.api as sm
from statsmodels.formula.api import logit, glm, ols

# statsmodels works nicely with pandas dataframes
dat = pd.DataFrame(data, columns = ['Temperature', 'Failure'])
logit_model = logit('Failure ~ Temperature',dat).fit()
print(logit_model.summary())


# In[74]:


# calculate predicted failure probabilities for new termperatures
x = np.linspace(50, 85, 1000)
p = logit_model.params
eta = p['Intercept'] + x*p['Temperature']
y = np.exp(eta)/(1 + np.exp(eta))


# In[75]:


# plot data
temps, pfail = data[:,0], data[:,1]
plt.scatter(temps, pfail)
axes=plt.gca()
plt.xlabel('Temperature')
plt.ylabel('Failure')
plt.title('O-ring failures')

# plot fitted values
plt.plot(x, y)

# change limits, for a nicer plot
plt.xlim(50, 85)
plt.ylim(-0.1, 1.1)

