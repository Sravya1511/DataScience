#!/usr/bin/env python
# coding: utf-8

# Lab-1

# In[1]:


import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options


# In[2]:


df=pd.read_csv("all.csv", header=None,
               names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],
)
df.head()


# In[3]:


df.dtypes


# In[4]:


df.shape


# In[5]:


df.shape[0], df.shape[1]


# In[6]:


df.columns


# In[7]:


type(df.rating), type(df)


# In[8]:


df.rating < 3


# In[9]:


np.sum(df.rating < 3)


# In[11]:


print (1*True, 1*False)


# In[12]:


np.sum(df.rating < 3)/df.shape[0]


# In[13]:


np.sum(df.rating < 3)/float(df.shape[0])


# In[14]:


np.mean(df.rating < 3.0)


# In[15]:


(df.rating < 3).mean()


# In[16]:


df.query("rating > 4.5")


# In[17]:


df[df.year < 0]


# In[18]:


df[(df.year < 0) & (df.rating > 4)]


# Clean Data

# In[19]:


df.dtypes


# In[20]:


df[df.year.isnull()]


# In[21]:


df = df[df.year.notnull()]
df.shape


# In[22]:


df['rating_count']=df.rating_count.astype(int)
df['review_count']=df.review_count.astype(int)
df['year']=df.year.astype(int)


# In[23]:


df.dtypes


# Visualizing

# In[24]:


df.rating.hist();


# In[26]:


sns.set_context("notebook")
meanrat=df.rating.mean()
#you can get means and medians in different ways
print(meanrat, np.mean(df.rating), df.rating.median())
with sns.axes_style("whitegrid"):
    df.rating.hist(bins=30, alpha=0.4);
    plt.axvline(meanrat, 0, 0.75, color='r', label='Mean')
    plt.xlabel("average rating of book")
    plt.ylabel("Counts")
    plt.title("Ratings Histogram")
    plt.legend()
    #sns.despine()


# In[27]:


df.review_count.hist(bins=np.arange(0, 40000, 400))


# In[28]:


df.review_count.hist(bins=100)
plt.xscale("log");


# In[29]:


plt.scatter(df.year, df.rating, lw=0, alpha=.08)
plt.xlim([1900,2010])
plt.xlabel("Year")
plt.ylabel("Rating")


# Pythons and ducks

# In[30]:


alist=[1,2,3,4,5]
asquaredlist=[i*i for i in alist]
asquaredlist


# In[31]:


plt.scatter(alist, asquaredlist);


# In[33]:


print(type(alist))


# In[34]:


plt.hist(df.rating_count.values, bins=100, alpha=0.5);


# In[35]:


print(type(df.rating_count), type(df.rating_count.values))


# In[36]:


alist + alist


# In[37]:


np.array(alist)


# In[38]:


np.array(alist)+np.array(alist)


# In[39]:


np.array(alist)**2


# In[40]:


newlist=[]
for item in alist:
    newlist.append(item+item)
newlist


# In[42]:


a=np.array([1,2,3,4,5])
print(type(a))
b=np.array([1,2,3,4,5])

print(a*b)


# In[43]:


a+1

