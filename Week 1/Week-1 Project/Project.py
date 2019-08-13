#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
from bs4 import BeautifulSoup
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")


# In[2]:


import requests
req = requests.get("https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_1970")
page = req.text
soup = BeautifulSoup(page, 'html.parser')


# In[3]:


soup.table["class"]


# In[4]:


table_html = str(soup.find("table", "wikitable"))


# In[5]:


from IPython.core.display import HTML

HTML(table_html)


# In[6]:


rows_list = []
rows_list = soup.find("table", "wikitable").find_all('tr')
final_list = []

for row in rows_list[1:]:
    dict1 = {}
    values = row.find_all('td')
    dict1["band_singer"] = values[2].get_text().replace("\n","")
    dict1["ranking"] = values[0].get_text()
    dict1["title"] = values[1].get_text()
    str1 = values[1].find('a')
    dict1["url"] = str1.get('href')
    final_list.append(dict1)
print(final_list)


# In[7]:


import requests
yearstext = {}
for year in range(1992,2015):
    req = requests.get(" http://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_"+str(year))
    yearstext[year] = req


# In[8]:


def parse_year(the_year, yeartext_dict):
    year_text = yeartext_dict[the_year].text
    soup = BeautifulSoup(year_text, 'html.parser')
    rows = [row for row in soup.find("table", "wikitable").find_all("tr")]
    songsList = []
    for row in rows[1:]:
        values = row.find_all("td")
        ranking = row.find("th").get_text().replace("\n", "")
        songurllinks = values[0].find_all('a')
        song = []
        songurl = []
        band_singer = []
        url = []
        titletext = ''
        if(len(songurllinks) == 0):
            songurl = [None]
            song = [values[0].get_text()]
            titletext = values[0].get_text()
        else:
            for urllink in songurllinks:
                songurl.append(urllink.get("href"))
                song.append(urllink.get_text())
                titletext = titletext + '" ' + urllink.get_text() + ' "' +' / '
            titletext = titletext.strip(" / ")
        artistlinks = values[1].find_all("a")
        if(len(artistlinks) == 0):
            url = [None]
            band_singer = [values[1].get_text()]
        else:
            for artist in artistlinks:
                band_singer.append(artist.get_text())
                url.append(artist.get("href"))
        songdetails = {
            'band_singer' : band_singer,
            'ranking' : ranking,
            'song' : song,
            'songurl' : songurl,
            'titletext' : titletext,
            'url' : url
        }
        songsList.append(songdetails)
    return songsList


# In[9]:


parse_year(1997, yearstext)[:5]


# In[10]:


yearinfo = {}
for year in range(1992,2015):
    yearinfo[year] = parse_year(year, yearstext)
print(yearinfo[1997])


# In[11]:


import json


# In[12]:


fd = open("yearinfo.json","w")
json.dump(yearinfo, fd)
fd.close()
del yearinfo


# In[13]:


with open("yearinfo.json", "r") as fd:
    yearinfo = json.load(fd)
print(yearinfo['1997'])


# In[14]:


songs = []
for year in yearinfo.keys():
    for info in yearinfo[year]:
        info['year'] = year
        songs.append(info)
print(songs)
print(len(songs))


# In[15]:


about_to_remove = []
for each_song in songs:
    if(len(each_song['band_singer']) > 1):
        for each_singer in each_song['band_singer']:
            index = each_song['band_singer'].index(each_singer)
            each_singer_dict = {
                'band_singer' : [each_singer],
                'ranking' : each_song['ranking'],
                'song' : each_song['song'],
                'songurl' : each_song['songurl'],
                'titletext' : each_song['titletext'],
                'url' : each_song['url'][index],
                'year' : each_song['year']
            }
            songs.append(each_singer_dict)
        about_to_remove.append(songs.index(each_song))


# In[16]:


for removeIndex in about_to_remove:
    del songs[removeIndex]


# In[17]:


import pandas as pd


# In[18]:


flatframe = pd.DataFrame(songs)
flatframe


# In[19]:


for every_song in songs:
    for every_feature in every_song.keys():
        every_song[every_feature] = str(every_song[every_feature])
        every_song[every_feature] = every_song[every_feature].strip("[]")
        every_song[every_feature] = every_song[every_feature].strip("''")


# In[20]:


artist_count = flatframe["band_singer"].value_counts()
artist_count


# In[21]:


urlcache={}
def get_page(url):
    if (url not in urlcache) or (urlcache[url]==1) or (urlcache[url]==2):
        time.sleep(1)
        try:
            r = requests.get("http://en.wikipedia.org%s" % url)

            if r.status_code == 200:
                urlcache[url] = r.text
            else:
                urlcache[url] = 1
        except:
            urlcache[url] = 2
    return urlcache[url]


# In[22]:


flatframe=flatframe.sort_values('year')
flatframe.head()


# In[ ]:


flatframe["url"].apply(get_page)


# In[24]:


with open("artistinfo.json","w") as fd:
    json.dump(urlcache, fd)
del urlcache


# In[25]:


with open("artistinfo.json") as json_file:
    urlcache = json.load(json_file)


# In[26]:


def singer_band_info(url, pagetext):
    bday_dict = {}
    bday = ''
    ya = ''
    # soupify our webpage
    soup = BeautifulSoup(page_text[url], "lxml")
    tables = soup.find('table', attrs={'class':'infobox vcard plainlist'})
    if (tables == None):
        tables = soup.find('table', attrs={'class':'infobox biography vcard'})
    bday = tables.find('span', {'class': 'bday'})
    if bday: 
        bday = bday.get_text()[:4]
        bday_dict = {'url' : url, 'born' : bday, 'ya' : ya}
    else:
        ya = False
        for tr in tables.find_all('tr'):
            if hasattr(tr.th, 'span'):
                if hasattr(tr.th.span, 'string'):
                    if tr.th.span.string == 'Years active':                
                        if hasattr(tr.th, 'span'):
                            #ya = tr.td.string
                            ya = tr.td.text   #DK add
                            bday = 'False'
                            bday_dict = {'url' : url, 'born' : 'False', 'ya' : ya}
    return(bday_dict)


# In[ ]:


url = '/wiki/Iggy_Azalea'
singer_band_info(url, urlcache)
list_of_addit_dicts = []
bday_dict = {}
for url in urlcache.keys():   
    try:
        bday_dict = singer_band_info(url, urlcache)
        list_of_addit_dicts.append(bday_dict)
    except:
        break
additional_df = pd.DataFrame(list_of_addit_dicts)

largedf = pd.merge(flatframe, additional_df, left_on='url', right_on='url', how="outer")
largedf


# In[28]:


import json


# In[ ]:


# DO NOT RERUN THIS CELL WHEN SUBMITTING
fd = open("data/yearinfo.json","w")
json.dump(yearinfo, fd)
fd.close()
del yearinfo


# In[ ]:


with open("data/yearinfo.json", "r") as fd:
    yearinfo = json.load(fd)


# In[31]:


urlcache={}


# In[32]:


def get_page(url):
    # Check if URL has already been visited.
    if (url not in urlcache) or (urlcache[url]==1) or (urlcache[url]==2):
        time.sleep(1)
        # try/except blocks are used whenever the code could generate an exception (e.g. division by zero).
        # In this case we don't know if the page really exists, or even if it does, if we'll be able to reach it.
        try:
            r = requests.get("http://en.wikipedia.org%s" % url)

            if r.status_code == 200:
                urlcache[url] = r.text
            else:
                urlcache[url] = 1
        except:
            urlcache[url] = 2
    return urlcache[url]


# In[35]:


flatframe=flatframe.sort_values('year')
flatframe.head()


# In[ ]:


flatframe["url"].apply(get_page)


# In[ ]:


print("Number of bad requests:",np.sum([(urlcache[k]==1) or (urlcache[k]==2) for k in urlcache])) # no one or 0's)
print("Did we get all urls?", len(flatframe.url.unique())==len(urlcache)) # we got all of th


# In[ ]:


with open("data/artistinfo.json","w") as fd:
    json.dump(urlcache, fd)
del urlcache


# In[ ]:


with open("data/artistinfo.json") as json_file:
    urlcache = json.load(json_file)


# In[ ]:




