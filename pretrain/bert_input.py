#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


train_df = pd.read_csv('./train_set.csv',sep='\t')


# In[3]:


test_df = pd.read_csv('./test_a.csv', sep ='\t')


# In[4]:


df = pd.concat((train_df,test_df))


# In[33]:


df['text'].values[0].split()[-2]


# In[52]:


anss = []
ans = ''
for text in df['text'].values:
    textline = text.strip().split()
    lenth = len(textline)
    #print(1)
    ans = ''
    for i in range(lenth): # line -> sentence
        if textline[i]=='900' or textline[i]=='2662' or textline[i]=='885':
            ans += textline[i]
            anss.append(ans)
            ans = '' 
        else:
            ans += (textline[i] + ' ')
            continue
    anss.append('sep')

print("start generate sentence")
lenth = len(anss)
with open('./sentence.txt','w') as f:
    for index in range(lenth):
        if anss[index] == 'sep' or len(anss[index])==1:
            continue
        else:
            f.write(anss[index]+'\n')




