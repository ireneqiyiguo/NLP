#!/usr/bin/env python
# coding: utf-8

# # 分词的概念
正向最大匹配法：对句子从左到右进行扫描，尽可能地选择与词典中最长单词匹配的词作为目标分词，然后进行下一次匹配。
逆向最大匹配法：对句子从右到左进行扫描，尽可能地选择与词典中最长单词匹配的词作为目标分词，然后进行下一次匹配。
双向最大匹配法：将正向最大匹配算法和逆向最大匹配算法进行比较，从而确定正确的分词方法。
# # unigram、bigram、trigram的概念
unigram 一元分词，把句子分成一个一个的汉字
bigram 二元分词，把句子从头到尾每两个字组成一个词语
trigram 三元分词，把句子从头到尾每三个字组成一个词语
# In[12]:


# 字符频率统计
import jieba
from collections import Counter
 
data = '北京大学和清华大学是中国的顶尖大学'
 
print('单词统计')
words = list(jieba.cut(data))
print(Counter(words))
 
print('字符统计')
print(Counter(list(data)))


# # 文本矩阵化
词袋模型，词级别的矩阵化，参考伽音的code
# In[14]:


import jieba
import pandas as pd
import tensorflow as tf
from collections import Counter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


# 读取停用词
def read_stopword(filename):
    stopword = []
    fp = open(filename, 'r')
    for line in fp.readlines():
        stopword.append(line.replace('\n', ''))
    fp.close()
    return stopword


# In[16]:


# 切分数据，并删除停用词
def cut_data(data, stopword):
    words = []
    for content in data['content']:
        word = list(jieba.cut(content))
        for w in list(set(word) & set(stopword)):
            while w in word:
                word.remove(w)
        words.append(' '.join(word))
    data['content'] = words
    return data


# In[17]:



# 获取单词列表
def word_list(data):
    all_word = []
    for word in data['content']:
        all_word.extend(word)
    all_word = list(set(all_word))
    return all_word


# In[18]:


# 计算文本向量
def text_vec(data):
    count_vec = CountVectorizer(max_features=300, min_df=2)
    count_vec.fit_transform(data['content'])
    fea_vec = count_vec.transform(data['content']).toarray()
    return fea_vec


# In[20]:


if __name__ == '__main__':
    data = pd.read_csv('cnews.test.txt', names=['title', 'content'], sep='\t')  # (10000, 2)
    data = data.head(50)
 
    stopword = read_stopword('stopword.txt')
    data = cut_data(data, stopword)
 
    fea_vec = text_vec(data)
    print(fea_vec)


# In[ ]:




