
# coding: utf-8

# 
# # やりたいこと
# ## その1  記事をクラスターに分ける
# ### 
# 
# 
# ## その2 記事をトピックに分ける
# ### 
# 

# # データ読み込み

# ## ライブラリ読み込み

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


# ## データ読み込み

# - データの特徴
# 
#   - 分析用サーバを使って、nekolog-dで分かち書きしたデータを読みこむ
#   - 記事id, 分かち書きしたデータが入ったデータ（）
#   - 4000記事×3000文字以上
#   
# - データサンプル
# 
#         id wakachi    
#          1  ページ まとめ ニート 現在・・・         
#          2  ページ まとめ 退職 理由 前 職・・・   
#          3  ページ まとめ ハローワーク 正式名称 公共職業安定所 ・・・

# In[110]:


df = pd.read_pickle('columns_wakachi_2018-10-03.pickle')


# In[111]:


df = df[['id','wakachi']]
df


# # 前処理

# ## 単語の頻度を把握

# In[112]:


# ランダムサンプリング
# 全部かけようと思ったら重すぎたので1割のデータに絞った
df_freq_sample = df.sample(frac=0.1)
freq_sample_list = list(df_freq_sample['wakachi'])
freq_sample = ''.join(freq_sample_list)
freq_sample


# In[78]:


# 頻度をカウントする
import collections
noun_cnt = collections.Counter(freq_sample.split())
noun_cnt

nouns_data = pd.DataFrame.from_dict(noun_cnt, orient='index').reset_index()
nouns_data.columns = ['nouns', 'count']
nouns_data = nouns_data.sort_values(by=["count"], ascending=None)
nouns_data['ratio'] = nouns_data['count']/nouns_data['count'].sum() 
nouns_data = nouns_data.reset_index(drop=True)


# In[79]:


nouns_data


# In[80]:


# 上位10%の単語で80%を占めている
nouns_data[nouns_data.index < len(nouns_data)*0.1].ratio.sum()


# ## 高頻度で不要語を選定して除去

# In[81]:


nouns_data.head(40)


# In[82]:


import sys
import re
stop_word_list = ["ページ","まとめ","こと", "の", "よう","人","自分", "場合", "者","方", "nbsp", "ため", "的" ,"もの", "時", "中","hellip","とき"]
stop_word_list = set(stop_word_list)


# In[83]:


def stop_word(documents):
    texts = [word for word in documents.lower().split() if word not in stop_word_list ]
    texts = " ".join(texts)
    return texts


# In[113]:


df["wakachi_del_stopword"] = list(map(lambda text:stop_word(text) , df.wakachi))
df["wakachi_del_stopword"]


# # トピック分類

# ## 実装

# In[85]:


from pathlib import Path
from janome.charfilter import *
from janome.analyzer import Analyzer
from janome.tokenizer import Tokenizer
from janome.tokenfilter import *
from gensim import corpora, models
import  gensim as  gensim
import pickle


# In[86]:


#データフレームのテキストデータをもとに、gensimで読み込み可能なフォーマットに変換するための関数
def nested_list(strings):
    words = strings.split(" ")
    return words


# In[108]:


df


# In[114]:


# トピック分類の準備
df["wakachi_list"] = list(map(lambda text:nested_list(text),df.wakachi_del_stopword))

dictionary = gensim.corpora.Dictionary(df['wakachi_list'])
corpus = [dictionary.doc2bow(doc) for doc in df['wakachi_list']]

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

with open('corpus_tfidf.dump', mode='wb') as f:
    pickle.dump(corpus_tfidf, f)


# In[89]:


#トピック割合を格納するためのデータフレームを作成する関数
def making_topic_detaframe(integer, n):
    topic_table = pd.DataFrame(index=n.index)
    for topic_number in range(integer):
        column_name = "topic" + str(topic_number)
        topic_table[column_name] = 0
    return topic_table

#トピック割合を格納する関数
def topic_ratio_extract(corpus, topic_table):
    n_topic = len(topic_table.columns)
    i = 0
    for bow in corpus:
        t = lda.get_document_topics(bow)
        
        for each_topic in t:
            for topic_id in range(n_topic):
                if each_topic[0] == topic_id:
                    topic_table.iloc[i, topic_id]  = each_topic[1]
                
        i = i + 1
    return topic_table


# In[95]:


# トピック分類

#最適数は課題。一旦いくつか試してみて、納得感があるのはこれくらい。
the_number_of_topic = 30

lda = gensim.models.ldamodel.LdaModel(
                            corpus=corpus,
                            alpha='auto', 
                            eta = 'auto',
                            num_topics=the_number_of_topic,
                            id2word=dictionary
)

#トピックの数に応じたデータフレームを作成
topic_table = making_topic_detaframe(the_number_of_topic, df)

#推定したトピックごとの文書ごとの確率を格納
topic_table = topic_ratio_extract(corpus, topic_table)

#元データとトピックテーブルを結合
df_numeric_with_topic = pd.concat([df, topic_table], axis=1)

#実数に変換
df_numeric_with_topic[list(topic_table.columns)] = df_numeric_with_topic[list(topic_table.columns)].astype("float")


# In[96]:


df_numeric_with_topic


# In[97]:


topic_table


# In[100]:


for i in range(the_number_of_topic):
    print('【','TOPIC:', i, '】', lda.print_topic(i))


# ## トピックモデルの評価
# https://qiita.com/icoxfog417/items/7c944cb29dd7cdf5e2b1

# In[104]:


from gensim import models

topic_range = range(2, 5)
test_rate = 0.2

def split_corpus(c, rate_or_size):
    import math
    
    size = 0
    if isinstance(rate_or_size, float):
        size = math.floor(len(c) * rate_or_size)
    else:
        size = rate_or_size
    
    # simple split, not take sample randomly
    left = c[:-size]
    right = c[-size:]
    
    return left, right

def calc_perplexity(m, c):
    import numpy as np
    return np.exp(-m.log_perplexity(c))

def search_model(c, rate_or_size):
    most = [1.0e6, None]
    training, test = split_corpus(c, rate_or_size)
    print("dataset: training/test = {0}/{1}".format(len(training), len(test)))

    for t in topic_range:
        m = models.LdaModel(corpus=training, id2word=dictionary, num_topics=t, iterations=250, passes=5)
        p1 = calc_perplexity(m, training)
        p2 = calc_perplexity(m, test)
        print("{0}: perplexity is {1}/{2}".format(t, p1, p2))
        
        if p2 < most[0]:
            most[0] = p2
            most[1] = m
    
    return most[0], most[1]

perplexity, model = search_model(corpus, test_rate)
print("Best model: topics={0}, perplexity={1}".format(model.num_topics, perplexity))


# ※記事から引用
# 
# >この評価には、パープレキシティという指標を用います。
# パープレキシティの逆数が文書中の単語の出現を予測できる度合いを示しており、
# よって最高は1で、モデルの精度が悪くなるほど大きな値になります(2桁ならよし、
# 3桁前半でまあまあ、それ以後は悪い、という感じで、
# 1桁の場合は逆にモデルやパープレキシティの算出方法に誤りがないか見直した方がよいです)。
# 
# 

# # クラスターわけ

# ## 実装

# In[115]:


wakachi = df[['wakachi_del_stopword']]

# 配列に一応変換
docs = np.array(wakachi)
docs


# ### vectorizerの生成

# In[117]:


# 1文字の単語もストップワードにしない（今回は分かち書きのときに名詞だけのデータにしているため）
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')


# ###### メモφ(..)
# TfidfVectorizer
# -  use_idf = 逆文章頻度の再重み付けを有効にするか否か
# - token_pattern = トークンとして認識するものをどの単位にするか。
#                     デフォルトは一文字のトークンが除外されてしまう。
#                     2文字以上の英数字はトークンとみなされるが、1文字の場合はストップワードとして除外される
#                     除外されたくなければ、token_pattern=u'(?u)\\b\\w+\\b' 　などと記載する。
#     ※参考　http://otknoy.hatenablog.com/entry/2015/10/11/200650
#     
# -  http://ailaby.com/tfidf/#id2_1 
# -  TfidfVectorizer http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# -  https://qiita.com/katryo/items/f86971afcb65ce1e7d40 によると　TfidfVectorizerを使うということがわかる
# 　

# In[118]:


vectorizer


# In[119]:


vecs = vectorizer.fit_transform(docs.ravel())
#https://stackoverflow.com/questions/26367075/countvectorizer-attributeerror-numpy-ndarray-object-has-no-attribute-lower
vecs.toarray()


#  -  ravel = Return the flattened underlying data as an ndarray

# ### クラスターを指定する

# In[120]:


cluster_num = 50
clusters = KMeans(n_clusters=cluster_num, random_state=0).fit_predict(vecs)


# In[121]:


for doc,cls in zip(docs, clusters):
    print(cls, doc)


# In[122]:


clusters

# dfに変換
cluster_df = pd.DataFrame(clusters)
cluster_df.columns = ['cluster']


# ## クラスターの評価

# ### クラスターごとの記事数を確認

# In[123]:


cluster_df.iloc[:,0].value_counts()


# ### 記事データとクラスターを結合し、目視で確認

# In[124]:


columns_cluster = pd.concat([cluster_df, df], axis=1)
columns_cluster.sort_values(by='cluster')


# In[16]:


columns_cluster.to_pickle('useful_cluster_2018-10-05.pickle')


# # 各クラスターに自動でタグをつける

# ## 関数作成
# 　- https://programminghistorian.org/en/lessons/counting-frequencies

# In[125]:


# リストか文字列を入れると辞書型で単語と出現回数がでてくる関数
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))


# In[126]:


# 辞書を出現頻度降順に並べ替える関数
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


# ## テスト
# - clusterを一つ絞ってやってみた

# taishoku = columns_cluster[columns_cluster['cluster'] == 25]
# temp_taishoku = taishoku[['wakachi']]
# 
# - for文を回すためにindexを綺麗にする
# temp_taishoku = temp_taishoku.reset_index(drop = True)
# 
# test = ['aaa', 'bbb', 'ccc', 'ccc']
# 
# -  strを関数に入れるとできる
# a = wordListToFreqDict(temp_taishoku.wakachi[52].split())
# b = sortFreqDict(a)
# b[0:5]
# 
# type(temp_taishoku.wakachi[2])
# 
# 
# -  200件位すぐ回った
# 
# for i in range(0,len(temp_taishoku.wakachi)):
#     a = wordListToFreqDict(temp_taishoku.wakachi[i].split())
#     b = sortFreqDict(a)
#     print(b[0:5])

# ## クラスターごとにまとめる

# In[140]:


columns_cluster


# In[141]:


columns_cluster2 = columns_cluster[['cluster','wakachi_del_stopword']]
cluster_group = columns_cluster2.groupby('cluster').agg({'wakachi_del_stopword':  lambda x: ','.join(x)})


# In[329]:


# クラスターの数を決める
cluster_num = 50


# In[285]:


# 空のデータフレームをつくる
min_text_output = pd.DataFrame([], columns=['min_text'])
min_text_output

for i in range(0,len(cluster_group['wakachi_del_stopword'])):
    min_text = (cluster_group['wakachi_del_stopword'][i].split())[:50000]
    str2 = ' '.join(min_text)
    min_text_df = pd.DataFrame([str2],columns=['min_text'])
    min_text_output = min_text_output.append(min_text_df)

cluster_mintext = min_text_output.reset_index(drop=True)


# In[317]:


list(cluster_mintext.min_text[i].split())


# In[333]:


###タイトルの単語数をカウントして結合してデータフレームにする

#空テーブル作成
count = []
stock_datatable = pd.DataFrame(columns=["tag"])

###タイトルの単語数をカウントして結合
for i in range(0,cluster_num):
    counter = collections.Counter(list(cluster_mintext.min_text[i].split()))
    values, counts = zip(*counter.most_common(5))
# print(list(values))
    aa = list(values)
    countstr =','.join(map(str, aa))

    aaaa = pd.DataFrame([countstr],columns=["tag"])
    stock_datatable = stock_datatable.append(aaaa)

stock_datatable.head(3)
stock_datatable = stock_datatable.reset_index(drop=True)


# In[334]:


stock_datatable


# #### 頻度が高い単語をカウントする
# for i in range(0,len(cluster_mintext.min_text)):
#     a = wordListToFreqDict(cluster_mintext.min_text[i].split())
#     b = sortFreqDict(a)
#     tag = b[0:5]
#     print(tag)    
