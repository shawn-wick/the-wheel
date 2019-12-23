# coding = utf-8
'''
Author: Albert Shawn
Email: weichangtsing@163.com

date = 2019/11/8 2:20 下午
dsec
'''
import csv
import json
import logging
import math
import re
import time
import warnings
from collections import Counter

import jieba
import jieba.analyse
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import numpy as np
import pymysql
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models import word2vec
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def jieba_tokenize(text):
    return jieba.lcut(text, )


def jieba_pseg(text):
    pseg_list = pseg.cut(text)
    return pseg_list


def clean(doclist):
    for doc in doclist:
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        r2 = u'\s+;'
        doc_ = re.sub(r1, '', doc)  # 过滤内容中的各种标点符号
        doc_ = re.sub(r2, '', doc_)
        doclist[doclist.index(doc)] = doc_
    stopwords = []
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    texts = [[word for word in jieba_tokenize(pat) if word not in stopwords and len(word) > 1] for pat in doclist if
             len(pat) > 10]
    allWords = [word for data in texts for word in data]

    # 去掉停用词

    wordCount = Counter(allWords)  # 统计词频
    sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

    # 去除低频词
    words = [item[0] for item in sortWordCount if item[1] >= 4]
    texts = [[word for word in text if word in words] for text in texts]

    return texts


def pseg_process(doclist):
    word_pseg_dic = {}
    n_pseg_list = []
    v_pseg_list = []
    c_pseg_list = []
    for i in doclist:
        a = jieba_pseg(i)
        for k in a:
            word_pseg_dic[k.word] = k.flag
    for key in word_pseg_dic:
        if word_pseg_dic[key] in v_type:
            v_pseg_list.append(key)
    for key in word_pseg_dic:
        if word_pseg_dic[key] in n_type:
            n_pseg_list.append(key)

    # with open('词性处理文件.txt','w') as f:
    #     for i in clean_pseg_list:
    #         f.write(str(i))
    #         f.write('\n')
    return v_pseg_list, n_pseg_list


def BasicLDA(doclist, num_topics):
    start = time.clock()
    num_topics = num_topics
    texts = clean(doclist)
    print(texts[1])
    # frequency = {}
    # for text in texts:
    #     for token in text:
    #         if token not in frequency:
    #             frequency[token] = 0
    #         else:
    #             frequency[token] += 1
    dictionary = corpora.Dictionary(texts)
    size_dictionary = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=500, passes=10, iterations=100)
    topics = []
    for i in lda.show_topics(num_topics=-1, num_words=20):
        print(i)
        topics.append(i)

    for i in lda.get_document_topics(corpus):  # i是按照词袋中的顺序，每个文档的主题分布
        s = str(i)
        pattern1 = r'\((\d+),'
        a = re.findall(pattern1, s)
        print(a)  # 匹配出每个文档的包含的主题标签

        word_list = []  # 存放当前文档包含的所有的主题
        for idx in a:  # 取主题号
            w = topics[int(idx)]  # 取主题词分布
            word_list.append(w)  # 按照主题标签， 把对应主题的词分布 ，按照顺序存起来

        l = [list(k)[1] for k in i]  # list(k)[1] 每个主题的取概率
        doc2top = {}
        for num in range(len(l)):
            doc2top[l[num]] = word_list[num]

        print(doc2top)
        break
        # print(list(chain.from_iterable(zip(l, word_list))))

    elapsed = time.clock() - start
    return lda, corpus, dictionary, size_dictionary, elapsed


def sim_wordbag(texts, modelname, v_pseg_list, n_pseg_list):
    dictionary = corpora.Dictionary(texts)
    token2id = dictionary.token2id
    id2token = {v: k for k, v in token2id.items()}
    lda_word_list = [id2token[key] for key in id2token]
    model = word2vec.Word2Vec.load(modelname)

    n_sim_dic = {}
    v_sim_dic = {}
    n_sim_list = []
    v_sim_list = []
    v_vec_list = []
    n_vec_list = []
    a = model.wv.vocab.keys()
    v_set = set(a) & set(v_pseg_list) & set(lda_word_list)
    for word in v_set:
        v_sim_list.append(word)
        v_vec_list.append(model[word])

        k = model.most_similar(word, topn=10)
        list = []
        list.append(word)
        for word_vec_pair in k:
            # if word_vec_pair[0] in lda_word_list:
            list.append(word_vec_pair[0])
        # else:
        # pass
        v_sim_dic[word] = set(list)
    n_set = set(a) & set(n_pseg_list) & set(lda_word_list)
    for word in n_set:
        n_sim_list.append(word)
        n_vec_list.append(model[word])
        k = model.most_similar(word, topn=6)
        list = []
        list.append(word)
        for word_vec_pair in k:
            # if word_vec_pair[0] in lda_word_list:
            list.append(word_vec_pair[0])
        # else:
        # pass
        n_sim_dic[word] = set(list)

    n_vec_list = np.mat(n_vec_list)
    v_vec_list = np.mat(v_vec_list)
    print('clustering...')
    kmeans = KMeans(n_clusters=850, max_iter=5, n_jobs=-1, random_state=0)
    kmeans.fit(n_vec_list)
    label = kmeans.labels_
    word2label = {}
    for i in range(len(n_sim_list)):
        word2label[n_sim_list[i]] = label[i]

    center = kmeans.cluster_centers_
    # label是按照样本顺序进行存放的  cluster center是按照聚类的顺序存放的
    list = []
    for index, cluster_id in enumerate(kmeans.labels_):
        list.append([n_sim_list[index], cluster_id])
    dic = {}
    for i in list:
        if i[1] not in dic.keys():
            dic[i[1]] = [i[0]]
        else:
            dic[i[1]].append(i[0])

    '''对kmeans结果进行筛选'''
    listdic = {}  # 存放词簇中心词
    for key in dic:
        dist_list = []
        w = dic[key][0]
        label_num = word2label[w]
        c = center[label_num]  # 取聚类中心的坐标
        for w in dic[key]:
            dist = np.sqrt(np.sum(np.square(np.array(model[w]) - np.array(c))))
            dist_list.append([dist, dic[key].index(w)])
        centers = []
        try:
            for i in range(2):
                center_w = sorted(dist_list, key=lambda item: item[0])[i][1]
                centers.append(center_w)
            listdic[word2label[w]] = [dic[key][center_w] for center_w in centers]
        except:
            #print('类别中词不够', dic[key])
            center_w = sorted(dist_list, key=lambda item: item[0])[0][1]
            centers.append(center_w)
            listdic[word2label[w]] = [dic[key][center_w] for center_w in centers]
    n_token2tag_dict = {}
    new_tag_list = []
    n_tag_dic = {}
    for key in dic:
        # print(dic[key])
        sim_word1 = []
        not_sim_word = []
        sim_word2 = []

        w = dic[key][0]  # 获取当前词簇中的一个词，来判断属于哪个簇
        label_num = word2label[w]
        center_words = listdic[label_num]
        print(dic[key])
        for w in dic[key]:  # 这个相似词典是生成的原始的，key是词，value是topn
            for center_word in center_words:
                s = n_sim_dic[w] & n_sim_dic[center_word]  # 取这两个词相似词的交集，重叠词
                if len(s) >= 2 or len(set(w) & set(center_word)) > 0:  # 5作为一个判断的阈值，如果满足重叠的词的个数的阈值，那么判断两个词相似。将这两个词放进一个词中
                    sim_word1.append(w)
                    break
                elif len(set(w) & set(center_word)) >= 2:
                    sim_word1.append(w)
                    break
                else:
                    not_sim_word.append(w)

        for wi in not_sim_word:
            for wj in sim_word1:
                s = n_sim_dic[wi] & n_sim_dic[wj]
                if len(s) >= 2 or len(set(wi) & set(wj)) > 0:
                    sim_word2.append(wi)
                    break

        for i in sim_word2:
            if i in not_sim_word:
                not_sim_word.remove(i)

        sim_word1.extend(sim_word2)
        sim_word = set(sim_word1)

        new_tag_list.append(sim_word)
        for w in set(not_sim_word):
            new_tag_list.append([w])

    for num, words in enumerate(new_tag_list):
        n_tag_dic[num] = words

    kmeans = KMeans(n_clusters=350, max_iter=5, n_jobs=-1, random_state=0)
    kmeans.fit(v_vec_list)
    label = kmeans.labels_
    word2label = {}
    for i in range(len(v_sim_list)):
        word2label[v_sim_list[i]] = label[i]

    center = kmeans.cluster_centers_
    # label是按照样本顺序进行存放的  cluster center是按照聚类的顺序存放的
    list = []
    for index, cluster_id in enumerate(kmeans.labels_):
        list.append([v_sim_list[index], cluster_id])
    dic = {}
    for i in list:
        if i[1] not in dic.keys():
            dic[i[1]] = [i[0]]
        else:
            dic[i[1]].append(i[0])

    '''对kmeans结果进行筛选'''
    listdic = {}  # 存放词簇中心词
    for key in dic:
        dist_list = []
        w = dic[key][0]
        label_num = word2label[w]
        c = center[label_num]  # 取聚类中心的坐标
        for w in dic[key]:
            dist = np.sqrt(np.sum(np.square(np.array(model[w]) - np.array(c))))
            dist_list.append([dist, dic[key].index(w)])
        centers = []
        try:
            for i in range(2):
                center_w = sorted(dist_list, key=lambda item: item[0])[i][1]
                centers.append(center_w)
            listdic[word2label[w]] = [dic[key][center_w] for center_w in centers]
        except:
            # print('类别中词不够', dic[key])
            center_w = sorted(dist_list, key=lambda item: item[0])[0][1]
            centers.append(center_w)
            listdic[word2label[w]] = [dic[key][center_w] for center_w in centers]
    new_tag_list = []
    v_tag_dic = {}
    for key in dic:
        # print(dic[key])
        sim_word1 = []
        not_sim_word = []
        sim_word2 = []

        w = dic[key][0]  # 获取当前词簇中的一个词，来判断属于哪个簇
        label_num = word2label[w]
        center_words = listdic[label_num]
        for w in dic[key]:  # 这个相似词典是生成的原始的，key是词，value是topn

            for center_word in center_words:

                s = v_sim_dic[w] & v_sim_dic[center_word]  # 取这两个词相似词的交集，重叠词
                if len(s) >= 2 or len(set(w) & set(center_word)) > 0:  # 5作为一个判断的阈值，如果满足重叠的词的个数的阈值，那么判断两个词相似。将这两个词放进一个词中
                    sim_word1.append(w)
                    break
                elif len(set(w) & set(center_word)) >= 1:

                    sim_word1.append(w)
                    break
                else:
                    not_sim_word.append(w)
        for wi in not_sim_word:
            for wj in sim_word1:
                s = v_sim_dic[wi] & v_sim_dic[wj]
                if len(s) >= 2 or len(set(wi) & set(wj)) > 0:
                    sim_word2.append(wi)
                    break

        for i in sim_word2:
            if i in not_sim_word:
                not_sim_word.remove(i)

        sim_word1.extend(sim_word2)
        sim_word = set(sim_word1)

        new_tag_list.append(sim_word)
        for w in set(not_sim_word):
            new_tag_list.append([w])

    for num, words in enumerate(new_tag_list):
        v_tag_dic[num] = words


    return n_tag_dic, v_tag_dic


def LDA(texts, num_topics, token2tag, token2tag_dic, dic):
    num_topics = num_topics
    '''将输入文本中的所有的词，按照相似性，变成tag_num'''
    '''将文本中的词替换称编号，如果词不在字典中，则增加到字典中'''
    # 获取未替换的文本的tfidf
    dictionary = corpora.Dictionary(texts)
    token2id = dictionary.token2id
    id2token = {v: k for k, v in token2id.items()}
    # dictionary.save('patent_dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    # corpora.MmCorpus.serialize('patent_corpuse.mm', corpus)
    tfidf = models.TfidfModel(corpus)
    # tfidf.save('patent_tfidf.model')
    corpus_tfidf = tfidf[corpus]
    sort_keywords = []
    # 将tfidf进行降序排序，获取前三的tfidf的词作为关键词，否则就取一个。
    for doc in corpus_tfidf:
        sorttfidf = sorted(doc, key=lambda x: x[1], reverse=True)
        n = 0
        if len(sorttfidf) >= 3:
            sort_keywords.append([id2token[sorttfidf[0][0]], id2token[sorttfidf[1][0]], id2token[sorttfidf[2][0]]])
        elif len(sorttfidf) == 2:
            sort_keywords.append([id2token[sorttfidf[0][0]], id2token[sorttfidf[1][0]]])
        elif len(sorttfidf) == 1:
            sort_keywords.append([id2token[sorttfidf[0][0]]])

    for text in texts:
        for word in text:
            text[text.index(word)] = str(token2tag[word])
    topic = {}
    for key, item in token2tag_dic.items():
        topicword = ' '.join(list(item))
        # print(topicword)
        topic[str(key)] = topicword
        insert = "insert into topicword_info(主题词ID,主题词) value (%s,%s)"
        # print(insert,[key,topicword])
        cur.execute(insert, [key, topicword])
        con.commit()
    start = time.clock()
    dictionary = corpora.Dictionary(texts)
    dictionary.save('patent_dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('patent_corpuse.mm', corpus)
    tfidf = models.TfidfModel(corpus)
    tfidf.save('patent_tfidf.model')
    corpus_tfidf = tfidf[corpus]
    # sort_keywords = []
    # #将tfidf进行降序排序，获取前三的tfidf的词作为关键词，否则就取一个。
    # for doc in corpus_tfidf:
    #     sorttfidf = sorted(doc, key=lambda x: x[1], reverse=True)
    #     n = 0
    #     if len(sorttfidf)>=3:
    #         sort_keywords.append([sorttfidf[0][0],sorttfidf[1][0],sorttfidf[2][0]])
    #     else:
    #         sort_keywords.append([sorttfidf[0][0]])
    # print('关键词',sort_keywords)

    size_dictionary = len(dictionary)  # 所有单词的长度
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=500, passes=100, iterations=100)
    lda.save('patent_lda.model')
    topic1 = {}  # 用来存放每个主题的词分布 按照从0到最大的顺序
    for i in lda.show_topics(num_topics=-1, num_words=20):
        s = str(i)
        pattern1 = r'"(\d+)"'
        a = re.findall(pattern1, s)  # 匹配出 词簇tag
        topic1[i[0]] = [topic[a[0]], topic[a[1]], topic[a[2]], topic[a[3]], topic[a[4]]] # 用来存放每个主题的词分布 按照从0到最大的顺序
        insert = 'insert into topic_info(主题ID,主题词ID1,主题词ID2,主题词ID3,主题词ID4) values(%s,%s,%s,%s,%s)'
        # print(insert,[i[0],a[0],a[1],a[2],a[3]])
        cur.execute(insert, [i[0], a[0], a[1], a[2], a[3]])
        con.commit()
        # word_list = []
        # for i in a:
        #     w = token2tag_dic[int(i)]
        #     word_list.append(w)
        # pattern2 = r'"\d+"'
        # st = re.split(pattern2, s)
        # topics.append(list(chain.from_iterable(zip(st, word_list))))
        # print(list(chain.from_iterable(zip(st, word_list))))
    c = 0
    topics = []
    ppp = []
    for i in lda.get_document_topics(corpus):  # i是按照词袋中的顺序，每个文档的主题分布
        s = str(i)
        ppp.append([])
        vec = [0.0] * 30
        for j in i:
            a = int(j[0])
            vec[a] = j[1]
            ppp[-1].extend([j[1],topic1[j[0]]])
        topics.append(vec)
        # print(vec)
        pattern1 = r'\((\d+),'
        a = re.findall(pattern1, s)
        # print(a)  #匹配出每个文档的包含的主题标签
        keyword = sort_keywords[c]
        # print(keyword)
        # guanjianci = ' '.join([str(token2tag_dic[int(j)]) for j in keyword]) #[1,2,3] 按照索引把
        guanjianci = ' '.join(keyword)  # [1,2,3] 按照索引把
        detail = dic[c]
        name, abstract, id = detail['专利名称'], detail['专利摘要'], detail['专利号']
        if len(a) >= 4:
            insert = 'insert into patent_info(专利名,专利摘要,专利号,关键词,主题ID1,主题ID2,主题ID3,主题ID4,主题向量) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)'
            cur.execute(insert, [name, abstract, id, guanjianci, a[0], a[1], a[2], a[3], str(vec)])
            con.commit()
            c += 1
        else:
            if len(a) == 1:
                insert = 'insert into patent_info(专利名,专利摘要,专利号,关键词,主题ID1,主题向量) values(%s,%s,%s,%s,%s,%s)'
                cur.execute(insert, [name, abstract, id, guanjianci, a[0], str(vec)])
                con.commit()
                c += 1
            elif len(a) == 2:
                insert = 'insert into patent_info(专利名,专利摘要,专利号,关键词,主题ID1,主题ID2,主题向量) values(%s,%s,%s,%s,%s,%s,%s)'
                cur.execute(insert, [name, abstract, id, guanjianci, a[0], a[1], str(vec)])
                con.commit()
                c += 1
            elif len(a) == 3:
                insert = 'insert into patent_info(专利名,专利摘要,专利号,关键词,主题ID1,主题ID2,主题ID3,主题向量) values(%s,%s,%s,%s,%s,%s,%s,%s)'
                cur.execute(insert, [name, abstract, id, guanjianci, a[0], a[1], a[2], str(vec)])
                con.commit()
                c += 1
    for i in ppp:
        print(i)
    with open('lda文本的主题词', 'w', encoding='utf8') as f:
        f.write(str(ppp))
    # word_list = [] #存放当前文档包含的所有的主题
    # for idx in a: #取主题号
    #     w = topics[int(idx)] #取主题词分布
    #     word_list.append(w) #按照主题标签， 把对应主题的词分布 ，按照顺序存起来
    #
    # l = [list(k)[1] for k in i] #list(k)[1] 每个主题的取概率
    # doc2top = {}
    # for num in range(len(l)):
    #     doc2top[l[num]]  = word_list[num]
    #
    # print(doc2top)
    # print(list(chain.from_iterable(zip(l, word_list))))
    # doctopic = []
    # for i in lda.get_document_topics(corpus)[:]:
    #     listj = []
    #     for j in i:
    #         listj.append(j[1])
    #     bz = listj.index(max(listj))
    #
    #     k = i[bz][0]
    #     doctopic.append(k)
    with open('lda_topic', 'w', encoding='utf8') as f:
        f.write(str(topics))
    elapsed = time.clock() - start
    return lda, corpus, dictionary, size_dictionary, elapsed


def basicLDA(texts, num_topics, ):
    num_topics = num_topics
    # 获取未替换的文本的tfidf
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    start = time.clock()
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, chunksize=500, passes=100, iterations=100)
    lda.save('patent_basiclda.model')
    topic = {}  # 用来存放每个主题的词分布 按照从0到最大的顺序
    for i in lda.show_topics(num_topics=-1, num_words=20):
        s = str(i)
        pattern1 = "[\u4e00-\u9fa5]+"
        a = re.findall(pattern1, s)  # 匹配出 词簇tag
        topic[i[0]] = i[1]  # 用来存放每个主题的词分布 按照从0到最大的顺序

    #     # word_list = []
    #     # for i in a:
    #     #     w = token2tag_dic[int(i)]
    #     #     word_list.append(w)
    #     # pattern2 = r'"\d+"'
    #     # st = re.split(pattern2, s)
    #     # topics.append(list(chain.from_iterable(zip(st, word_list))))
    #     # print(list(chain.from_iterable(zip(st, word_list))))
    c = 0
    ppp = []
    topics = []
    for i in lda.get_document_topics(corpus):  # i是按照词袋中的顺序，每个文档的主题分布
        s = str(i)
        vec = [0.0] * 30
        ppp.append([])
        for j in i:
            a = int(j[0])
            vec[a] = j[1]
            ppp[-1].extend([j[1],topic[j[0]]])
        topics.append(vec)
    for i in ppp:
        print(i)
    with open('basiclda文本的主题词', 'w', encoding='utf8') as f:
        f.write(str(ppp))
        # print(vec)
        # pattern1 = r'\((\d+),'
        # a = re.findall(pattern1, s)
        # print(a)  #匹配出每个文档的包含的主题标签
        # print(keyword)
        # guanjianci = ' '.join([str(token2tag_dic[int(j)]) for j in keyword]) #[1,2,3] 按照索引把

        # word_list = [] #存放当前文档包含的所有的主题
        # for idx in a: #取主题号
        #     w = topics[int(idx)] #取主题词分布
        #     word_list.append(w) #按照主题标签， 把对应主题的词分布 ，按照顺序存起来
        #
        # l = [list(k)[1] for k in i] #list(k)[1] 每个主题的取概率
        # doc2top = {}
        # for num in range(len(l)):
        #     doc2top[l[num]]  = word_list[num]
        #
        # print(doc2top)
        # print(list(chain.from_iterable(zip(l, word_list))))
    # doctopic = []
    # for i in lda.get_document_topics(corpus)[:]:
    #     listj = []
    #     for j in i:
    #         listj.append(j[1])
    #     bz = listj.index(max(listj))
    #
    #     k = i[bz][0]
    #     doctopic.append(k)
    with open('basiclda_topic', 'w', encoding='utf8') as f:
        f.write(str(topics))
    elapsed = time.clock() - start
    return lda, corpus, dictionary, elapsed


def cluster(v_vec_list, m, n, k):
    K = range(m, n, k)
    mean_distortions = []
    sc_scores = []
    for k in K:
        kmeans = KMeans(n_clusters=k, n_jobs=-1)
        kmeans.fit(v_vec_list)
        mean_distortions.append(
            sum(np.min(cdist(v_vec_list, kmeans.cluster_centers_, metric='euclidean'), axis=1)) / v_vec_list.shape[0])
        sc_score = silhouette_score(v_vec_list, kmeans.labels_, metric='euclidean')  # 评价
        sc_scores.append(sc_score)
        print('完成一次聚类')
        # plt.title(u'K = %s, 轮廓系数= %0.03f' %(k, sc_score))
    plt.plot(K, mean_distortions, color="red", linewidth=2)
    plt.xlabel('K')
    plt.ylabel('SSE Score')

    plt.show()
    # plt.figure()
    plt.plot(K, sc_scores, color="red", linewidth=2)
    plt.xlabel('K')
    plt.ylabel('SC Score')  # kmeans = KMeans(n_clusters=100, n_jobs=-1)

    plt.show()

    # kmeans = KMeans(n_clusters=400)
    # kmeans.fit(v_vec_list)
    #
    # a = kmeans.cluster_centers_[0]
    # print(a)
    #
    # list = []
    # for index, cluster_id in enumerate(kmeans.labels_):
    #     list.append([v_sim_list[index],cluster_id])
    # dic = {}
    # for i in list:
    #     if i[1] not in dic.keys():
    #         dic[i[1]]  =  [i[0] ]
    #     else:
    #         dic[i[1]].append(i[0])
    #
    # print(sorted(dic.items(),key= lambda item: item[0],reverse=False))
    #


def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):  # 计算困惑度
    """calculate the perplexity of a lda-model"""
    # dictionary : {7822:'deferment', 1841:'circuitry',19202:'fabianism'...]
    print('the info of this ldamodel: \n')
    print('num of testset: %s; size_dictionary: %s; num of topics: %s' % (len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word2prob_list = []  # 用来储存每个主题下关键词的概率y:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id,
                                         size_dictionary)  # show_topics可以得到每个主题下面的的关键词[[word,probability],....],显示的是size_dictionary，每个主题下面所有的单词的概率
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability  # 生成一个字典，每个主题一个字典，字典里面是{word1: probability1,...}，覆盖所有的单词
        topic_word2prob_list.append(dic)  # 将每个主题各自的词-概率字典存进列表
    doc_topics_list = []  # store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_list.append(ldamodel.get_document_topics(doc,
                                                            minimum_probability=0))  # 获得训练集中每一篇文章对应的属于每个主题的概率# [[(topic1,prob),(topic2,prob)....],[...]....]
    testset_word_num = 0
    for i in range(len(testset)):  # len是corpus 中有多少篇文章
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]  # 依次取testset中的每个文本
        doc_word_num = 0  # the num of words in the doc 计数用
        for word_id, num in doc:  # corpus中是（id,频率）的组合
            prob_word = 0.0  # the probablity of the word
            doc_word_num += num
            word = dictionary[word_id]  # 从dictionary中取按照word_id取词
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_list[i][topic_id][1]  # 获得每个doc对应的每个主题的概率
                prob_topic_word = topic_word2prob_list[topic_id][word]  # 获取每个主题下，每个词的对应的概率
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("the perplexity of this ldamodel is : %s" % prep)
    return prep


# def test():
#     labels = []
#     datasetpath = "数据集/answer"
#     pattern = u'[C\d+-]'
#     dirs = [re.sub(pattern, '', i) for i in os.listdir(datasetpath)]
#     dirs.sort()
#     for i in dirs:
#         labels.append(i)
#     train_data = []
#     num = 0
#     filenum = 0
#     list = os.listdir(datasetpath)  # 列出文件夹下所有的文件夹名
#     for i in range(0, len(list)):
#         doc_path = os.path.join(datasetpath, list[i])
#         filenum+=1
#         if filenum>3:
#             break
#         else:
#             pass
#         list_documents = os.listdir(doc_path)
#         for j in range(0, len(list_documents)): #在同一类的文本的文件夹里
#             label = []
#             filepath = os.path.join(doc_path, list_documents[j])
#             if os.path.isfile(filepath):  # 你想对文件的操作
#                 label.append(num)
#                 with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
#                     print(filepath)
#                     train_data.append(f.read())
#         num += 1
#     texts = clean(train_data)  #这里的测试集，需要先分词，然后把clean后的文档（这个文档是一个列表，列表中的每一个元素的都是个列表，每个列表包含的一个篇文档，是分词，去停用词，筛选词性后的文档）
#     # clean_pseg_list = pseg_process(texts) #词性筛选后的合格的文档
#     for text in texts:
#         for word in text:
#             if word in token2tag_dict.keys(): #判断是不是词簇里的词
#                 text[text.index(word)] = 'tag' + str(token2tag_dict[word]) #如果是，打上相同的标签
#             else:
#                 pass
#
#     #得到了一个新的，词簇替换过后的测试文本
#     test_wordlist = [] #生成了训练文本的词汇表
#     for i in range(len(dictionary)): #这里的dictionary是包含tag的，用tag替换过的
#         test_wordlist.append(dictionary[i]) #获得训练集的词汇簇
#     new_texts = []
#     new_text = []
#     for text in texts:
#         for word in text:
#             if word in test_wordlist: #去除掉不是训练集的词
#                 new_text.append(word)
#             else:
#                 pass
#         new_texts.append(new_text)
#     #得到了和训练集字典一致的新测试文本


def graph_draw(topic, perplexity1, perplexity2, c_v1, c_v2, time1, time2):  # 做主题数与困惑度的折线图 现在需要画两条，对比图
    x = topic
    y = perplexity1
    z = perplexity2
    a = c_v1
    b = c_v2
    m = time1
    n = time2
    plt.plot(x, y, color="red", linewidth=2)
    plt.plot(x, z, color="green", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()

    plt.plot(x, a, color="red", linewidth=2)
    plt.plot(x, b, color="green", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Coherence score")
    plt.show()

    plt.plot(x, m, color="red", linewidth=2)
    plt.plot(x, n, color="green", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Train Time")
    plt.show()


if __name__ == '__main__':
    con = pymysql.connect(
        host='localhost',
        user='root',
        passwd='12345678',
        charset='utf8mb4',
    )
    cur = con.cursor()
    cur.execute('use lda')
    train_data = []
    # #filename = '../论文/p.csv'
    n_type = ['n', 'ns', 'nt', 'nz', 'nl', 'ng']
    v_type = ['v', 'vn']
    pseg_type = ['n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng', 'v', 'vn']
    # print('正在生成列表')
    filename = 'p.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        dic = []
        num = 0
        for i in reader:

            if i[18] == None or i[19] == None:
                pass
            else:
                det = {}
                det['专利名称'] = i[6]
                det['专利摘要'] = i[18] + i[19]
                det['专利号'] = i[1]
                dic.append(det) if len(jieba.lcut(''.join([det['专利名称'], det['专利摘要']]))) > 10 else dic
                num += 1
                if num > 500:
                    break
    print(dic)
    # train_data.append(eval(item))
    # train_data = clean(train_data) #获得的还是字符串
    # with open('clean潍柴.txt', 'w', encoding='utf8') as f:#存放去除了标点字母的文本
    #     for i in train_data:
    #         f.write(i)
    #         f.write('\n')
    # texts = filter_cut(train_data)
    # model = word2vec.Word2Vec(texts, sg=0, size=200, window=5, min_count=0, hs=0,workers=4)
    # model.save(modelname)
    # print(model.most_similar('齿轮',topn=10))
    # print(model.similarity('齿轮','啮合'))
    # print(model.most_similar('液压',topn=10))
    # start = time.clock()
    # elapsed = time.clock() - start
    # print('文本清理完成！用时：', elapsed)
    # print('正在进行词性处理...')
    # start = time.clock()
    # v_pseg_list,n_pseg_list = pseg_process(train_data)
    # with open('名词集','w',encoding='utf8') as f:
    #     f.write(str(n_pseg_list))
    # with open('动词集','w',encoding='utf8') as f:
    #     f.write(str(v_pseg_list))
    # elapsed = time.clock() - start
    # print('词性处理完成！用时：', elapsed)
    # print('正在生成词簇...')
    count = 0
    texts = []
    with open('filter_cut.txt', 'r', encoding='utf8') as f:
        for i in f.readlines():
            texts.append(eval(i)) if len(eval(i)) > 10 else texts
            count += 1
            if count > 500:
                break

    with open('名词集', 'r', encoding='utf8') as f:
        n_pseg_list = eval(f.readline())
    with open('动词集', 'r', encoding='utf8') as f:
        v_pseg_list = eval(f.readline())
    n_tag_dict, v_tag_dict = sim_wordbag(texts, modelname, v_pseg_list, n_pseg_list)
    # with open('名词融合结果','w',encoding='utf8') as f:
    #     f.write(json.dumps(n_tag_dict))
    # with open('动词融合结果','w',encoding='utf8') as f:
    #     f.write(json.dumps(v_tag_dict))
    token2tag_list = []
    tag2word_dic = {}
    token2tag = {}

    for key in n_tag_dict:
        token2tag_list.append(n_tag_dict[key])
    for key in v_tag_dict:
        token2tag_list.append(v_tag_dict[key])
    print('token2tag_list', token2tag_list)
    whole_list1 = [i for pat in token2tag_list for i in pat]
    whole_list2 = [i for text in texts for i in text]
    list1 = list(set(whole_list2).difference(set(whole_list1)))
    print('===================================')
    print(list1)
    for i in list1:
        token2tag_list.append([i])
    for num, item in enumerate(token2tag_list):
        tag2word_dic[num] = item
    print('token2tag_dic', tag2word_dic)
    for num in tag2word_dic:
        for w in tag2word_dic[num]:
            token2tag[w] = num
    print('token2tag', token2tag)
    with open('word2tag.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(token2tag))
    # get_clustercenter(v_sim_dic, n_sim_dic,c_sim_dic)

    c_v1 = []
    c_v2 = []
    lm_list = []
    topic_nums = []
    per1 = []
    per2 = []
    time1 = []
    time2 = []
    with open('train_texts', 'w', encoding='utf8') as f:
        f.write(str(texts))
    blda, bcorpus, bdictionary, btime = basicLDA(texts, 30)

    lda, corpus, dictionary, size_dictionary, time_ = LDA(texts, 30, token2tag, tag2word_dic, dic)

    for num_topics in range(5, 50, 5):
        topic_nums.append(num_topics)
        lda, corpus, dictionary, size_dictionary, time_ = LDA(texts, num_topics, token2tag)
        time1.append(time_)
        cm1 = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        c_v1.append(cm1.get_coherence())
        perplexity1 = perplexity(lda, corpus, dictionary, size_dictionary, num_topics)
        per1.append(perplexity1)

        blda, bcorpus, bdictionary, bsize_dictionary, btime = BasicLDA(train_datas, num_topics)
        time2.append(btime)
        cm2 = CoherenceModel(model=blda, corpus=bcorpus, dictionary=bdictionary, coherence='u_mass')
        c_v2.append(cm2.get_coherence())
        perplexity2 = perplexity(blda, bcorpus, bdictionary, bsize_dictionary, num_topics)
        per2.append(perplexity2)
    graph_draw(topic_nums, per1, per2, c_v1, c_v2, time1, time2)
