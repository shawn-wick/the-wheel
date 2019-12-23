# coding = utf-8
'''
Author: Albert Shawn
Email: weichangtsing@163.com

date = 2019/11/8 2:20 下午
dsec
'''
import json

import jieba.analyse
import numpy as np
import pymysql
from gensim import corpora, models, similarities


def word2tag_map(l):
    with open('word2tag.txt', 'r', encoding='utf8') as f:
        word2tag = json.loads(f.readline())
    trans = []
    for i in l:
        try:
            trans.append(str(word2tag[i]))
        except:
            pass
    return trans


if __name__ == '__main__':
    con = pymysql.connect(
        host='localhost',
        user='root',
        passwd='12345678',
        charset='utf8mb4',
    )
    cur = con.cursor()
    cur.execute('use lda')
    lda = models.ldamodel.LdaModel.load('patent_lda.model')
    tfidf = models.TfidfModel.load('patent_tfidf.model')
    print(tfidf)
    dictionary = corpora.Dictionary.load('patent_dictionary.dict')
    corpus = corpora.MmCorpus('patent_corpuse.mm')
    corpus_tfidf = tfidf[corpus]
    print(corpus)
    print(corpus_tfidf)

    test_doc = '单缸顶置分体式摩托车发动机汽缸头'
    index = similarities.MatrixSimilarity(corpus_tfidf)
    new_vecs = dictionary.doc2bow(word2tag_map(list(jieba.cut(test_doc))))
    print('test_vec', new_vecs)
    new_vec_tfidf_ls = tfidf[new_vecs]
    sims = index[new_vec_tfidf_ls]  # sim获得是这个index[new_vec_tfidf]得到的句子和文本集中所有的其他文本的相似度
    print(sims)
    rec_list1 = np.argsort(sims).tolist()[::-1][:10]
    print(rec_list1)
    print('当前文本描述：单缸顶置分体式摩托车发动机汽缸头')
    print('根据TFIDF分析，进行相似专利推荐：')
    print('找到以下可能有用的数据')
    for i in rec_list1:
        if sims[i] > 0.3:
            sql = "SELECT * FROM patent_info WHERE 专利ID = %s" % (i + 1)
            try:
                # 执行SQL语句
                cur.execute(sql)
                # 获取所有记录列表
                results = cur.fetchall()
                for row in results:
                    print('专利ID', row[0], '专利名', row[1], '专利号', row[2])
            except:
                pass

    test_doc = list(jieba.cut(test_doc))  # 新文档进行分词 采用精确切词模式
    test_doc = word2tag_map(test_doc)
    doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    a = sorted(doc_lda, key=lambda x: x[1], reverse=True)
    print('此文本分析得到主题降序排名：')
    for i in a:
        print('包含主题：', i[0], '其概率：', i[1])
    print('根据主题分析，进行相似专利推荐：')
    sql = "SELECT * FROM patent_info WHERE 主题ID1 = %s" % (a[0][0])
    try:
        # 执行SQL语句
        cur.execute(sql)
        # 获取所有记录列表
        results = cur.fetchall()
        print('找到以下也比较可能有用的数据')
        for row in results:
            print('专利ID', row[0], '专利名', row[1], '专利号', row[2])
    except:
        print('没有相关数据')
    sql2 = "SELECT * FROM patent_info WHERE 主题ID2 = %s" % (a[0][0])
    try:
        # 执行SQL语句
        cur.execute(sql2)
        # 获取所有记录列表
        results = cur.fetchall()
        print('找到以下或许有用的数据')
        for row in results:
            print('专利ID', row[0], '专利名', row[1], '专利号', row[2])
    except:
        print('没有相关数据')
