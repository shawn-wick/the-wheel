# coding = utf-8
'''
Author: Albert Shawn
Email: weichangtsing@163.com

date = 2019/11/8 2:20 下午
dsec
'''
import pymysql
import jieba
print('请输入您想查询的关键词:',end='\n')
a = input()
keyword = jieba.cut_for_search(a)
print(keyword)
con = pymysql.connect(
    host='localhost',
    user='root',
    passwd='12345678',
    charset='utf8mb4',
)
cur = con.cursor()
cur.execute('use lda')
sql = "SELECT * FROM patent_info WHERE %s in 专利名" %(a)
try:
    # 执行SQL语句
    cur.execute(sql)
    # 获取所有记录列表
    results = cur.fetchall()
    for row in results:
        print('专利ID', row[0], '专利名', row[1], '专利号', row[2])
except Exception as e:
    print(e)

