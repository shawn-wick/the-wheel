# coding = utf-8
'''
Author: Albert Shawn
Email: weichangtsing@163.com

date = 2019/11/8 2:20 下午
dsec
'''
import pymysql

con = pymysql.connect(
    host='localhost',
    user='root',
    passwd='12345678',
    charset='utf8mb4',
)
cur = con.cursor()
cur.execute('drop database if exists lda')
cur.execute('create database lda character set utf8')
cur.execute('use lda')  # 使用weichai数据库
cur.execute('drop table if exists patent_info')  # 在有名为front_param_resualt的表的情况之下删除之
sql = 'create table patent_info(专利ID int auto_increment primary key, 专利名 VARCHAR(100) , 专利号 VARCHAR(32),专利链接 VARCHAR(100),专利摘要 VARCHAR(2000),关键词 VARCHAR(500),主题ID1 int, 主题ID2 int, 主题ID3 int, 主题ID4 int, 主题向量 VARCHAR(1000)) character set utf8mb4'
cur.execute(sql)  # 执行创建表的sql语句，进行表的创建
cur.execute('drop table if exists topic_info')  # 在有名为front_param_resualt的表的情况之下删除之
sql = 'create table topic_info(主题ID int ,主题词ID1 int, 主题词ID2 int, 主题词ID3 int, 主题词ID4 int) character set utf8mb4'
cur.execute(sql)  # 执行创建表的sql语句，进行表的创建
cur.execute('drop table if exists topicword_info')  # 在有名为front_param_resualt的表的情况之下删除之
sql = 'create table topicword_info(主题词ID int ,中心词 VARCHAR(10), 主题词 VARCHAR(5000)) character set utf8mb4'
cur.execute(sql)  # 执行创建表的sql语句，进行表的创建
#
#
# sql_insert = 'insert into front_params(参数名,别名)value("获取时间","获取时间")'
# cur.execute(sql_insert)
# con.commit()
# '''此程序按照竞品模版表的格式创建竞品爬取结果的表，可以在数据库中去进行手动添加'''
# # list = ['柴油机型号', '生产厂家', '用途', '结构形式', '气缸数目', '缸径', '行程', '行程/缸径', '排量', '标定功率', '备用功率', '系列最大标定功率', '标定转速', '最大扭矩',
# #         '最大扭矩初始转速', '最大扭矩终止转速', '大功率机型最大扭矩', '低速扭矩', '压缩比', '排放控制技术路线', '排放水平', '宣传重量', '宣传长度', '宣传宽度', '宣传高度', 'B10寿命',
# #         '大修期', 'DPF清灰周期', '气门调整周期', '机油更换周期', '机油容量_高', '机油容量_低', '最大制动功率', '高原能力', '冷启动', '扭矩储备率', '超级扭矩']
# #
# # list1 = ['engine_id', 'brand', 'factory', 'series', 'engine_type', 'rated_power', 'maximum_power_output', 'frequency',
# #          'fuel', 'smoke',
# #          'Minimum_fuel_consumption', 'compression_ratio', 'Engine_Dimension', 'weight', 'output_standard',
# #          'displacement', 'nosie', 'rated_speed', 'Bore_stroke', 'Max_Hp', 'continuous_power', 'Valve_train',
# #          'cylinder_num', 'commen_power', 'cylinder_type', 'technical_route', 'adaptation_scope',
# #          'Admission_mehtod', 'maximum_torque', 'rated_power_speed_rate', 'max_power_speed_rate',
# #          'maximum_torque_speed']
# # dic = {'engine_id': '柴油机型号', 'brand': '生产厂家', 'factory': '生产厂家', 'series': '系列', 'engine_type': '结构形式',
# #        'rated_power': '标定功率', 'maximum_power_output': '系列最大标定功率', 'frequency': '频率', 'fuel': '燃料',
# #        'smoke': '烟度', 'Minimum_fuel_consumption': '最低燃油耗率', 'compression_ratio': '压缩比',
# #        'Engine_Dimension': '宣传长度 宣传宽度 宣传高度', 'weight': '宣传重量', 'output_standard': '排放水平', 'displacement': '排量',
# #        'nosie': '噪声', 'rated_speed': '标定转速', 'Bore_stroke': '缸径*行程', 'Max_Hp': '最大马力', 'continuous_power': '持续功率',
# #        'Valve_train': '单缸气门数', 'cylinder_num': '气缸数目', 'commen_power': '标定功率', 'cylinder_type': '气缸形式',
# #        'technical_route': '排放控制技术路线', 'adaptation_scope': '用途', 'Admission_mehtod': '进气方式', 'maximum_torque': '最大扭矩',
# #        'rated_power_speed_rate': '额定功率/转速', 'max_power_speed_rate': '超负荷功率/转速',
# #        'maximum_torque_speed': '最大扭矩初始转速'}
# # a = {dic[i]: j for i, j in dic.items()}
#
# cur.execute('drop table if exists front_param_result')#如果需要添加新的参数，需要先删除旧表
# sql = 'create table if not exists front_param_result(主键ID int auto_increment primary key, 机型型号 VARCHAR(100) ,制造厂商 VARCHAR(100),创建人 VARCHAR(100),创建时间 VARCHAR(100),用途 VARCHAR(200),数据来源 VARCHAR(500),系列 VARCHAR(32),结构形式 VARCHAR(50),' \
#       '单缸气门数 int(2),进气方式 VARCHAR(32),气缸数目 int(2),气缸形式 VARCHAR(32),缸径 float(8,4),行程 int(6), `行程/缸径` float(8,4),排量 float(6,4),备用功率 float(8,4), 标定功率 float(8,4),系列最大标定功率 float(8,4),标定转速 float(6),最大扭矩 float(9,4),' \
#       '最大扭矩转速 float(9,4),最大扭矩起始转速 float(9,4),最大扭矩终止转速 float(9,4),大功率机型最大扭矩 float(9,4),低速扭矩 float(9,4),压缩比例 float(8,4),排放控制路线 VARCHAR(50),排放水平 VARCHAR(40),宣传重量 float(10,4), ' \
#       '宣传长度 float(9,4),宣传宽度 float(9,4),宣传高度 float(9,4),B10寿命 float(7,4), `大修期(h)` float(6),`大修期(km)` float(7,4), `DPF清灰周期(h)` float(6), `DPF清灰周期(km)` float(10,4),' \
#       '`气门调整周期(h)` float(6), `气门调整周期(km)` float(9,4), `机油更换周期(h)` float(6), `机油更换周期(km)` float(10,4),`机油容量(高)` float(7,4),`机油容量(低)` float(7,4),高原能力 VARCHAR(10),`冷启动(辅助加热)` VARCHAR(10),扭矩储备率 float(8,4),超级扭矩 float(9,4),' \
#       '低速扭矩转速 float(6), `制动功率(标定点)` float(8,4), `制动功率(标定点)转速` float(6), `制动功率(低速)` float(8,4), `制动功率(标定点)低速` float(6), `P1功率(船机)` float(8,4),最大输出功率 float(8,4),频率 VARCHAR(32),燃料 VARCHAR(32),烟度 VARCHAR(32),最低油耗 float(6),噪声 VARCHAR(32),最大马力 float(6),持续功率 VARCHAR(32),' \
#       '技术路线 VARCHAR(32),`额定功率/转速` VARCHAR(32),`超负荷功率/转速` VARCHAR(32),最大制动功率 VARCHAR(32) ) character set utf8mb4'
# cur.execute(sql)
# '''此表是进行数据源表的定义，表主要包括数据源名称和数据源网址两个部分，网址按照提供竞品网站预设了三十个，可以在mysql的表中进行手动的后续添加
# 这个网站是开始竞品爬取的起始网站，爬虫以此网站为基础，以三层网站为爬取深度，获取和分析每一个页面中的表格是否包含发动机的信息'''
# cur.execute('drop table if exists urls_manage')  # 在有名为front_param_resualt的表的情况之下删除之
# sql = 'create table urls_manage(编号 int auto_increment primary key, 数据源名称 VARCHAR(32) , 数据源网址 VARCHAR(2000)) character set utf8mb4'
# cur.execute(sql)  # 执行创建表的sql语句，进行表的创建
# urls = ['https://www.weichai.com/cpyfw/wmdyw/dlzc/kcyfdj/',
#         'https://product.360che.com/price/c3_s61_b0_s0.html' ]
# with open('changjia','r',encoding='utf8') as f:
#     for i in f.readlines():
#         urls.append(i.strip('\n'))
# for i in urls:
#     sql_insert_1 = 'insert into urls_manage(数据源网址)'
#     sql_insert_2 = ' values("%s")' % (i)
#     print(sql_insert_1 + sql_insert_2)
#     cur.execute(sql_insert_1 + sql_insert_2)
#     con.commit()
