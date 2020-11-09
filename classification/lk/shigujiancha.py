import pandas as pd
import numpy as np
import os
import re
import jieba
from gensim.models import word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import json
import hashlib
import json
import os

from pyecharts import options as opts
from pyecharts.charts import Page, Tree, WordCloud

ls = ['缺陷设备','缺陷内容','设备类型','生产厂家','缺陷描述','处理结果','设备型号','检修建议','分类依据','完成情况']
ls1 = ['缺陷设备','缺陷内容','设备类型','生产厂家','缺陷描述','设备型号','分类依据']
ls2 = ['处理结果','设备型号','检修建议','分类依据','完成情况']

def filter_biaodian(cnt):
	pat = r'[！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]+|[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^\_\`\{\|\}\~0-9]+'#去标点
	return re.sub(pat, '', cnt)
def filter_kong(x):
	if ' ' in x:
		return False
	if 'nan'==x:
		return False
	if '\n'==x:
		return False
	return True

def title_handle(title):
	# 去除标点符号
	title = filter_biaodian(title)
	title = ' '.join(list(filter(filter_kong, jieba.lcut(title)))) + ' '
	return title

def line2words(l):
	rr = []
	for w in title_handle(l).split(' '):
		if w in stopws:
			continue
		if w in vocab and not w in rr:
			rr.append(w)
	return rr

stop_words_path = 'stop_words_ch.txt'
def stop_words():
	#获取stopwords，返回列表
	stop_words_file = open(stop_words_path, 'r', encoding='gbk')
	stopwords_list = []
	for line in stop_words_file.readlines():
		stopwords_list.append(line[:-1])
		#print(type(stopwords_list[-1]))
	return stopwords_list

yuliao_path = 'out.txt'
model_path = 'dianli_model.bin'


def load_word2vec_model():
	'''
	加载训练好的模型
	'''
	print('加载模型文件...')
	return word2vec.Word2Vec.load(model_path) 

stopws = stop_words()
stopws.extend(['号','相','体','组'])
model = load_word2vec_model()
vocab = list(model.wv.vocab.keys())
print ('ml:',len(vocab))

def savejson(r='沭城变沭汪7H23开关靠近刀闸侧压排发热455摄氏度'):
	md5 = hashlib.md5()
	md5.update(r.encode('utf-8'))
	jsonname = md5.hexdigest()
	jsonname = os.path.join('res',str(jsonname)+'.json')

	if os.path.exists(jsonname):
		return json.load(open(jsonname,'r'))

	wr = line2words(r)
	#print (wr)
	# gendfwords()
	df_file2 = 'outwall.csv'
	df2 = pd.read_csv(df_file2)
	ks = ['缺陷设备','缺陷内容','设备类型','生产厂家','缺陷描述','处理结果','设备型号','检修建议','分类依据','完成情况']
	#print (df2[ks].iloc[10].values)
	res = np.zeros([len(df2)])
	for index,row in df2.iterrows():
		ww = eval(row['words'])
		# print (ww)
		ss = model.n_similarity(wr,ww)
		# print (index,ss)
		res[index]=ss
		# break
	# print (res)
	rdx = res.argsort()[-20:][::-1]
	#print (rdx)
	resjson = {}
	ls = []
	
	words = {}
	sugg = {}
	for i in rdx:
		#print (i,res[i]) 
		#print (df2['words'][i])
		#print (df2['检修建议'][i])
		ls.append([res[i],df2['words'][i],list(df2[ks].iloc[i].values)])
		for w in eval(df2['words'][i]):
			print ('kw:',w)
			if w in words:
				words[w]+=1
			else:
				words[w]=1
		w = df2['检修建议'][i]
		if not w!=w:
			if w in sugg:
				sugg[w] +=1
			else:
				sugg[w] =1


		# print (df2['分类依据'][i])
	resjson['name'] = r
	resjson['list'] = ls
	resjson['words'] = words
	resjson['sugg'] = sugg

	print (sugg)
	
	fp = open(jsonname,'w')
	json.dump(resjson,fp)

	return resjson

def gettreedata(df):
	res = {}
	res['name'] = '故障分类'
	res['children'] = []
	for k,v in df.groupby('ZYLB'):
		#print (k,len(v))
		cc1 = {}
		cc1['name'] = k
		cc1['children'] = []
		for kk,vv in v.groupby('SBLX'):
			ccc1 = {}
			ccc1['name'] = kk
			print ('  ',kk,len(vv))
			for kkk,vvv in vv.groupby('SBMC'):
				print ('     ',kkk,len(vvv))
			cc1['children'].append(ccc1)
		res['children'].append(cc1)
	#print (res)
	c = (
		Tree()
		.add("", [res])
		.set_global_opts(title_opts=opts.TitleOpts(title="故障原因分类"))
	)
	# c.render()
	return c.dump_options_with_quotes()
	# return c

def getwords(ws):
	words = []
	for k in ws:
		words.append((k,ws[k]))
	c = (
		WordCloud()
		.add("", words, word_size_range=[20, 100])
		.set_global_opts(title_opts=opts.TitleOpts(title="缺陷故障关键词"))
	)
	# c.render('wd.html')
	return c.dump_options_with_quotes()

def getsimdf(r):
	wr = line2words(r)
	#print (wr)
	# gendfwords()
	df_file2 = 'newdata.csv'
	df2 = pd.read_csv(df_file2,encoding = "GBK")
	ks = ['ZYLB','SBLX','SBMC','QXNR','QXLB','QXDJ','SSJGHDY']
	ks2 = ['sm','ZYLB','SBLX','SBMC','QXNR','QXLB','QXDJ','SSJGHDY']
	#print (df2[ks].iloc[10].values)
	res = np.zeros([len(df2)])
	for index,row in df2.iterrows():
		# print (row['QXNR'])
		if pd.isna(row['QXNR']):
			continue
		ww = line2words(row['QXNR'])
		# print (ww)
		if len(ww)==0:
			continue
		# ww = eval(row['words'])
		# print (ww)
		ss = model.n_similarity(wr,ww)
		# print (index,ss)
		res[index]=ss
		# break
	# print (res)
	rdx = res.argsort()[-20:][::-1]
	# print (type(rdx),rdx)
	ls = []
	words = {}
	for i in rdx.tolist():
		#print (i,res[i]) 
		# print (df2['QXNR'][i])
		r = ["%.2f" % res[i]]
		r.extend(list(df2[ks].iloc[i].values))
		ls.append(r)
		for w in line2words(df2['QXNR'][i]):
			#print ('kw:',w)
			if w in words:
				words[w]+=1
			else:
				words[w]=1

	df = pd.DataFrame(ls)
	df.columns = ks2
	#print (df)
	#print (words)
	res = {}
	
	
	res['tree'] = gettreedata(df)
	res['words'] = getwords(words)
	res['table'] = json.dumps(ls)

	return res



def readnewdata():
	df_file2 = 'newdata.csv'
	df2 = pd.read_csv(df_file2,encoding = "GBK")
	#print (len(df2))
	#print (df2.columns)
	cs = ['ZYLB','SBLX','SBMC','QXNR','QXLB','QXDJ','SSJGHDY']
	df = df2[cs]
	# print (df)
	gettreedata(df)
	# dt = [{}]
	# for k,v in df.groupby('ZYLB'):
	# 	print (k,len(v))
	# 	for kk,vv in v.groupby('SBLX'):
	# 		print ('  ',kk,len(vv))
	# 		for kkk,vvv in vv.groupby('SBMC'):
	# 			print ('     ',kkk,len(vvv))

def main():
	r ='充电屏直流监视装置黑屏'
	# res = savejson(r)
	print (r)
	# readnewdata()
	getsimdf(r)

if __name__ == '__main__':
	main()