from os import replace
from os.path import join
import string
from gensim.utils import tokenize
from keras_preprocessing.text import Tokenizer
from numpy.core.defchararray import mod
from numpy.core.fromnumeric import size
import pandas as pd 
import numpy as np
import re
import jieba
import jieba.posseg as pseg 
from keras import models
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from sklearn import metrics
from keras.models import load_model
import matplotlib.pyplot as plt
train_data=pd.read_csv(r'E:/work_space/NLP/lx/liangku/data/train.csv')
test_data=pd.read_csv(r'E:/work_space/NLP/lx/liangku/data/test.csv')

def encodeLabel(data):
    labelList=[]
    for label in data['label']:
        labelList.append(label)
    # print(labelList)
    le=LabelEncoder()
    resultLabel=le.fit_transform(labelList)
    return resultLabel
trainlabel=encodeLabel(train_data)
testlabel=encodeLabel(test_data)
# print(trainlabel)
# print(testlabel)

def getContent(data):
    contentList=[]
    for content in data['content']:
        contentList.append(content)
    return contentList
trainContent=getContent(train_data)
testContent=getContent(test_data)
# print(trainContent)
# print(testContent)

#分词
def stopwordslist():#加载停用词
    stopwords=[line.strip() for line in open(r'E:/work_space/NLP/lx/liangku/stop/stopword.txt',encoding='utf-8').readlines()]
    return stopwords

def deleteStop(sentence):#去停用词
    stopwords=stopwordslist()
    outstr=''
    for i in sentence:
        if i not in stopwords and i!="\n" and i!=' ':
            outstr+=i
    return outstr

def wordCut(Content):
    Mat=[]
    for con in Content:
        seten=[]
        con=re.sub('[%s]' % re.escape(string.punctuation),'',con)
        fenci=jieba.lcut(con)#精准模式分词
        stc=deleteStop(fenci)
        seg_list=pseg.cut(stc)
        for word,flag in seg_list:#词性标注
            if flag not in['nr','ns','nt','nz','m','f','ul','l','r','t']:
                seten.append(word)
        Mat.append(seten)
    return Mat
trainCut=wordCut(trainContent)
testCut=wordCut(testContent)
wordCut=trainCut+testCut
print(wordCut)

fileDic=open('wordCut.txt','w',encoding='UTF-8')
for i in wordCut:
    fileDic.write(" ".join(i))
    fileDic.write('\n')
fileDic.close()
words=[line.strip().split(" ")for line in open('wordCut.txt',encoding='UTF-8').readlines()]

maxlen=100
#word2vec训练
#设置词向量维度
num_features=100
#最低频度
min_word_count=3
#并行化
num_workers=4
#上下文窗口
context=4
model=word2vec.Word2Vec(wordCut,workers=num_workers,size=num_features,min_count=min_word_count,window=context)
#强制归一化
model.init_sims(replace=True)
model.save(r'E:/work_space/NLP/lx/liangku/model/w2vModel')
model.wv.save_word2vec_format('VectorFenci',binary=False)
print('model'     ,model)

#加载模型
w2v_model=word2vec.Word2Vec.load(r'E:/work_space/NLP/lx/liangku/model/w2vModel')

#fit_on_texts函数可以将输入的文本中的每个词根据词频，词频越大，编号越小
tokenizer=Tokenizer()
tokenizer.fit_on_texts(wordCut)
vocab=tokenizer.word_index
# print(vocab)


#特征数字编号，不足在前面补0
trainID=tokenizer.texts_to_sequences(trainCut)
# print(trainID)
testID=tokenizer.texts_to_sequences(testCut)
trainSeq=pad_sequences(trainID,maxlen=100)
# print(trainSeq)
testSeq=pad_sequences(testID,maxlen=100)

#标签的独热编码
trainCate=to_categorical(trainlabel,num_classes=6)
# print(trainCate)
testCate=to_categorical(testlabel,num_classes=6)

embedding_matrix=np.zeros((len(vocab)+1,100))
for word,i in vocab.items():
    try:
        embedding_vector=w2v_model[str(word)]
        embedding_matrix[i]=embedding_vector
    except KeyError:
        continue

main_input=Input(shape=(maxlen,),dtype='float64')
embedder=Embedding(len(vocab)+1,100,input_length=maxlen,weights=[embedding_matrix],trainable=False)
model=Sequential()
model.add(embedder)
model.add(Conv1D(256,3,padding='same',activation='relu'))
model.add(MaxPool1D(maxlen-5,3,padding='same'))
model.add(Conv1D(32,3,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(trainSeq,trainCate,batch_size=256,epochs=8
,validation_split=0.2)
model.save('TextFenci')


#预测与评估
mainModel=load_model('TextFenci')
result=mainModel.predict(testSeq)
print(result)
print(np.argmax(result,axis=1))
score=mainModel.evaluate(testSeq,testCate,batch_size=32)
print(score)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()