# -*- coding=utf-8 -*-
'''
Created on 2014年6月23日
基础函数集合
@author: Mafing
'''
import dataPre
import os
import pickle
import random
import math
import numpy as np
import jieba
from gensim import corpora, models, similarities, matutils


def stopWords():#获取停用词
    stop_words = set()
    '''
    add more stopwords
    '''
    f = open("../Data/stopWords.txt")
    for l in f:
        l = l.strip()
        stop_words.add(l)
    return stop_words 

def getTrainFileList(path = "C:/PythonWork/BaiduEntitySimilarity/Data/Type/"):
    #获取所有待测量数据集
    fileList =[path+val for val in os.listdir(path) if "train-" in val]
    return fileList
def getFeatureFileList(featureType= "tfIdf",path = "C:/PythonWork/BaiduEntitySimilarity/Data/Type/"):
    #获取所有特征数据集
    fileList =[path+val for val in os.listdir(path) if (featureType) in val]
    return fileList



def getAttr(entityDic):
    '''获取某一类实体的所有属性
    随机挑选50个实例，将他们的attr合并去重
    '''
    InfoDic = dataPre.readInfo()
    setAttr = set()
    tmpList = entityDic.items()
    for i in range(0,50):
        sampleInt = random.randint(0,len(entityDic)-1)#随机一个数
        tmpKey = tmpList[sampleInt][0]
        tmpDic = InfoDic[tmpKey]
        attrList = [val for val in tmpDic]
        for val in attrList:
            setAttr.add(val)
    setAttr.remove("type")
    setAttr.remove("sid")
    #print  "该组的属性值为：\t"+"\t".join(list(setAttr))
    return setAttr


def getRatio(listB):#给定list，根据其内已有的比例，返回其4、3、2、1各占的比例
    tmpList = sorted(listB,reverse = True)#由大到小排列
    first  = tmpList.index(3)
    second = tmpList.index(2)
    third  = tmpList.index(1)
    return [first,second,third]
def getQuarter(listB):#根据四分之一来分配
    number    = len(listB)
    firstQtr  = int(number/4)             #取前四分之一的节点quarter
    secondQtr = int(number/2)
    thirdQtr  = int(number/4)*3  
    return [firstQtr,secondQtr,thirdQtr]  

def transToQuarter(listA,listB,listRat=None):
    #将第一个参数所得的相似度按照四分之一的规则正规化,第二个参数是判断4,3,2,1的index
    if listRat == None:
        listRat = getRatio(listB)#默认采用比例方式
    if len(listRat)!=3:
        print "输入的分割List不正确"
        return None
    tmpList = sorted(listA,reverse = True)#tmpList由大到小排列,不改变listA的顺序
    listAPlus = []
    firstQtr  = tmpList[listRat[0]]              #根据输入的阈值进行分配
    secondQtr = tmpList[listRat[1]]
    thirdQtr  = tmpList[listRat[2]]
    for val in listA:
        if val > firstQtr:
            listAPlus.append(4)
        elif val > secondQtr:
            listAPlus.append(3)
        elif val > thirdQtr:
            listAPlus.append(2)
        else:
            listAPlus.append(1)
    return listAPlus 

def logRegtransToQuarter(listA,listB,listRat=None):
    #直接按照给出的结果进行预测(1.5->1,1.5~2.5->2,……)
    if listRat == None:
        listRat = getRatio(listB)#默认采用比例方式
    if len(listRat)!=3:
        print "输入的分割List不正确"
        return None
    listAPlus = []
    firstQtr  = 3.5              #根据输入的阈值进行分配
    secondQtr = 2.5
    thirdQtr  = 1.5
    for val in listA:
        if val > firstQtr:
            listAPlus.append(4)
        elif val > secondQtr:
            listAPlus.append(3)
        elif val > thirdQtr:
            listAPlus.append(2)
        else:
            listAPlus.append(1)
    return listAPlus 

def getListIndexValBack(listValue,listIndex):
    tmpList = []
    for indexI in listIndex:
        tmpList.append(listValue[indexI])
    return tmpList
        
def featureInit(tag = 1):
    '''分成4种，只有name,des;之后添加其他标称型;之后添加其他标称型的count;之后添加LDA'''
    featureAttr = set() 
    featureAttr.add("name")
    featureAttr.add("description")
    if tag == 1:
        return featureAttr
    featureAttr.add("inLanguage")
    featureAttr.add("datePublish")
    featureAttr.add("country")
    featureAttr.add("actor")
    featureAttr.add("director")
    featureAttr.add("editor")
    if tag == 2:
        return featureAttr
    featureAttr.add("CountDirector")
    featureAttr.add("CountLanguange")
    featureAttr.add("CountCountry")
    featureAttr.add("CountEditor")
    featureAttr.add("CountActor")
    if tag == 3:
        return featureAttr
    featureAttr.add("NameLDACosSim")
    featureAttr.add("NameLDAHellingerSim")
    featureAttr.add("DesLDACosSim")
    featureAttr.add("DesLDAHellingerSim")
    return featureAttr  
   
def cutDocuments(infoDic):
    '''
    将全部实体的name和description进行分词，返回entity的序号
    '''
    if os.path.isfile('../Data/cut.txt'):
        return
    else:
        stop_words = stopWords()
        corpus = []
        for val in infoDic:
    #         if infoDic[val]["type"] in "Movie":
            name = infoDic[val]["name"]
            description = infoDic[val].get("description","")
            corpus.append(name+"\t"+description)
        for i in range(0,len(corpus)):
            l  = corpus[i]
            w = jieba.cut(l)
            w = [val for val in w if val not in stop_words]
            tmp = " ".join(w)
            corpus[i] = tmp
        fOut = open("../Data/cut.txt","w+")
        fOut.write("\n".join(corpus))#写入分词后的结果
#     fOutEntityDic = open("../Data/entityIndex.txt","w+")
#     fOut.write("\n".join(entityDic))#写入分词后的结果

def generateDictAndCorpus():
    '''
    产生全部实体的name和description构成的字典和语料
    '''
    if os.path.isfile('../Data/model/deerwester.dict'):
        dictionary = corpora.Dictionary.load('../Data/model/deerwester.dict')
        return dictionary
    else:
        documents= []
        f = open("../Data/cut.txt")
        for l in f:
            documents.append(l.replace("\r\n",""))
        # remove cosmmon words and tokenize移除stopwords
        stop_words = set()
        '''
        add more stopwords
        '''
        f = open("../Data/stopWords.txt")
        stop_words = {}
        for l in f:
            l = l.strip()
            stop_words[l] = 1
    
        tokenNum = {}
        for document in documents:
            tokenList = document.split()
            for word in tokenList:
                if word in stop_words:
                    tokenNum[word] = 0
                elif word not in tokenNum:
                    tokenNum[word] = 1
                else:
                    tokenNum[word] = tokenNum[word]+1        
    
        tokens_onceDic = {}
        for val in tokenNum:
            if tokenNum[val] <1:
                tokens_onceDic[val] = 0
        texts = [[word for word in document.lower().split() if word not in tokens_onceDic] for document in documents]#对于每个doc，其中的每个word  验证其是否在stoplist,且不在onceword里面
    
        dictionary = corpora.Dictionary(texts)#将词向量转为词袋字典<key-id>
        dictionary.save('../Data/model/deerwester.dict') # store the dictionary, for future reference
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize('../Data/model/deerwester.mm', corpus) # store to disk, for later use
        return dictionary
    
def generateLDA(num_topics = 50):
    '''
    输入：LDA模型topics的个数
    输出：根据全体字典和语料生成的LDA模型
    '''
    if os.path.isfile('../Data/model/model.lda'):
        lda = models.ldamodel.LdaModel.load('../Data/model/model.lda')
    else:
        id2word = corpora.Dictionary.load('../Data/model/deerwester.dict')
        corpus = corpora.MmCorpus('../Data/model/deerwester.mm')
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
        lda.save('../Data/model/model.lda')
    return lda

def generateLDAVector(lda,dictionary,strTmp):
    '''
    输入：lda是产生的LDA模型；dictionary是全部实体的name和description构成的字典；InfoDic存放的是所有实体的属性值对；attr="name",或者是"descroption"
    输出：产生实体为sid的attr的lda向量
    '''
    doc = strTmp
    doc = dictionary.doc2bow(doc)
    vec = lda[doc]
    return vec
        

def generateTFIDFModel():
    '''生成tfIDF模型    '''
    if os.path.isfile('../Data/model/model.tfidf'):
        tfidf = models.TfidfModel.load('../Data/model/model.tfidf')
    else:
        corpus = corpora.MmCorpus('../Data/model/deerwester.mm')
        tfidf = models.TfidfModel(corpus)
        tfidf.save('../Data/model/model.tfidf')
    return tfidf


def generateSingleTFIDF(tfidf,dictionary,strTmp):
    '''给定一个str和对应的tfidf模型，返回该str对应'''
    doc = strTmp
    doc = dictionary.doc2bow(doc)
    corpus_tfidf = tfidf[doc]
    return corpus_tfidf

def mergeResult(methodList,outFilePath):
    fout = open(outFilePath,"w+")
    type = ["Movie","MusicRecording","ShowSeries","SoftwareApplication","TVSeries","VideoGame"]
    for i in range(0,len(methodList)):
        fin = open("../Data/test/test-" + type[i] + ".txt")
        testTagArray = getTagList("../Data/result/result-test-" + type[i] + "-" + str(methodList[i]) + ".txt")
        k = 0
        for l in fin:##将每一行存放成六列数据的形式
            l = l.strip()
            tmpList = l.split("\t")
            key1 = tmpList[0].strip()
            key2 = tmpList[1].strip()
            value = testTagArray[k]
            fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[k]) + "\n")
            k += 1
        fin.close()
    return
        
def getTagList(filePath):
    testTagArray = []
    i = 0
    fin = open(filePath)
    for l in fin:
        l = l.strip()
        testTagArray[i] = int(l)
    return testTagArray


if __name__ == "__main__":
    entityDic = dataPre.readInfo()
#     typeDic = {}
#     for val in entityDic:
#         type = entityDic[val]["type"]
#         if type not in typeDic:
#             typeDic[type] = 0
#         else:
#             typeDic[type] = typeDic[type] + 1
#     SumPair = 0
#     for val in typeDic:
#         print val+"\t"+str(typeDic[val])
#         SumPair += typeDic[val]*typeDic[val]
#     print str(SumPair)
#     print str(math.sqrt(SumPair))
#     generateDictAndCorpus()
#     dictionary = corpora.Dictionary.load('../Data/model/deerwester.dict')
#     sid = "c266fad99b969ac73c605cf094381099"
#     lda = generateLDA()
#     v = generateLDAVector(lda,dictionary,entityDic,sid,"description")
#     print v
    print entityDic["ef8e7b75930725e05f95b0e8897b5e86"]["name"]
    print 1
                
        
        