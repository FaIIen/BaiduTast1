# -*- coding=utf-8 -*-
'''
Created on 2014年6月24日

@author: Mafing
'''
import simMeasure
import dataPre
import funUnit
import jieba
import datetime
import time 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities,matutils
import numpy as np





''' 将读入的训练集/测试集  进行特征生成，并输出成文件
@param inFile: 训练集/测试集路径
@param InfoDic: readInfo的全集
@param attrSet: 生成特征的列名（属性名字）
@return: 输出特征向量文件
'''
def featureGenerate(inFile,outFile,InfoDic,attrDic):
    '''
            生成特征文件
            首先 将entity分类，对于每个类别，其中的每个属性，求其相似度
            然后 从分类后的train集中，对于每一条测试数据，计算它们在各个属性维度上tf-idf向量的余弦相似度
            按照 id1,id2,name1,name2,att1-Sim,att2-Sim,……,Tag 这样的形式输出
    '''
    #读取训练集/测试集
    file = open(inFile)  

    #对读入的训练集/测试集进行排序                                                   
    entityDic = sortDic(file)    
 
    #按读入顺序排序,由小到大                                        
    sortEntity = sorted(entityDic.items(),key=lambda e:e[1]) 

    #根据特证名，将具体内容导入               
    featureAttr = attrExtract(attrDic,InfoDic,sortEntity) 
      
    #start 初始化lda模型
    lda = models.ldamodel.LdaModel.load('../Data/model/model.lda')
    #end 初始化lda模型
    
    #将相似度特征写入文件
    fOut = open(outFile,"w+")
    fOut.write("key1\tkey2\tname1\tname2\t"+"\t".join(list(attrDic.keys()))+"\tTag"+"\n")
    file = open(inFile)  
    for l in file:
        l = l.replace("\r\n","").replace("\n","")
        tmpList = l.split("\t") 
        key1   = tmpList[0]
        key2   = tmpList[1]   
        tag    = tmpList[2].replace("\n","") 
        simList = []
        for key,value in attrDic.iteritems():
            if "TFIDF" in key:                           #Name,Description 的TFIDF比较
                f1 = InfoDic[key1][key]
                f2 = InfoDic[key2][key]
                if len(f1)==0 or len(f2)==0:
                    tmpSim=0
                else:
                    tmpSim = cosine_similarity(f1, f2)[0,0] 
            elif "Int" in key:                           #数值型比较
                f1 = featureAttr[key].get(key1)[0]       
                f2 = featureAttr[key].get(key2)[0]
                tmpSim = simMeasure.datePublishSim(f1,f2)
            elif "Count" in key:                         #List型求交
                f1 = featureAttr[key].get(key1)
                f2 = featureAttr[key].get(key2)
                tmpSim = simMeasure.mixSim(f1, f2)
            elif "LDA" in key:                           #LDA比较
                vec_lda1 = InfoDic[key1]["LDA"]
                vec_lda2 = InfoDic[key2]["LDA"]
                if "Cos" in key:
                    tmpSim = matutils.cossim(vec_lda1, vec_lda2)
                elif "HellingerSim" in key:
                    dense1 = matutils.sparse2full(vec_lda1, lda.num_topics)
                    dense2 = matutils.sparse2full(vec_lda2, lda.num_topics)
                    tmpSim = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
                else:    
                    dense1 = matutils.sparse2full(vec_lda1, lda.num_topics)
                    dense2 = matutils.sparse2full(vec_lda2, lda.num_topics)
                    tmpSim = simMeasure.kl(dense1, dense2)
                    if tmpSim == float("inf"):
                        tmpSim = 0
            else:                                   #其余使用jaccardSim
                f1 = featureAttr[key].get(key1)
                f2 = featureAttr[key].get(key2)
                tmpSim = simMeasure.jaccardSim(f1, f2)
            simList.append(str(tmpSim))
        try:
            fOut.write(key1+"\t"+key2+"\t"+InfoDic[key1].get("name")+"\t"+InfoDic[key2].get("name")+"\t"+"\t".join(simList)+"\t"+tag+"\n")
        except:
            print l

'''从文件读入训练集/测试集     输出实体对
@param filePath:训练集/测试集的路径
@return: 实体对   [id:读入序号] 
'''
def sortDic(file):
    entityDic = {}#存放所有entity的序号
    entityId = 0
    
    for l in file:
        l = l.replace("\r\n","").replace("\n","")
        tmpList = l.split("\t")
        if tmpList[0] not in entityDic:#将实体排序
            entityDic[tmpList[0]] = entityId
            entityId += 1
        if tmpList[1] not in entityDic:
            entityDic[tmpList[1]] = entityId
            entityId += 1  
    
    return entityDic

'''根据给出的特证名字，将entity全集中的数据分类
@param attrDic:主函数给出的特证名     [特征名：提取的属性名]  e.g:[CountCountry:Country]
@param InfoDic:readInfo的entity全集
@param sortEntity:排序后的实体对 [id:读入序号]   
'''
def attrExtract(attrDic,InfoDic,sortEntity):
    featureAttr = {}#存放各个属性对应的特征向量
    for key,value in attrDic.items():
        if "TFIDF" in key:
            pass
        elif "LDA" in key:
            pass
        else:
            featureAttr[key] = attrListExtract(InfoDic,sortEntity,value)
    return featureAttr


def attrListExtract(InfoDic,sortEntity,attr):
    dicAttr = {}#存放返回的属性列表
    for entityVal in sortEntity:
        sid = entityVal[0]
        try:
            dicAttr[sid] = InfoDic[sid].get(attr,"none").split(",")
        except:
            print attr
            print InfoDic[sid].get(attr,"none")
    return dicAttr

def movieAttrDic():
    attrDic={}
    attrDic["Country"]="Country";
    attrDic["inLanguage"]="inLanguage";
    attrDic["editor"]="editor";
    attrDic["actor"]="actor";
    attrDic["director"]="director";
    attrDic["datePublishInt"]="datePublish";
    attrDic["CountCountry"]="Country";
    attrDic["CountActor"]="actor";
    attrDic["CountDirector"]="director";
    attrDic["CountEditor"]="editor";
    attrDic["CountLanguange"]="inLanguage";
    attrDic["LDACosSim"]="LDA";
    attrDic["LDAHellingerSim"]="LDA";
    attrDic["LDAKLSim"]="LDA";
    attrDic["descriptionTFIDF"]="descriptionTFIDF";
    attrDic["nameTFIDF"]="nameTFIDF";
    return attrDic

def musicAttrDic():
    attrDic={}
    attrDic["durationInt"]="duration";
    attrDic["CountinAlbum"]="inAlbum";
    attrDic["datePublishedInt"]="datePublished";
    attrDic["byArtist"]="byArtist";
    attrDic["CountbyArtist"]="byArtist";
    attrDic["nameTFIDF"]="nameTFIDF";
    attrDic["LDACosSim"]="LDA";
    attrDic["LDAHellingerSim"]="LDA";
    attrDic["LDAKLSim"]="LDA";
    return attrDic

def showseriesAttrDic(): 
    attrDic={}
    attrDic["genre"]="genre";
    attrDic["Countgenre"]="genre";
    attrDic["host"]="host";
    attrDic["Counthost"]="host";
    attrDic["descriptionTFIDF"]="descriptionTFIDF";
    attrDic["nameTFIDF"]="nameTFIDF";
    attrDic["LDACosSim"]="LDA";
    attrDic["LDAHellingerSim"]="LDA";
    attrDic["LDAKLSim"]="LDA";
    return attrDic

def softwareAttrDic(): 
    attrDic={}
    attrDic["inLanguage"]="inLanguage";
    attrDic["CountinLanguage"]="inLanguage";
    attrDic["fileSizeInt"]="fileSize";
    attrDic["operatingSystem"]="operatingSystem";
    attrDic["CountoperatingSystem"]="operatingSystem";
    attrDic["descriptionTFIDF"]="descriptionTFIDF";
    attrDic["nameTFIDF"]="nameTFIDF";
    attrDic["LDACosSim"]="LDA";
    attrDic["LDAHellingerSim"]="LDA";
    attrDic["LDAKLSim"]="LDA";
    return attrDic

def tvseriesAttrDic(): 
    attrDic={}
    attrDic["numberOfEpisodesInt"]="numberOfEpisodes";
    attrDic["descriptionTFIDF"]="descriptionTFIDF";
    attrDic["nameTFIDF"]="nameTFIDF";
    attrDic["actor"]="actor";
    attrDic["Countactor"]="actor";
    attrDic["director"]="director";
    attrDic["Countdirector"]="director";   
    attrDic["LDACosSim"]="LDA";
    attrDic["LDAHellingerSim"]="LDA";
    attrDic["LDAKLSim"]="LDA";
    return attrDic

def vediogameAttrDic(): 
    attrDic={}
    attrDic["numberOfEpisodesInt"]="numberOfEpisodes";
    attrDic["descriptionTFIDF"]="descriptionTFIDF";
    attrDic["nameTFIDF"]="nameTFIDF";
    attrDic["genre"]="genre";
    attrDic["Countgenre"]="genre";
    attrDic["Countpublisher"]="publisher";
    attrDic["Countversion"]="version";
    attrDic["LDACosSim"]="LDA";
    attrDic["LDAHellingerSim"]="LDA";
    attrDic["LDAKLSim"]="LDA";
    return attrDic

def mainFunOfFeatureExtract(InfoDic,filePath):
    if "train" in filePath:
        filePath = filePath + "/train-"
    else:
        filePath = filePath + "/test-"
    
    type="Movie"
    inFile=filePath + type + ".txt"
    outFile=filePath + type + "-feature.txt"
    attrDic=movieAttrDic()
    featureGenerate(inFile,outFile,InfoDic,attrDic)
       
    attrDic=musicAttrDic()
    type="MusicRecording"
    inFile=filePath + type + ".txt"
    outFile=filePath + type + "-feature.txt"
    featureGenerate(inFile,outFile,InfoDic,attrDic)
    
    attrDic=showseriesAttrDic()
    type="ShowSeries"
    inFile=filePath + type + ".txt"
    outFile=filePath + type + "-feature.txt"
    featureGenerate(inFile,outFile,InfoDic,attrDic)
    
    attrDic=softwareAttrDic()
    type="SoftwareApplication"
    inFile=filePath + type + ".txt"
    outFile=filePath + type + "-feature.txt"
    featureGenerate(inFile,outFile,InfoDic,attrDic)
    
    attrDic=tvseriesAttrDic()
    type="TVSeries"
    inFile=filePath + type + ".txt"
    outFile=filePath + type + "-feature.txt"
    featureGenerate(inFile,outFile,InfoDic,attrDic)
    
    attrDic=vediogameAttrDic()
    type="VideoGame"
    inFile=filePath + type + ".txt"
    outFile=filePath + type + "-feature.txt"
    featureGenerate(inFile,outFile,InfoDic,attrDic)
 
if __name__ == '__main__':
    filePath = '../Data/train'