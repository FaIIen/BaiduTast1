# -*- coding=utf-8 -*-
'''
Created on 2014年7月19日

@author: Yang
'''

import dataPre
import featureExtract
import classifier
import evaluation
import plot_precious_loss
import numpy



if __name__ == '__main__':
	
	#将最初始的entity转为<sid,attr,value>三元组
	'''
	dataPre.processJson('../Data/entity_normalization1.json', '../Data/task1_out_ttl.txt')
	#读取所有entity信息,并产生所有entity的额外属性
	infoDic = dataPre.readInfo()
	#将初始的train集合,以及测试集合按不同类别存放
	dataPre.outPutDifferentTypeResult(infoDic,'../Data/train/train.txt')
	dataPre.outPutDifferentTypeResult(infoDic,'../Data/test/test.txt')
	#生成训练集和测试集的特征
	featureExtract.mainFunOfFeatureExtract(infoDic,'../Data/train')
	featureExtract.mainFunOfFeatureExtract(infoDic,'../Data/test')
	'''
	
	tag = 2
	parameterModel = 2
	losslist = []
	acclist=[]
# 	for tag in range(1,8):
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-Movie-feature.txt","../Data/train/train-Movie-feature.txt","../Data/result/result-train-Movie-" + str(tag) + str(parameterModel) + ".txt",tag,parameterModel)
# 		[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-MusicRecording-feature.txt","../Data/train/train-MusicRecording-feature.txt","../Data/result/result-train-MusicRecording-" + str(tag) +".txt",tag)
	acc=evaluation.compute_accuracy(testTagArray,trueTagArray)
# 		matrix=evaluation.confusion_table(testTagArray, trueTagArray)
# 		p,microp=evaluation.matrix_precision(evaluation.confusion_table_withEdge(testTagArray,trueTagArray))
# 		r,micror=evaluation.matrix_recall(evaluation.confusion_table_withEdge(testTagArray,trueTagArray))
# 		print microp,micror
	#打印总体的整体正确率和召回率
	loss = evaluation.compute_Loss(testTagArray,trueTagArray)
	loss=float( '%.2f' % loss)
	losslist.append(loss) 
	a=acc*100
	t=float( '%.2f' % a)
	acclist.append(t)
	print acc
	print loss
		#打印各个类别的正确率和召回率
# 	methodlist=(u'SVM',u'LogReg',u'RandomForest',u'AdaBoost',u'GradientBoost',u'Bagging',u'LDA')
	#plot_precious_loss.plot_loss_precious(plist, rlist, methodlist)
# 	plot_precious_loss.plot_loss_precious(acclist,losslist,methodlist,"Movie on train",0)
	'''
	fout = open('../Data/result/test8.txt',"w+")
	 
	type = ["Movie","MusicRecording","ShowSeries","SoftwareApplication","TVSeries","VideoGame"]
	
	tag = 1
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-Movie-feature.txt","../Data/test/test-Movie-feature.txt","../Data/result/result-test-Movie-" + str(tag) +".txt",tag)
	fin = open("../Data/test/test-Movie.txt")
	i = 0
	for l in fin:##将每一行存放成六列数据的形式
		l = l.strip()
		tmpList = l.split("\t")
		key1 = tmpList[0].strip()
		key2 = tmpList[1].strip()
		value = testTagArray[i]
		fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[i]) + "\n")
		i += 1
		
# 	tag = 6
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-MusicRecording-feature.txt","../Data/test/test-MusicRecording-feature.txt","../Data/result/result-test-MusicRecording-" + str(tag) +".txt",tag)
	fin = open("../Data/test/test-MusicRecording.txt")
	i = 0
	for l in fin:##将每一行存放成六列数据的形式
		l = l.strip()
		tmpList = l.split("\t")
		key1 = tmpList[0].strip()
		key2 = tmpList[1].strip()
		value = testTagArray[i]
		fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[i]) + "\n")
		i += 1
	
# 	tag = 6
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-ShowSeries-feature.txt","../Data/test/test-ShowSeries-feature.txt","../Data/result/result-test-ShowSeries-" + str(tag) +".txt",tag)
	fin = open("../Data/test/test-ShowSeries.txt")
	i = 0
	for l in fin:##将每一行存放成六列数据的形式
		l = l.strip()
		tmpList = l.split("\t")
		key1 = tmpList[0].strip()
		key2 = tmpList[1].strip()
		value = testTagArray[i]
		fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[i]) + "\n")
		i += 1
		
# 	tag = 6
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-SoftwareApplication-feature.txt","../Data/test/test-SoftwareApplication-feature.txt","../Data/result/result-test-SoftwareApplication-" + str(tag) +".txt",tag)
	fin = open("../Data/test/test-SoftwareApplication.txt")
	i = 0
	for l in fin:##将每一行存放成六列数据的形式
		l = l.strip()
		tmpList = l.split("\t")
		key1 = tmpList[0].strip()
		key2 = tmpList[1].strip()
		value = testTagArray[i]
		fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[i]) + "\n")
		i += 1
			
# 	tag = 6
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-TVSeries-feature.txt","../Data/test/test-TVSeries-feature.txt","../Data/result/result-test-TVSeries-" + str(tag) +".txt",tag)
	fin = open("../Data/test/test-TVSeries.txt")
	i = 0
	for l in fin:##将每一行存放成六列数据的形式
		l = l.strip()
		tmpList = l.split("\t")
		key1 = tmpList[0].strip()
		key2 = tmpList[1].strip()
		value = testTagArray[i]
		fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[i]) + "\n")
		i += 1
			
# 	tag = 6
	[testTagArray,trueTagArray] = classifier.getAllData("../Data/train/train-VideoGame-feature.txt","../Data/test/test-VideoGame-feature.txt","../Data/result/result-test-VideoGame-" + str(tag) +".txt",tag)
	fin = open("../Data/test/test-VideoGame.txt")
	i = 0
	for l in fin:##将每一行存放成六列数据的形式
		l = l.strip()
		tmpList = l.split("\t")
		key1 = tmpList[0].strip()
		key2 = tmpList[1].strip()
		value = testTagArray[i]
		fout.write(key1 + " \t" + key2 + " \t" + str(testTagArray[i]) + "\n")
		i += 1
	'''
		
