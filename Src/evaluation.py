#-*- coding=utf-8 -*-

#得到混淆矩阵，不标类别
import math
def confusion_table(trainedList, labeledList):
    tagList=[]#存放所有类别
    matrix=[]#存放最终的混淆矩阵
    for i in trainedList:
        if tagList.__contains__(i)==False:
            tagList.append(i)
    tagList.sort(cmp=None, key=None, reverse=False)
    tagLength = len(tagList)
    listLength = len(trainedList)
    
    for line in range(tagLength):
        countList = []
        for c in range(tagLength):
            countList.append(0)
        for pair in range(listLength):#对每一对实体对
            if labeledList[pair]==tagList[line]:#labeledList的类别必须与line的对应类别相同
                if trainedList[pair]==tagList[line]:
                    countList[line]=countList[line]+1
                else:
                    countList[tagList.index(trainedList[pair])]=countList[tagList.index(trainedList[pair])]+1
        #循环完所有的实体对
        matrix.append(countList)
        
    return matrix

#得到混淆矩阵，标类别;第一行和第一列为类别标签，[0][0]是类别标签的总数目
def confusion_table_withEdge(trainedList, labeledList):
    tagList=[]#存放所有类别
    matrix=[]#存放最终的混淆矩阵
    for i in trainedList:
        if tagList.__contains__(i)==False:
            tagList.append(i)
    tagList.sort(cmp=None, key=None, reverse=False)
    tagLength = len(tagList)
    listLength = len(trainedList)
   
    for line in range(tagLength):
        countList = []
        for c in range(tagLength):
            countList.append(0)
        for pair in range(listLength):#对每一对实体对
            if labeledList[pair]==tagList[line]:#labeledList的类别必须与line的对应类别相同
                if trainedList[pair]==tagList[line]:
                    countList[line]=countList[line]+1
                else:
                    countList[tagList.index(trainedList[pair])]=countList[tagList.index(trainedList[pair])]+1
        #循环完所有的实体对
        matrix.append(countList)
        
    #加边界
    for i in range(tagLength):
        matrix[i].insert(0,tagList[i])
    tagList.insert(0, tagLength)
    matrix.insert(0, tagList)    
                  
    return matrix

#得到混淆矩阵的精确率
def matrix_precision(matrix):
    N = matrix[0][0]
    P = 0
    sumList = []
    correctList = []
    microP = []
    for i in range(N):
        sumList.append(0)#存在matrix中每一列的和（除去类别的数值）
    for loop in range(N):
        correctList.append(matrix[loop+1][loop+1])
        for item in range(N):
            sumList[loop]=sumList[loop]+matrix[item+1][loop+1]
    for i in range(N):
        P = P+float(correctList[i])/float(N*sumList[i])
        microP.append(float(correctList[i])/float(sumList[i]))
    return P,microP#总体的精确率和每一类别对应的精确率
    
#得到混淆矩阵的召回率            
def matrix_recall(matrix):
    N = matrix[0][0]
    R = 0
    microR = []
    for loop in range(N):
        correctNum = matrix[loop+1][loop+1]
        sum = 0
        for item in matrix[loop+1]:
            sum = sum+int(item)
        sum=sum-int(matrix[loop+1][0])
        if sum==0:
            return '存在分母为0的情况'#目前还未遇到这种情况
        R = R+float(correctNum)/float(sum*N)
        microR.append(float(correctNum)/float(sum))
    return R,microR#总体的召回率和每一类别对应的召回率

#读列表文件得到两个list，第一列为机器标注结果，第二列为人工标注结果
def get_list(path):
    trainedList=[]
    labeledList=[]
    file = open(path)
    for line in file:
        trainedList.append(line.split("\t")[0])
        labeledList.append(line.replace("\n","").split("\t")[1])
    return trainedList,labeledList

def compute_Loss(evaList,rightList):
    '''计算损失率'''
    n=len(evaList)
    loss = 0
    for i in range(0,n):
        loss += (evaList[i] - rightList[i]) * (evaList[i] - rightList[i])
    return math.sqrt(loss)


def compute_accuracy(evaList,rightList):
    '''计算正确率'''
    same=0
    index=0
    length=len(evaList)
    if(length==len(rightList)):
        while index<length:
            if(evaList[index]==rightList[index]):
                same +=1
            index +=1
    else:
        print "列表长度不同"
    return same*1.0/length

'''
测试
'''
'''
list1=[1,2,4,4,2,1,3,4,3,4]
list2=[1,4,2,4,2,1,3,3,3,4]
listmatrix = confusion_table(list1,list2)
for m in listmatrix:
    print m
print matrix_precision(listmatrix)
print matrix_recall(listmatrix)


l1,l2=get_list('C:/test.txt')#读列表文件得到两个list，第一列为机器标注结果，第二列为人工标注结果
newmatrix= confusion_table_withEdge(l1,l2)
for m in newmatrix:
    print m
print matrix_precision(newmatrix)
print matrix_recall(newmatrix)
'''