#-*- coding=utf-8 -*-
#!
#FileName:plot_precious_loss.py
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
 
def plot_loss_precious(preciouslist,recalltuple,methodtuple,title,disLegend):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    '''
    methodtuple采用这样的格式，例如methodlist=(u'方法名1',u'方法2名'…………),u不可少
    preciouslist 采用这样的格式,例如，preciouslist=（50,60,75…………），代表50%，60%，75%…………,也可以是百分比的小数元组或列表
    loss采用的格式和preciouslist类似。
    '''
   
    plt.xlabel(u"方法",fontproperties=font)
    plt.ylabel(u"百分比",fontproperties=font)
    plt.ylim(0,100)
    length=len(recalltuple)
    tick=range(1,3*length,3)
    plt.xticks(tick,methodtuple)
    tick=range(0,3*length,3)
    rectloss=plt.bar(left=tick,height=recalltuple,width=1,yerr=0.0001)
    autolabel(rectloss)
    a=range(1,length*3,3)#为了使一个方法的正确率和折损率的柱形图相邻
    rectprecious=plt.bar(left=a,height=preciouslist,width=1,color='red',yerr=0.0001)
    autolabel(rectprecious)
    #如需更换图表的名称，请替换下一句代码中字符串中的文字
    plt.title(title)
    if disLegend == 1:
        plt.legend((rectloss,rectprecious),(u'loss',u'accuracy'),loc=0)
    plt.show()
    #plt.savefig("1.png",dpi=120)#调用plt.show()会阻塞进程，而这句代码不会


def plot_loss(losstuple,methodtuple):
    '''计算只有折损率的情况，list的每个项目中含有（“方法名”，折损率）
    methodlist采用这样的格式，例如methodlist=(u'方法名1',u'方法2名'),u不可少
    '''
    plt.xlabel(u"方法")
    plt.ylabel(u"百分比")     
    #如需更换图表的名称，请替换下一句代码中字符串中的文字
    #plt.title(u"折损率")
    length=len(losstuple)
    tick=range(length)
    plt.xticks(tick,methodtuple)
    rect=plt.bar(left=tick,height=losstuple,width=0.5,align="center",yerr=0.000001)
    autolabel(rect)
    plt.legend((rect,),(u'折损率',),loc=0)
    plt.show()

def autolabel(rects):
    for rect in rects:
        height=rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2,1.03*height,'%s'%float(height))
        
def changeList(plist,rlist,zlist):#得到各个类别的率
    pplist=plist[1]
    rrlist=rlist[1]
    zzlist=zlist[1]
    return pplist,rrlist,zzlist

def ListGetP_R_Z(plist,rlist,zlist):#得到的正确率，折损率，召回率
    pplist=[plist[0]]
    for val in plist[1]:
        pplist.append(val)
    rrlist=[rlist[0]]
    for val in rlist[1]:
        rrlist.append(val)
    zzlist=[zlist[0]]
    for val in zzlist:
        zzlist.append(val)
    return pplist,rrlist,zzlist

#对于只有方法名和折损率的情况的绘图，利用函数plot_loss进行试验,如下代码：
'''
a=(u"放任",u"B",u"c")
b=(30,50,40)
plot_loss(b,a)
'''
#对于含有正确率和折损率的调用plot_loss_precious函数，下面代码可以试验：
'''
m=(u"A",u"B",u"C",u'D')#方法名
z=(30,50,40,70)#折损率
zz=(0.3,0.5,0.4,0.7)
pp=(0.5,0.75,0.6,0.9)
p=(50,75,60,90)
plot_loss_precious(pp,zz,m)
'''

