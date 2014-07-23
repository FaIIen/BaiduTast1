# -*- coding=utf-8 -*-
import numpy as np
import math

def kl(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions����KLɢ��
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    thanks to gist : https://gist.github.com/larsmans/3104581
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    sum_pq = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    sum_qp = np.sum(np.where(q != 0, q * np.log(q / p), 0))
    return (sum_pq+sum_qp)/2 # symmetric

def cos_dist(a, b):##�����������ƶ�
    if a.shape[1] != b.shape[1]:
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for i in range(0,a.shape[1]):
        a1 = a[0,i]
        b1 = b[0,i]
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / (part_down+0.0)
    
def datePublishSim(date1,date2):
    '''�����֮������ƶ�'''
    maxmargin = 100#�������֮����󳤶�Ϊ100
    minus = 0 #���֮���ֵ��ʼֵΪ0
    if date1 in "none" or date2 in "none":
        minus = 100
    else:
        minus = math.fabs(float(date1)-float(date2))
    return ((1+1.0/maxmargin)/(minus+1) - 1.0/maxmargin)

def jaccardSim(list1,list2):
    '''��jaccard���ƶ�'''
    if list1 == ["none"] or list2 == ["none"]:#�����ڿռ�
        return 0
    try:
        listInter  = [val for val in list1 if val in list2]#����
        listMerge  = list(set(list1+list2))
        return (len(listInter)+0.0)/(len(listMerge)+0.0)
    except:
        return 0

def mixSim(list1,list2):
    '''�������б?����С'''
    if list1 == ["none"] or list2 == ["none"]:#�����ڿռ�
        return 0
    
    listInter  = [val for val in list1 if val in list2]#����
    return len(listInter)

