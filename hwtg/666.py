import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
#####################二分-K均值聚类算法############################
 
def distEclud (vecA, vecB):
    """
    计算两个坐标向量之间的街区距离 
    """
    return np.sum(abs(vecA - vecB))
 
def randCent( dataSet, k):
    n = dataSet.shape[1]  #列数
    centroids = np.zeros((k,n)) #用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:,j],axis = 0)
        rangeJ = float(np.max(dataSet[:,j])) - minJ
        for i in range(k):
            centroids[i:,j] = minJ + rangeJ * (i+1)/k
    return centroids
 
def kMeans (dataSet,k,distMeas = distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m,2))  #这个簇分配结果矩阵包含两列，一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的街区距离
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[ np.nonzero(clusterAssment[:,0]==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)
    return centroids , clusterAssment
 
 
 
def biKmeans(dataSet,k,distMeas= distEclud):
    """
    这个函数首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。
    输入：dataSet是一个ndarray形式的输入数据集 
          k是用户指定的聚类后的簇的数目
         distMeas是距离计算函数
    输出:  centList是一个包含类质心的列表，其中有k个元素，每个元素是一个元组形式的质心坐标
            clusterAssment是一个数组，第一列对应输入数据集中的每一行样本属于哪个簇，第二列是该样本点与所属簇质心的距离
    """
    m = dataSet.shape[0]
    clusterAssment =np.zeros((m,2))
    centroid0 = np.mean(dataSet,axis=0).tolist()
    centList = []
    centList.append(centroid0)
    for j in range(m):
         clusterAssment[j,1] = distMeas(np.array(centroid0),dataSet[j,:])**2
    while len(centList) <k:       #小于K个簇时
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0] == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = np.sum(splitClustAss[:,1])
            sseNotSplit = np.sum( clusterAssment[np.nonzero(clusterAssment[:,0]!=i),1])
            if (sseSplit + sseNotSplit) < lowestSSE:         #如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0] ==1)[0],0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0,:].tolist()
        centList.append( bestNewCents[1,:].tolist())
        clusterAssment[np.nonzero(clusterAssment[:,0] == bestCentTosplit)[0],:] = bestClustAss
    return centList, clusterAssment
 
 
def split_licensePlate_character(plate_binary_img):
    """
    此函数用来对车牌的二值图进行水平方向的切分，将字符分割出来
    输入： plate_gray_Arr是车牌的二值图，rows * cols的数组形式
    输出： character_list是由分割后的车牌单个字符图像二值图矩阵组成的列表
    """
    plate_binary_Arr = np.array ( plate_binary_img )
    row_list,col_list = np.nonzero (  plate_binary_Arr >= 255 )
    dataArr = np.column_stack(( col_list,row_list))   #dataArr的第一列是列索引，第二列是行索引，要注意
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])
    split_list =[]
    for centroids_ in  centroids_sorted:
        i = centroids.index(centroids_)
        current_class = dataArr[np.nonzero(clusterAssment[:,0]==i)[0],:]
        x_min,y_min = np.min(current_class,axis =0 )
        x_max, y_max = np.max(current_class, axis=0)
        split_list.append([y_min, y_max,x_min,x_max])
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]:split_list[i][3]]
        character_list.append( single_character_Arr )
        cv2.imshow('character'+str(i),single_character_Arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    return character_list              #character_list中保存着每个字符的二值图数据
 
    ############################
    #测试用
    #print(col_histogram )
    #fig = plt.figure()
    #plt.hist( col_histogram )
    #plt.show()
    ############################


if __name__ == "main":
    card_imgs, colors = pretreatment(car_pic="car_test.jpg")
    cv2.imshow("card_imgs", card_imgs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
