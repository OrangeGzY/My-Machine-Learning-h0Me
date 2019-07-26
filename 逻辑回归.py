#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:09:46 2019

@author: apple
"""
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from math import exp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length ', 'sepal width ', 'petal length ', 'petal width ', ' label ']
    data = np.array(df.iloc[:100,[0,1,-1]])
    return data[:,:2] , data[:, -1]

X , y = create_data() # 加载数据，x是两个特征的np矩阵， y是相对应的输出
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.3)  #切分出训练集和测试集

'''
#本段代码用于调试，方便理解matrix函数中的实际意义
print(y_train[:5])   
print(X_train[:5])
print("\n")
data_mat = []
for d in X_train:
    data_mat.append([1.0, *d])  #  *号的作用是统一将其转化为np.array 转化后相当于将 X_train中的每一哥向量添加了一个元素 1.0 变成[1.0 6.3 2.3]
print(np.array(data_mat))
'''


'''底层实现
class LogisticRegressionclf:
    def __init__(self,max_iter = 200, l_rate = 0.01):
        self.maxiter = max_iter
        self.l_rate = l_rate
        
    def sigmoid(self, x):
        return 1/(1+exp(-x))
    
    def matrix(self,X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat
    
    def fit(self,X,y):
        data_mat = self.matrix(X)
        self.weights = np.zeros((len(data_mat[0]),1), dtype = np.float32) # 参数向量的维数是data_mat矩阵第一行向量的长度n*1
        #下面开始进行迭代#
        #error = 0
        
        for iter_ in range(self.maxiter):
            for i in range(len(X)):                                            #len(X) 是指X矩阵的行数 此处对行做一个迭代， 在每一行上操作                
                result = self.sigmoid(np.dot(data_mat[i] , self.weights))       #sigmoid做数据的压缩处理每一次相乘都是一次fit
                error = y[i] - result                                          #误差计算
                self.weights += self.l_rate * error * np.transpose([data_mat[i]])  #更新参数向量进行梯度下降（负梯度方向）
        
        print("当前的学习率为：%f"%self.l_rate)
        print("最大迭代次数为：%d"%self.maxiter)
       
        
    def acc(self,X_test, y_test):
        right = 0
        X_test = self.matrix(X_test)
        for x,y in zip(X_test,y_test):
            result = np.dot(x,self.weights)  #此时参数矩阵weights已经训练好
            if(result > 0 and y == 1 or result < 0 and y ==0): #正确分类
                right += 1
        return right/len(X_test)   #返回acc

lr = LogisticRegressionclf()
lr.fit(X_test,y_test)
print(lr.acc(X_test,y_test))

x_points = np.arange(4,8)
y_ = -(lr.weights[1] * x_points + lr.weights[0])/lr.weights[2]
plt.plot(x_points, y_)
'''

#sklearn实现#
x_points = np.arange(4,8)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
print("参数向量：",clf.coef_)
print("截距：",clf.intercept_)
y_ = -(clf.coef_[0][0]*x_points + clf.intercept_)/clf.coef_[0][1]



plt.plot(x_points,y_)
plt.scatter(X[:50,0],X[:50,1], label = '0')
plt.scatter(X[50:,0],X[50:,1], label= '1')
plt.legend()







    
             
                

        
    
        
        
            

            
        
        
    


    
