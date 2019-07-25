#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:26:45 2019

@author: apple
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
import math
#%matplotlib inline

#数据加载
iris = load_iris()
a = iris.data
df = pd.DataFrame(a,columns = iris.feature_names)

df['label'] = iris.target
df.columns = ['sepal length','sepal width','petal length','petal width','label'] #重构colums

#print(df.iloc[:5,[0,1,-1]])


'''
plt.scatter(df[:50]['sepal length'],df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'], label = '1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
'''

data = np.array(df.iloc[:100,[0,1,-1]])  #使用iris数据集的前100的样本的第一列第二列 倒数第一列
X = data[:100,:-1]
y = data[:100,-1]

y = np.array([1 if i==1 else -1 for i in y])  #分成正负两类


#print(len(data[0])-1)w
#print(np.ones(len(data[0])))
#w = np.ones(len(data[0]-1)-1, dtype = np.float32)
#print(w.shape)
#print(X[0].shape)

'''
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]-1)-1, dtype = np.float32)
        self.b = 0
        self.l_rate = 0.1 #学习率
    
    def sign(self,x,w,b):
        y = np.dot(x,w)+b
        return y
    
    #随机梯度下降
    def fit(self,X_train,y_train):
        is_wrong = False
        while not is_wrong: #当is_wrong为假
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w,self.b) <=0 :  #误分类
                    self.w = self.w + self.l_rate * y * X
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong  = True                
        return 'Perceptron Model!'
    
    def score(self):
        pass

perceptron = Model()
outcome = perceptron.fit(X,y)
print(outcome)
x_points = np.linspace(4, 7,10)
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
plt.plot(x_points, y_)
'''


clf = Perceptron(fit_intercept = False,max_iter = 1000,shuffle = False) #intercept:截距  max_iter:最大迭代次数 shuffle:随机排序，洗牌
clf.fit(X,y)
x_points = np.arange(4, 8)  # x 轴的取值
#print(clf.coef_)
#print(clf.intercept_)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_)/(clf.coef_[0][1]-4)
plt.plot(x_points, y_)  #一条直线
#print(clf.coef_[0][0])
#print(clf.coef_[0][1])

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()




'''
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
'''


    
        
           
                    
                    
                

        
        
    
        










