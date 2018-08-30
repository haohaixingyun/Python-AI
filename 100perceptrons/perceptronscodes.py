'''
单层感知机是最简单的神经网络，这是是用code 实现单层感知机分类
'''

import  numpy as np
import matplotlib.pyplot as plt

###定义输入X  ,得到一个3行3列的矩阵

X = np.array([[1,3,3],
             [1,4,3],
             [1,1,1]])

#定义标签
Y = np.array([1,1,-1])

###定义三个权重 W  是一个3行1列的数据
W= (np.random.random(3)-0.5)*2   #x = (np.random.random(3)-0.5)*2   # generate  3  numbers   0  1  ---(-0.5)-->   -0.5 0.5   ----*2 --->  -1 1

###实际输出
O = 0
'''
学习率的调整   -learning rate adjustment
为了能够使得梯度下降法有较好的性能，我们需要把学习率的值设定在合适的范围内。
学习率决定了参数移动到最优值的速度快慢。如果学习率过大，很可能会越过最优值；
反而如果学习率过小，优化的效率可能过低，长时间算法无法收敛。
所以学习率对于算法性能的表现至关重要。
'''
#设置学习率
lr = 0.11
##计算迭代次数
n = 0

# 更新权重
def update():
    global  X,Y,W,lr ,n
    n+=1
    O = np.sign(np.dot(X, W.T))  # 实际输出
    W_D = lr*(Y-O).dot(X)/X.shape[0]  #德尔特
    wd = (Y-O).dot(X)/X.shape[0]
    print(wd)
    print(W_D)
    W =W + W_D

for _ in range(100):
    update()
    print(W)
    print(n)

    O = np.sign(np.dot(X, W.T))
    if(Y.T == O).all():
        print(W)
        print(n)
        break

#正样本
x1 = [3,4]
y1 = [3,3]
#负样本
x2 = [1]
y2 = [1]

#计算分界线的斜率以及截距  这个地you意思
k = -W[1]/W[2]
d = -W[0]/W[2]
print('k=',k)
print('d=',d)

'''
x0=1
x0*w0+x1*w1+x2*w2 = 0    #等于零是因为我们的 Y 是 -1 1 属于分类问题
#所以会就出 x1 x2 的关系 ，构成  k 和 d   ，一元一次方程 ，求得 Y
'''


xdata = np.linspace(0,5)

plt.figure()
plt.plot(xdata,xdata*k+d,'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()




