# Larger CNN for the MNIST Dataset
# 2.Negative dimension size caused by subtracting 5 from 1 for 'conv2d_4/convolution' (op: 'Conv2D') with input shapes
# 3.UserWarning: Update your `Conv2D` call to the Keras 2 API: http://blog.csdn.net/johinieli/article/details/69222956
# 4.Error when checking input: expected conv2d_1_input to have shape (None, 28, 28, 1) but got array with shape (60000, 1, 28, 28)

# talk to wumi,you good .

# python 3.5.4
# keras.__version__  : 2.0.6
# thensorflow 1.2.1
# theano 0.10.0beta1

# good blog
# http://blog.csdn.net/shizhengxin123/article/details/72383728
# http://www.360doc.com/content/17/0415/12/1489589_645772879.shtml

# recommand another framework  http://tflearn.org/examples/

import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.preprocessing import image
import skimage.io



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
#mnist.load_data()函数返回的对象data包含是一个三元组tuple:
#应该这样构造 (X_train, y_train),(X_val,y_val), (X_test, y_test) = mnist.load_data()  ，示例只有训练集和测试集。
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)

plt.imshow(X_train[12], cmap=plt.get_cmap('gray'))
#plt.show()
print(y_train)
print(y_train[12])
print(X_train.shape[0])
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# print(X_test[1].shape)  (28, 28, 1)
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')    <---4
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# print(X_test[1].shape)  归一化没有变化
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
#print(y_train[0])  #[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


###raw
# define the larger model
def larger_model():
    # create model
    model = Sequential()   #初始化一个神经网络
    #“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
    model.add(Conv2D(30, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    # model.add(Conv2D(30, (5, 5), padding='valid', input_shape=(28, 28,1), activation='relu'))   <----3,2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))   #  f防止过拟合 ，提高模型的泛化能力 Dropout layer 随机扔掉当前层一些weight，相当于废弃了一部分Neurons 它还有另一个名字 dropout regularization, 所以你应该知道它有什么作用了：
                               #降低模型复杂度，增强模型的泛化能力，防止过拟合[1]。
                               #顺带降低了运算量。。。
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())    #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到(Convolution)全连接层(Dense)的过渡。
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    # optimizer  优化器
    # loss 损失函数      梯度下降
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = larger_model()

#check the summary of model
model.summary()

# Fit the model
# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,
          verbose=2)  # epochs 200 too bigger
# model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=200, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

# save the model
model.save('D:\\works\\jetBrians\\PycharmProjects\\tryPicture\\my_model_test1.h5')  # creates a HDF5 file 'my_model.h5'
del model

''''
数据很庞大的时候,需要使用 epochs，batch size，迭代这些术语，在这种情况下，一次性将数据输入计算机是不可能的。
因此，为了解决这个问题，我们需要把数据分成小块，一块一块的传递给计算机，
在每一步的末端更新神经网络的权重，拟合给定的数据。
epochs:  当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch。
batch_size: 当一个 epoch 对于计算机而言太庞大的时候，就需要把它分成多个小块。
verbose:verbose: 0 表示不更新日志, 1 更新日志, 2 每个epoch一个进度行. 

'''