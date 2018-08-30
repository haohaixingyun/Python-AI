import numpy
import skimage.io
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model

#if the picture is bigger than 28*28 will get below error
#ValueError: cannot reshape array of size 775440 into shape (1,28,28,1)

image0 = 'D:\\sthself\\ml\\reshape0.jpg'
image1 = 'D:\\sthself\\ml\\reshape1.jpg'
image5 = 'D:\\sthself\\ml\\reshape5.jpg'
image3 = 'D:\\sthself\\ml\\reshape3.jpg'
image9 = 'D:\\sthself\\ml\\reshape9.jpg'
image8 = 'D:\\sthself\\ml\\reshape8.jpg'
image7 = 'D:\\sthself\\ml\\reshape7.jpg'
image4 = 'D:\\sthself\\ml\\reshape4.jpg'
img0 = skimage.io.imread(image0,as_grey=True)
img1 = skimage.io.imread(image1,as_grey=True)
img3 = skimage.io.imread(image3,as_grey=True)
img4 = skimage.io.imread(image4,as_grey=True)
img5 = skimage.io.imread(image5,as_grey=True)
img7 = skimage.io.imread(image7,as_grey=True)
img8 = skimage.io.imread(image8,as_grey=True)
img9 = skimage.io.imread(image9,as_grey=True)
#skimage.io.imshow(img6)
#plt.show()

#img3 is a matrix
img0 = numpy.reshape(img0,(1,28,28,1)).astype('float32')
img1 = numpy.reshape(img1,(1,28,28,1)).astype('float32')
img3 = numpy.reshape(img3,(1,28,28,1)).astype('float32')
img4 = numpy.reshape(img4,(1,28,28,1)).astype('float32')
img5 = numpy.reshape(img5,(1,28,28,1)).astype('float32')
img7 = numpy.reshape(img7,(1,28,28,1)).astype('float32')
img8 = numpy.reshape(img8,(1,28,28,1)).astype('float32')
img9 = numpy.reshape(img9,(1,28,28,1)).astype('float32')

# rebuild the model  ,do we need to add the layer ?  AttributeError: 'Sequential' object has no attribute 'load_model'

#If you stored the complete model, not only the weights, in the HDF5 file, then it is as simple as
#from keras.models import load_model
#model = load_model('model.h5')
# examples https://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras
modelTrained = load_model('D:\\works\\jetBrians\\PycharmProjects\\tryPicture\\my_model.h5')

# we should get a correct answer is  2
predict = modelTrained.predict(img5, verbose=0)
#list of predicted labels and their probabilities
print(predict[0])
#[ 0.04785086  0.02547075  0.06954221  0.03620625  0.01439319  0.03016909   0.03120618  0.00815302  0.70513636  0.03187207]

# AttributeError: 'Sequential' object has no attribute 'prediect_classes'
result = modelTrained.predict_proba(img8,batch_size=1, verbose=1)
#print(result)
# this will tell us the picture number
print('your number is %d ' % numpy.argmax(result))

print("tensorflow hello word is done")
#0
#[[  9.16964829e-01   7.21327378e-04   1.40018184e-02   4.35708603e-03  2.05853744e-03   8.09434336e-03   1.69250034e-02   4.08325950e-03   2.04275250e-02   1.23660685e-02]]
#1
#[  3.36092742e-12   1.00000000e+00   1.08299070e-09   5.20451598e-11    9.97397720e-09   1.72608400e-10   6.49094223e-10   3.02566017e-09   2.90529165e-08   3.44137469e-10]
#2
#[[  1.57670729e-06   6.46576664e-05   9.73940611e-01   6.88192609e-04   9.37903587e-06   2.25085461e-08   3.93761432e-08   2.52519064e-02    2.84847065e-05   1.53232359e-05]]
#3
#[[  2.90367685e-09   9.20389880e-07   2.24796640e-05   9.99938488e-01  4.98295227e-09   1.78251739e-05   5.97671190e-10   5.83014980e-06   9.59870795e-06   4.71648946e-06]]
#4
#[[  1.56145532e-08   5.23463200e-07   6.88577231e-08   2.33658180e-11  9.99997020e-01   1.09018260e-10   1.06191813e-07   1.43476242e-07    1.74371181e-07   2.02444994e-06]]
#5
#6
#[[  4.32068575e-03   2.60723429e-03   2.94188480e-03   8.93787597e-04  5.23575395e-03   2.51416136e-02   8.03044200e-01   1.16640826e-04   1.54893070e-01   8.05063930e-04]]
#7
#[[  4.48786039e-07   1.18592514e-04   8.68750084e-03   4.42354055e-03   1.39327667e-05   4.07292958e-07   6.23601437e-09  9.85468328e-01   5.98585466e-05   1.22733077e-03]]