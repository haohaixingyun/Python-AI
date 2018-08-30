from keras.models import  load_model
from keras.preprocessing import image
import  numpy as np
import matplotlib.pylab as plt
from keras import models
from keras import layers
model = load_model('D:\\sthself\\ml\\cats_and_dogs_small_2.h5')
model.summary()

img_path='D:\\sthself\\ml\\dog.1111.jpg'
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)

img3 = np.reshape(img_tensor,(1,150,150,3)).astype('float32')

predict= model.predict(img3,batch_size=1,verbose=0)
print(predict[0])



