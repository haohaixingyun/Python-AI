import  numpy as np
import  matplotlib.pyplot as plt

xdate =np.linspace(-5,5)

plt.figure()

plt.plot(xdate,np.sigmoid(xdate),'r')

plt.show()