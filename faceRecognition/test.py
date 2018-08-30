#导入CV模块
import cv2 as cv


#读取图像，支持 bmp、jpg、png、tiff 等常用格式
img = cv.imread("D:\sthself\ml\guobiting.jpg")
#打印图片的大小
print(img.shape)
print(img.dtype)
#创建窗口并显示图像
cv.namedWindow("Image")
res=cv.resize(img,(280,280),interpolation=cv.INTER_CUBIC)
cv.imshow("Image",res)
cv.waitKey(0)
#释放窗口
cv.destroyAllWindows()