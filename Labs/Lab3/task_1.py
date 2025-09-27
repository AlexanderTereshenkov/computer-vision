import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img/car1.jpg')

print(img.shape)
print("Pixel color at [100, 100]: ", img[100, 100])

#blue pixel -> BGR
print("Blue pixel: ", img[100, 100, 0])

img[100,100] = [255,255,255]
print("Modify pixel: ", img[100,100])

#Use some new methods from np instead of np like python because of perfomance
print("Red pixel value: ", img.item(10,10,2))

# img.itemset((10,10,2),100) REMOVED FROM LIBRARY
# print("Modify pixel new method: ", img.item(10,10,2))

print("Img shape = ", img.shape)
print("Img size CxHxW = ", img.size)
print("Img dtype = ", img.dtype)


area = img[120:450, 200:500]
img[0:330, 0:300] = area
cv.imshow("Results", img)
cv.waitKey(0)

#split and merge channels
b,g,r = cv.split(img)
img = cv.merge((b,g,r))

#or slices
b = img[:,:,0]
img[:,:,2] = 0
cv.imshow("Results", img)
cv.waitKey(0)

BLUE = [255,0,0]
 
img1 = cv.imread('img/car2.jpg')

size = 100

replicate = cv.copyMakeBorder(img1,size,size,size,size,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,size,size,size,size,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,size,size,size,size,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,size,size,size,size,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,size,size,size,size,cv.BORDER_CONSTANT,value=BLUE)
 
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
 
plt.show()