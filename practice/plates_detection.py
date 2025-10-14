import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("car_plates_img/car_p_2.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, th_img = cv.threshold(img, 103, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(th_img, cv.RETR_EXTERNAL,
                                            cv.CHAIN_APPROX_NONE)


area = -1
x_plate, y_plate, w_plate, h_plate = 0, 0, 0, 0
plate_cnt = None

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    aspect = w / h
    if aspect > 3 and aspect < 7:
        count_area = cv.contourArea(cnt)
        if count_area > area:
            area = count_area
            x_plate, y_plate, w_plate, h_plate = x, y, w, h
            plate_cnt = cnt

# rect = cv.minAreaRect(plate_cnt)
# print(rect)
# center, angle, scale = rect[0], rect[2], 1
# height, width = img.shape
# center = (int(width / 2), int(height / 2))
# M = cv.getRotationMatrix2D(center, angle, scale)

# croped_img = img[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

# rotated_image = cv.warpAffine(img, M, img.shape)
# cv.imshow("AAAAA", rotated_image)
# cv.waitKey(0)

rect = cv.minAreaRect(plate_cnt)
box = cv.boxPoints(rect)
box = np.intp(box)

# Угол
angle = rect[2]
if rect[1][0] < rect[1][1]:
    angle += 90

# Матрица поворота
(h, w) = img.shape
M = cv.getRotationMatrix2D(rect[0], angle, 1.0)

# Поворот изображения
rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC)

# После поворота — вырезаем номер
x, y, w, h = cv.boundingRect(cv.boxPoints(((rect[0]), (rect[1]), 0)))
cropped = rotated[y:y+h, x:x+w]

img2 = cv.imread("car_plates_img/car_p_3.jpg")
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

plt.subplot(121)
plt.imshow(img2, cmap="gray")
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cropped, cmap="gray")
plt.title('Croped')
plt.xticks([])
plt.yticks([])
plt.show()