import cv2 as cv


def find_plate_contours(path):
    img = cv.imread("car_plates_img/car_p_1.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, th_img = cv.threshold(img, 185, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
    contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if i == 0:
            continue

        # Approximate contour shape
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)

        # Draw contour
        cv.drawContours(th2, [contour], 0, (0, 0, 255), 5)

        # Find center
        # M = cv.moments(contour)
        # if M['m00'] != 0:
        #     x = int(M['m10'] / M['m00'])
        #     y = int(M['m01'] / M['m00'])

        # # Detect shape
        # sides = len(approx)
        # if sides == 3:
        #     label = 'Triangle'
        # elif sides == 4:
        #     label = 'Quadrilateral'
        # elif sides == 5:
        #     label = 'Pentagon'
        # elif sides == 6:
        #     label = 'Hexagon'
        # else:
        #     label = 'Circle'

        # cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


# for i in range(50, 255, 10):
#     img = read_img_threshhold("car_plates_img/car_p_1.jpg", i)
#     cv.imshow(str(i), img)
#     cv.waitKey(0)

img = find_plate_contours("car_plates_img/car_p_1.jpg")
cv.imshow("abob", img)
cv.waitKey(0)