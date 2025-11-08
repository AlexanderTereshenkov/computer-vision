import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def find_plate_2(img, debug=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Улучшаем бинаризацию
    edges = cv.Canny(blur, 100, 200)
    
    # Морфологическая обработка — соединяем символы в прямоугольники
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    morph = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)

    _, morph = cv.threshold(morph, 105, 255, cv.THRESH_BINARY)
    
    contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    plate_cnt = None
    best_score = 0

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect = w / h if h != 0 else 0
        area = cv.contourArea(cnt)
        rect_area = w * h
        fill_ratio = area / rect_area if rect_area > 0 else 0

        # Фильтруем по базовым геометрическим признакам
        if 2.0 < aspect < 8 and 1000 < area < 25000 and 0.1 < fill_ratio < 0.9:
            # Проверка на горизонтальность (номера не сильно наклонены)
            rect = cv.minAreaRect(cnt)
            angle = abs(rect[2])
            if angle > 15:
                continue

            score = area * fill_ratio
            if score > best_score:
                best_score = score
                plate_cnt = cnt

    if plate_cnt is None:
        return None

    x, y, w, h = cv.boundingRect(plate_cnt)
    cropped = img[y:y + h, x:x + w]

    if debug:
        debug_img = img.copy()
        cv.drawContours(debug_img, [plate_cnt], -1, (0, 255, 0), 2)
        plt.imshow(cv.cvtColor(debug_img, cv.COLOR_BGR2RGB))
        plt.title("Detected Plate")
        plt.show()

    # Контраст для OCR
    gray_plate = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    gray_plate = cv.equalizeHist(gray_plate)
    
    return gray_plate


# Пример использования
for i in range(1, 9):
    img = cv.imread(f"car_plates_img/car_p_{i}.jpg")
    result = find_plate_2(img, debug=False)
    if result is not None:
        cv.imshow("Detected Plate", result)
        cv.waitKey(0)
