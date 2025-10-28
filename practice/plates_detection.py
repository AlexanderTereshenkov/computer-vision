import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def find_plate(img, th_img):
    contours, hierarchy = cv.findContours(th_img, cv.RETR_EXTERNAL,
                                                cv.CHAIN_APPROX_SIMPLE)
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

    rect = cv.minAreaRect(plate_cnt)
    box = cv.boxPoints(rect)
    box = np.intp(box)

    angle = rect[2]
    if rect[1][0] < rect[1][1]:
        angle += 90

    (h, w) = img.shape
    M = cv.getRotationMatrix2D(rect[0], angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC)
    x, y, w, h = cv.boundingRect(cv.boxPoints(((rect[0]), (rect[1]), 0)))
    cropped = rotated[y:y+h, x:x+w]
    return cropped




def find_plate_2(img, th_img, debug=False):
    kernel = np.ones((5, 5), np.uint8)
    th_closed = cv.morphologyEx(th_img, cv.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv.findContours(th_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = 0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 500:  # отсечь мелкие области
            continue
        rect = cv.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        # --- Нормализация ориентации ---
        if w < h:
            w, h = h, w
            angle += 90

        aspect = w / h
        if not (2.0 < aspect < 8.0):
            continue

        # --- Оценка плотности (заполненности) области ---
        box = cv.boxPoints(rect)
        box = np.intp(box)
        mask = np.zeros(th_closed.shape, dtype=np.uint8)
        cv.drawContours(mask, [box], -1, 255, -1)
        filled = cv.countNonZero(cv.bitwise_and(mask, th_closed))
        fill_ratio = filled / (w * h)

        if fill_ratio < 0.2:  # слишком пустая область
            continue

        score = area * fill_ratio
        if score > best_score:
            best_score = score
            best_rect = rect

    if best_rect is None:
        return None, None

    (cx, cy), (w, h), angle = best_rect

    # --- Коррекция угла ---
    # if angle < -45:
    #     angle += 90
    #     w, h = h, w

    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = 90 + angle

    # --- Поворот изображения ---
    M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_CUBIC)

    # --- Вырезаем область ---
    x_start = int(cx - w/2)
    y_start = int(cy - h/2)
    x_end = int(cx + w/2)
    y_end = int(cy + h/2)

    # Проверяем границы
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(img.shape[1], x_end)
    y_end = min(img.shape[0], y_end)

    cropped = rotated[y_start:y_end, x_start:x_end]

    # --- Визуализация (опционально) ---
    if debug:
        debug_img = img.copy()
        box = cv.boxPoints(best_rect)
        box = np.intp(box)
        cv.drawContours(debug_img, [box], -1, (0, 255, 0), 2)
        cv.imshow("Detected Plate", debug_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return cropped, best_rect

img_count = 9


for i in range(1, img_count + 1):
    path = f"car_plates_img/car_p_{i}.jpg"
    print("Путь:", repr(path))
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, th_img = cv.threshold(img, 105, 255, cv.THRESH_BINARY)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    cropped_img, rect = find_plate_2(img, th_img, True)
    if(cropped_img is None or rect is None):
        continue

    # ax1.imshow(img, cmap="gray")
    # ax1.set_title(f'Исходное изображение {i}')
    # ax1.axis('off')

    # ax2.imshow(cropped_img, cmap="gray")
    # ax2.set_title(f'Вырезанная часть {i}')
    # ax2.axis('off')

    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    # plt.tight_layout()
    # plt.show()