import cv2
import numpy as np


def check_point(x_center, y_center, rec_list):
    for x, y, w, h in rec_list:
        if x <= x_center <= x + w and y <= y_center <= y + h:
            return True
    return False


img = cv2.imread('1.png')
bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
black_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

count = 0
extend = 1
threshold_match = 0.7
spot_list = []
for cnt in cnts:
    color = list(np.random.random(size=3) * 256)
    img_copy = img.copy()
    x, y, w, h = cv2.boundingRect(cnt)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.018 * perimeter, True)
    if len(approx) == 4:
        if w * h > 2000:
            continue
        if w * h < 50:
            continue
        x, y, w, h = x + extend, y + extend, w - extend * 2, h - extend * 2
        x_center, y_center = x + w // 2, y + h // 2
        if check_point(x_center, y_center, spot_list):
            continue
        crop = img[y:y + h, x:x + w]
        result = cv2.matchTemplate(img, crop, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold_match)
        locations = list(zip(*locations[::-1]))
        count += 1
        print(count)
        if locations:
            for loc in locations:
                x, y, w, h = loc[0], loc[1], w, h
                spot_list.append([x, y, w, h])
                # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(black_image, str(count), (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.imshow("img", img)
        # cv2.waitKey()


cv2.imshow('black_image', black_image)
# cv2.imshow('thresh', thresh)
cv2.imshow('img', img)
# cv2.imshow('img', cv2.resize(img, (0, 0), fx=2, fy=2))
cv2.waitKey(0)
cv2.destroyAllWindows()
