import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import sys
import argparse

order_point = 0
ref_points = []
cur_point = []
parking_lot_num = 0


def draw_for_user(image, color, index, thick):
    global ref_points

    cv.rectangle(image, (ref_points[index][0], ref_points[index][1]), (ref_points[index][2], ref_points[index][3]), color, thick)
    return image


def catch_point(event, x, y, flags, param):
    global order_point
    global ref_points
    global cur_point
    global parking_lot_num

    if event == cv.EVENT_LBUTTONDOWN:
        cur_point.append(x)
        cur_point.append(y)
    elif event == cv.EVENT_LBUTTONUP:
        if order_point == 1:
            ref_points.append(cur_point)
            parking_lot_num = parking_lot_num + 1
            cur_point = []
            draw_for_user(image, [255, 0, 0], parking_lot_num-1, 5)
        order_point = (order_point + 1) % 2


def filter_region(image, vertices):
    pt_1 = [vertices[0], vertices[1]]
    pt_2 = [vertices[2], vertices[1]]
    pt_3 = [vertices[2], vertices[3]]
    pt_4 = [vertices[0], vertices[3]]
    vertices_new = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv.fillPoly(mask, vertices_new, 255)
    return cv.bitwise_and(image, mask)


def remove_background(img):
    __, res = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    return res


def img_process(rec, img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_selected = filter_region(img_gray, rec)
    img_no_bg = remove_background(img_selected)
    contours, hierarchy = cv.findContours(img_no_bg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    print(len(contours))
    if len(contours) <= 20:
        return True
    else:
        return False
    # cv.waitKey()


img = cv.imread('./parking_lot/2.jpg')
image = img.copy()
clone = img.copy()
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", catch_point)

while True:
    cv.imshow("image", image)
    key = cv.waitKey(1) & 0xFF
    if key == ord("r"):
        image = clone.copy()
    elif key == ord("d"):
        break

index = 0
cnt_empty = 0
overlay = np.copy(img)
for rec in ref_points:
    if (img_process(rec, img) == True):  # true is empty slot, draw green rec
        overlay = draw_for_user(overlay, [0, 100, 0], index, -1)
        cnt_empty = cnt_empty + 1
    else:
        overlay = draw_for_user(overlay, [0, 0, 255], index, 5)  # false if occupied
    index = index + 1

cv.addWeighted(overlay, 0.5, img, 1 - 0.5, 0, img)

cv.putText(img, "Available: %d spots" % cnt_empty, (30, 95),
            cv.FONT_HERSHEY_SIMPLEX,
            2, (255, 255, 255), 7)

cv.putText(img, "Total: %d spots" % parking_lot_num, (30, 200),
            cv.FONT_HERSHEY_SIMPLEX,
            2, (255, 255, 255), 7)
cv.namedWindow("output", cv.WINDOW_NORMAL)
cv.imshow("output", img)
cv.waitKey()
