import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import sys
import argparse
import operator


def show_img(img):
    x = cv.resize(img, (960, 540))
    cv.imshow('', x)


def draw_lines(image, lines, color=[0, 0, 255], thickness=2, make_copy=True):           # vẽ đường thẳng từ 2 điểm đã cho
    if make_copy:
        image = np.copy(image)
    for line in lines:
        cv.line(image, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return image


def draw_for_user(line1, line2, image, color, index, thick):                            # xuất output (vùng xanh cho ô trống, cạnh đỏ cho chỗ có xe)
    cv.rectangle(image, (min(line1[0], line1[2]), max(line1[1], line2[1])), (min(line2[0], line2[2]), min(line1[3], line2[3])), color, thick)
    return image


def remove_background(img):                                                             # làm mờ sau đó sử dụng ngưỡng động otsu
    img_gray_blur = cv.GaussianBlur(src=img, ksize=(5, 5), sigmaX=0)
    ret2, th2 = cv.threshold(img_gray_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th2


def detect_edges(image, low_threshold=50, high_threshold=100):                          # phát hiện cạnh bằng Canny
    return cv.Canny(image, low_threshold, high_threshold)


def select_region(image):                                                               # dành cho ảnh có vùng bãi đỗ xe không phải là phần chính
    rows, cols = image.shape[:2]
    # print(image.shape[:2])

    pt_1 = [cols * 0.13, rows * 0.24]
    pt_2 = [cols * 0.15, rows * 0.84]
    pt_3 = [cols, rows * 0.84]
    pt_4 = [cols, rows * 0.24]
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)
    point_img = image.copy()
    point_img = cv.cvtColor(point_img, cv.COLOR_GRAY2RGB)
    for point in vertices[0]:
        cv.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
    # show_img(point_img)
    return filter_region(image, vertices)


def filter_region(image, vertices):                                                         # Lấy vùng từ ảnh
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv.fillPoly(mask, vertices, 255)
    return cv.bitwise_and(image, mask)


def hough_lines(image):                                                                     # trả về tập hợp các cạnh phát hiện được
    # The input image needs to be the result of edge detection!!!
    # Minlinelengh (the shortest length of a line is ignored) and MaxLineCap (the maximum interval between two lines, which is less than this value, is considered as a line)
    # rho distance precision, theta angle precision (the smaller the two precision, the more accurate), threshod exceeds the set threshold before the line segment is detected
    return cv.HoughLinesP(image, rho=1, theta=np.pi/1, threshold=0, minLineLength=10, maxLineGap=10)


def filter_lines(lines):                     # Vạch kẻ đường là đường thẳng
    cleaned = []
    clusters = {}

    for line in lines:                       # Chỉ lấy vạch kẻ thẳng đứng, bỏ các vạch kẻ nằm ngang trong tập cạnh
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < 5:
                cleaned.append((x1, y1, x2, y2))

    cleaned = sorted(cleaned, key=operator.itemgetter(1))       # sort theo 0y để tiến hành gom cụm

    k = 1
    num = -1
    while k < len(cleaned):                                     # gom cụm theo hàng ngang
        check = False
        for j in range(len(clusters)):
            if cleaned[k][1]+10 >= clusters[j][0][1] >= cleaned[k][1]-10:
                clusters[j].append(cleaned[k])
                check = True
        if not check:
            num = num + 1
            clusters[num] = []
            clusters[num].append(cleaned[k])
            
        k = k + 1

    for i in range(len(clusters)):                              # loại bỏ line trùng do thickness của line lớn
        clusters[i] = sorted(clusters[i], key=operator.itemgetter(0))
        j = 1
        while j < len(clusters[i]):
            if len(clusters[i]) == 2:
                break
            if abs(clusters[i][j][0] - clusters[i][j-1][0]) < 6:
                clusters[i].pop(j)
            j = j + 1

    return clusters


def check_object_in_slot(line1, line2, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    pt_1 = [min(line1[0], line1[2]), max(line1[1], line2[1])]                           # Từ 2 đoạn thẳng, lấy ra 4 điểm để xác định vùng đỗ xe
    pt_2 = [min(line2[0], line2[2]), max(line1[1], line2[1])]
    pt_3 = [min(line2[0], line2[2]), min(line1[3], line2[3])]
    pt_4 = [min(line1[0], line1[2]), min(line1[3], line2[3])]
    vertices_new = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)
    mask = np.zeros_like(img)
    if len(mask.shape) == 2:
        cv.fillPoly(mask, vertices_new, 255)
    img_selected = cv.bitwise_and(img, mask)
    img_no_bg = remove_background(img_selected)
    contours, hierarchy = cv.findContours(img_no_bg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # print(len(contours))
    if len(contours) <= 4:                                                              # giá trị có thể thay đổi tùy từng hình
        return True
    else:
        return False


img = cv.imread("./parking_lot/9.png")
img_gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
img_threshold = remove_background(img_gray)
img_edge_detection = detect_edges(img_threshold)
# img_selected = select_region(img_edge_detection)              # use when work with ROI

lines = hough_lines(img_edge_detection)
clusters = filter_lines(lines)                                       # cụm đã được gom (1 cụm là 1 hàng ngang)
# img_lines = draw_lines(img, clusters)                              # vẽ thử gom cụm

cnt_empty = 0
parking_lot_num = 0
overlay = np.copy(img)
for i in range(len(clusters)):
    j = 1
    while j < len(clusters[i]):
        if check_object_in_slot(clusters[i][j - 1], clusters[i][j], overlay):  # true is empty slot, draw green rec
            overlay = draw_for_user(clusters[i][j - 1], clusters[i][j], overlay, [0, 100, 0], i, -1)
            cnt_empty = cnt_empty + 1
        else:
            overlay = draw_for_user(clusters[i][j - 1], clusters[i][j], overlay, [0, 0, 255], i, 5)  # false if occupied
        j = j + 2
        parking_lot_num = parking_lot_num + 1

cv.addWeighted(overlay, 0.5, img, 1 - 0.5, 0, img)                  # output mờ hơn so với ảnh thật
print("Available: %d spots" % cnt_empty)
print("Total: %d spots" % parking_lot_num)

show_img(img)
cv.waitKey(0)
cv.destroyAllWindows()
