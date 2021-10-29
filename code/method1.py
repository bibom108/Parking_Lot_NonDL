import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import sys
import argparse


def show_img(img):
    x = cv.resize(img, (960, 540))  # Resize image
    cv.imshow('', x)


def remove_background(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    sensitivity = 71
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    mask = cv.inRange(hsv, lower_white, upper_white)
    res = cv.bitwise_and(img, img, mask=mask)
    return res


def detect_edges(image, low_threshold=50, high_threshold=100):
    return cv.Canny(image, low_threshold, high_threshold)


def select_region(image):
    rows, cols = image.shape[:2]
    print(image.shape[:2])
    # We manually create a few points that roughly enclose the outline of the parking lot
    pt_1 = [cols * 0.13, rows * 0.24]
    pt_2 = [cols * 0.15, rows * 0.84]
    pt_3 = [cols, rows * 0.84]
    pt_4 = [cols, rows * 0.24]
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)
    point_img = image.copy()
    point_img = cv.cvtColor(point_img, cv.COLOR_GRAY2RGB)
    # It's better to draw lines on the image than on the grayscale image, so we convert it into RGB color image (operate on the copied image)
    for point in vertices[0]:
        cv.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        # Drawing function, need to pass in the original image, coordinate points (tuple format), --, color, thickness
        # Filter is to filter out the framed part and pass in the image and fixed point
    # show_img(point_img)
    return filter_region(image, vertices)


def filter_region(image, vertices):  # Extract what I need: the outline of the parking lot
    # Image is the original image and vertices is the incoming vertex
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv.fillPoly(mask, vertices, 255)  # Based on the incoming vertices, it forms an area, and fills all with 255 (white)
    return cv.bitwise_and(image, mask)


def hough_lines(image):
    # The input image needs to be the result of edge detection!!!
    # Minlinelengh (the shortest length of a line is ignored) and MaxLineCap (the maximum interval between two lines, which is less than this value, is considered as a line)
    # rho distance precision, theta angle precision (the smaller the two precision, the more accurate), threshod exceeds the set threshold before the line segment is detected
    # It used to be HoughLines(), but it's faster to become houghlines P
    return cv.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold=10, minLineLength=0, maxLineGap=10)


def draw_lines(image, lines, color=[0, 0, 255], thickness=5, make_copy=True):
    # The line segments detected by Hoff are filtered and drawn in the original image
    if make_copy:
        image = np.copy(image)
    cleaned = []  # Filter the line segment and draw it in the image. cleaned is the filtered line segment
    for line in lines:
        for x1, y1, x2, y2 in line:  # A line is made up of (x1,y1) and (x2,y2)
            # if abs(y2 - y1) >= 300:
                # (y2-y1) to measure the inclination, the parking line must be a straight line or a line with small inclination
                # (x2-x1) measures the horizontal distance. If the horizontal distance is too long or too short, it is not a stop line
                cleaned.append((x1, y1, x2, y2))
                cv.line(image, (x1, y1), (x2, y2), color, thickness)
    print("The total number of segments is:", len(cleaned))
    return image

img = cv.imread("./parking_lot/4.png")
img_gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
img_gray_blur = cv.GaussianBlur(src=img_gray, ksize=(5, 5), sigmaX=0)
img_edge = cv.Canny(image=img_gray_blur, threshold1=50, threshold2=150)
# img_threshold = remove_background(img)

# img_edge_detection = detect_edges(img_threshold)

# img_selected = select_region(img_edge_detection)

lines = hough_lines(img_edge)
img_with_lines = draw_lines(img, lines)

show_img(img_edge)
# show_img(img_with_lines)

cv.waitKey(0)
cv.destroyAllWindows()
