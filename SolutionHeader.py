import cv2 as cv
import numpy as np
import operator
import skimage.segmentation
from skimage.segmentation import mark_boundaries
import operator

import cv2 as cv
import numpy as np
import skimage.segmentation
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set


ref_points = []
cur_point = []
kernel = np.ones((5, 5), np.uint8)
num_pt = 0


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


def catch_point(event, x, y, flags, param):
    global ref_points
    global cur_point
    global num_pt
    if event == cv.EVENT_LBUTTONDOWN:
        cur_point.append(x)
        cur_point.append(y)
    elif event == cv.EVENT_LBUTTONUP:
        ref_points.append(cur_point)
        cv.circle(param, (cur_point[0], cur_point[1]), 10, (0, 0, 255), 4)
        cur_point = []
        num_pt += 1


def getNumPoint():
    global num_pt
    return num_pt

def show_img(img, title):
    x = cv.resize(img, (960, 540))
    cv.imshow(title, x)


def draw_lines(image, lines, color=[0, 255, 0], thickness=3, make_copy=True):  # vẽ đường thẳng từ 2 điểm đã cho
    if make_copy:
        image = np.copy(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_for_user(line1, line2, image, color, index,
                  thick):  # xuất output (vùng xanh cho ô trống, cạnh đỏ cho chỗ có xe)
    cv.rectangle(image, (min(line1[0], line1[2]), max(line1[1], line2[1])),
                 (min(line2[0], line2[2]), min(line1[3], line2[3])), color, thick)
    return image


def remove_background(img):
    # img = cv.GaussianBlur(src=img, ksize=(5, 5), sigmaX=0)
    image = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    # white color mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv.inRange(image, lower, upper)
    # combine the mask
    mask = cv.bitwise_or(white_mask, yellow_mask)
    return mask


def morphological(img, noise_filter):
    if noise_filter == "1":
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    else:
        return img


# FELZENSZWALD'S METHOD

def nothing(x):
    pass


def felzenszwalbs(img, numScale):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    res2 = skimage.segmentation.felzenszwalb(lab, scale=numScale)
    return mark_boundaries(img, res2)


def dilation(img, faded_line):
    if faded_line == "1":
        return cv.dilate(img, kernel, iterations=1)
    else:
        return img


def detect_edges(image, low_threshold=50, high_threshold=80):  # phát hiện cạnh bằng Canny
    return cv.Canny(image, low_threshold, high_threshold)


def select_region(image, vertices):
    vertices = np.array(vertices)
    vertices_new = np.array([vertices], dtype=np.int32)
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv.fillPoly(mask, vertices_new, 255)
    return cv.bitwise_and(image, mask)


def hough_lines(image, theta):  # trả về tập hợp các cạnh phát hiện được
    return cv.HoughLinesP(image, 1, theta=theta, threshold=30, minLineLength=30, maxLineGap=20)


def remove_noise(img, noise_filter):
    thresh = remove_background(img)
    # noise removal
    opening = morphological(thresh, noise_filter)
    opening = cv.bitwise_not(opening)
    return opening


def find_foreground_area(opening):
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,3)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    return sure_fg


def marker(sure_fg, unknown, img, remarker):
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    if(remarker==0):
        markers[unknown==0] = -1
    return markers


# MORPHOLOGICAL SNAKES
def load(f, as_gray=False):
    """Load an image file located in the data directory.
    Parameters
    ----------
    f : string
        File name.
    as_gray : bool, optional
        Whether to convert the image to grayscale.
    Returns
    -------
    img : ndarray
        Image loaded from ``skimage.data_dir``.
    """
    # importing io is quite slow since it scans all the backends
    # we lazy import it here
    from skimage.io import imread
    return imread(f, as_gray=as_gray)


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


#######################################EXTEND PART#####################################################


def filter_lines(lines):  # Vạch kẻ đường là đường thẳng
    cleaned = []
    clusters = {}

    for line in lines:  # Chỉ lấy vạch kẻ thẳng đứng, bỏ các vạch kẻ nằm ngang trong tập cạnh
        for x1, y1, x2, y2 in line:
            cleaned.append((x1, y1, x2, y2))

    cleaned = sorted(cleaned, key=operator.itemgetter(1))  # sort theo 0y để tiến hành gom cụm

    k = 1
    num = -1
    while k < len(cleaned):  # gom cụm theo hàng ngang
        check = False
        for j in range(len(clusters)):
            if cleaned[k][1] + 10 >= clusters[j][0][1] >= cleaned[k][1] - 10:
                clusters[j].append(cleaned[k])
                check = True
        if not check:
            num = num + 1
            clusters[num] = []
            clusters[num].append(cleaned[k])
        k = k + 1

    # for i in range(len(clusters)):  # loại bỏ line trùng do thickness của line lớn
    #     clusters[i] = sorted(clusters[i], key=operator.itemgetter(0))
    #     j = 1
    #     while j < len(clusters[i]):
    #         if len(clusters[i]) == 2:
    #             break
    #         if abs(clusters[i][j][0] - clusters[i][j - 1][0]) < 6:
    #             clusters[i].pop(j)
    #         j = j + 1

    return clusters


def check_object_in_slot(line1, line2, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    pt_1 = [min(line1[0], line1[2]), max(line1[1], line2[1])]  # Từ 2 đoạn thẳng, lấy ra 4 điểm để xác định vùng đỗ xe
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
    if len(contours) <= 4:  # giá trị có thể thay đổi tùy từng hình ??????????
        return True
    else:
        return False

# ----------------------------------------------
# Bright and Contrast for solution 4

def funcBrightContrast(img,bright,contrast):
    effect = apply_brightness_contrast(img,bright,contrast)
    return  effect

def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)