from SolutionHeader import *

img_path = "20.jpeg"
line_type = "1" #input("The line is straight or non-straight? [0: straight, 1: non-straight] ")
slot_type = "0" #input("Each slot is separated or continuous ? [0: separated, 1: continuous] ")
color_type = "0" #input("What kind of color is the line ? [0: white, 1: yellow] ")
noise_filter = "1" #input("Do you want to reduce noise on this image (Many object same color with the line) ? [0: NO, 1: YES] ")
faded_line = "1" #input("Is the line blur or clear ? [0: Clear, 1: Blur] ")
solution_number = "1" #input("Which solution do you refer (Try them out to pick the best) ? [1: Morphological, 2:] ")

img = cv.imread(img_path)
clone = img.copy()
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
if input("Do you want to select ROI? [0: NO, 1: YES] ") == "1":
    print("Type 's' when you done!")
    param = img
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setMouseCallback("image", catch_point, param)
    while True:
        cv.imshow("image", img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("s"):
            cv.destroyAllWindows()
            break
    img = clone
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = select_region(img_gray, ref_points)
    img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

# img is ready to go
if solution_number == "1":
    img_threshold = remove_background(img)

    # Fill the line
    dila = dilation(img_threshold, faded_line)

    # Filter the noise
    opening = morphological(dila, noise_filter)

    # Detect edge
    # img_edge_detection = detect_edges(opening)

    if line_type == "0":
        lines = hough_lines(opening, np.pi / 2)
    else:
        lines = hough_lines(opening, np.pi / 180)

    img_final = draw_lines(img, lines)

    show_img(img_final)
    cv.waitKey(0)
    cv.destroyAllWindows()