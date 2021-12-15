from SolutionHeader import *

#"./dataset/highView/Block/main.jpg"
img_path = input("Enter file dir: ")
blur_img = input("Is the picture blurry or sharp?[0: Blur, 1:Sharp]")
noise_filter = input("Does many object have same color with the line (or noise) ? [0: NO, 1: YES] ")
faded_line = input("Is the line blur or clear ? [0: Clear, 1: Blur] ")
line_size = input("How big is the line from 0 (super small) to 9 (super large)? ")
line_number = input("Does the image have many slots? [0: Many, 1: A few] ")
solution_number = input("Which solution do you refer (Try them out to pick the best) ? [1: Morphological, 2: Watershed, 3: Felzenszwalbs, 4: Morphological Snakes] ")

img = cv.imread(img_path)
clone = img.copy()
if input("Do you want to select ROI? [0: NO, 1: YES] ") == "1":
    param = img
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setMouseCallback("image", catch_point, param)
    while True:
        cv.imshow("image", img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("s") or getNumPoint() == 2:
            cv.destroyAllWindows()
            break
    img = clone
    img = img[ref_points[0][1]:ref_points[1][1], ref_points[0][0]:ref_points[1][0]]

# img is ready to go
if solution_number == "1":
    img_threshold = remove_background(img)
    # Fill the line
    dila = dilation(img_threshold, faded_line)
    # Filter the noise
    opening = morphological(dila, noise_filter)
    # Detect edge
    # img_edge_detection = detect_edges(opening)
    lines = hough_lines(opening, np.pi / 180)
    img_final = draw_lines(img, lines)
    show_img(img_final, "Morphological Result")
    cv.waitKey(0)
    cv.destroyAllWindows()

elif solution_number == "2":
    opening = remove_noise(img)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    sure_fg = find_foreground_area(opening)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    clone = img.copy()
    markers = marker(sure_fg, unknown, clone, 0)
    clone[markers == -1] = [0, 255, 0]
    avg_color_per_row = np.average(clone, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if(avg_color[1]==255.0):
        markers = marker(sure_fg, unknown, img, 1)
        img[markers == -1] = [0, 255, 0]
    else:
        img = clone
    show_img(img, "Watershed algorithm's Result")
    cv.waitKey(0)
    cv.destroyAllWindows()

elif solution_number == "3":
    para = 500*int(line_size) + 500
    img_final = felzenszwalbs(img, para)
    show_img(img_final, "Felzenszwalbs method's Result")
    cv.waitKey(0)
    cv.destroyAllWindows()

elif solution_number == "4":
    # Get the image before inscreasing bright and contrast
    originImg = img.copy()
    if blur_img == "0":
        # Inscrease bright and contrast of img to reduce blurry
        img = funcBrightContrast(img, 200, 300)
        if noise_filter=="0":
            iteration = 1
        else:
            iteration = 3
    else:
        if (line_number == "0"):
            if noise_filter == "0":
                iteration = 4
            else:
                iteration = 8
        else:
            if noise_filter == "0":
                iteration = 23
            else:
                iteration = 68
    # Morphological ACWE
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = img_as_float(img_gray)
    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, iterations = iteration, init_level_set=init_ls, smoothing=3, iter_callback=callback)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes.flatten()
    ax[0].imshow(originImg, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='g')
    ax[0].set_title("Morphological Snakes's Result", fontsize=12)
    fig.tight_layout()
    plt.show()