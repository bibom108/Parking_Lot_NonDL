from SolutionHeader import *

#"./dataset/highView/Block/main.jpg"
img_path = input("Choose your path: ")
solution_number = input("Choose your solution: ")
configSol4 = configparser.ConfigParser()        # String type
configSol4.read('configSol4.ini')
configSol3 = configparser.ConfigParser()        # String type
configSol3.read('configSol3.ini')
img_name = img_path[-6:-4]
if img_name[0] == '/':
    img_name = img_name[-1:]
iteration = configSol4[img_name]['iter']
bright = configSol4[img_name]['bright']
contrast = configSol4[img_name]['contrast']
k = configSol3[img_name]['k']

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
    if "_1" in img_path:
        dila = dilation(img_threshold, "1")
    else:
        dila = dilation(img_threshold, "0")
    # Filter the noise
    if "1_" in img_path:
        opening = morphological(dila, "1")
    else:
        opening = morphological(dila, "0")
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
    img_final = felzenszwalbs(img, int(k))
    show_img(img_final, "Felzenszwalbs method's Result")
    cv.waitKey(0)
    cv.destroyAllWindows()

elif solution_number == "4":
    # Get the image before inscreasing bright and contrast
    originImg = img.copy()
    if bright != "-1" and contrast != "-1":
        # Inscrease bright and contrast of img to reduce blurry
        img = funcBrightContrast(img, int(bright), int(contrast))

    # Morphological ACWE
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = img_as_float(img_gray)
    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, int(iteration), init_level_set=init_ls, smoothing=3, iter_callback=callback)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes.flatten()
    ax[0].imshow(originImg, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='g')
    ax[0].set_title("Morphological Snakes's Result", fontsize=12)
    fig.tight_layout()
    plt.show()