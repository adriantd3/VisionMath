import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import clasiffication

debug = False


def gaussian_bell1D(x, sigma):
    base = 1 / (sigma * np.sqrt(2 * np.pi))
    exp = np.exp(-(x * x) / (2 * (sigma * sigma)))

    return base * exp


def gaussian_filter(image, w_kernel, sigma):
    # Create kernel using associative property
    s = sigma
    w = w_kernel
    kernel_1D = np.float32([gaussian_bell1D(z, s) for z in range(-w, w + 1)])  # Evaluate the gaussian in "expression"
    vertical_kernel = kernel_1D.reshape(2 * w + 1, 1)  # Reshape it as a matrix with just one column
    horizontal_kernel = kernel_1D.reshape(1, 2 * w + 1)  # Reshape it as a matrix with just one row
    kernel = signal.convolve2d(vertical_kernel, horizontal_kernel)  # Get the 2D kernel

    # Convolve image and kernel
    smoothed_img = cv2.filter2D(image, cv2.CV_8U, kernel)
    return smoothed_img


def binarize_kmeans(image, it):
    # Set random seed for centroids
    cv2.setRNGSeed(124)

    # Flatten image
    flattened_img = image.reshape((-1, 1))
    flattened_img = np.float32(flattened_img)

    # Set epsilon
    epsilon = 0.2

    # Establish stopping criteria (either `it` iterations or moving less than `epsilon`)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, it, epsilon)

    # Set K parameter (2 for thresholding)
    K = 2

    # Call kmeans using random initial position for centroids
    _, label, center = cv2.kmeans(flattened_img, K, None, criteria, it, cv2.KMEANS_RANDOM_CENTERS)

    # Colour resultant labels
    center = np.uint8(center)  # Get center coordinates as unsigned integers
    center[0] = 0
    center[1] = 255
    flattened_img = center[label.flatten()]  # Get the color (center) assigned to each pixel

    # Reshape vector image to original shape
    binarized = flattened_img.reshape(image.shape)

    return binarized


def non_maxima_suppression_contours(contours):
    n = len(contours)
    res_contours = []
    for (x, y, w, h) in contours:
        if w < 30:
            continue
        res_contours.append((x, y, w, h))
    return res_contours


def draw_contours_lr_order(binary_image, contours_sorted):
    # Draws the bounding rect and order of contours of the input image.

    image_drawn = np.copy(binary_image)
    image_drawn = cv2.cvtColor(image_drawn, cv2.COLOR_GRAY2RGB)

    count = 1
    for x, y, w, h in contours_sorted:
        image_drawn = cv2.putText(image_drawn, str(count), (x - 2, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=2, color=(0, 0, 255), thickness=4)
        image_drawn = cv2.rectangle(image_drawn, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 3)
        count += 1

    return image_drawn


def category_string_conversion(category):
    # returns the string that corresponds to the given category
    if category == 10:
        return '+'
    elif category == 11:
        return '-'
    elif category == 12:
        return '*'
    else:
        return str(category)


# Classifies the input data and returns the resultant string
def generate_string(data):
    global debug
    n_elements = data[0, 0, :].shape[0]
    res_string = ''

    n_cols_plot = int(n_elements / 5)
    if n_elements % 5 != 0:
        n_cols_plot += 1

    categories = []

    for i in range(n_elements):
        # Classify each image
        category = clasiffication.classify_input_model(data[:, :, i])
        # Insert category in categories array
        categories.append(category)
        prediction = category_string_conversion(category)

        if debug:
            plt.subplot(n_cols_plot, 5, i + 1)
            plt.title(prediction)
            plt.imshow(data[:, :, i], cmap='grey')

        res_string += prediction

    if debug:
        plt.show()

    # Convert to numpy array
    categories = np.array(categories)

    return res_string, categories


def identify_div(categories, contours):
    # Identifies the div

    # We must distinguish when a '-' means subtraction or division
    # print(categories)
    total_lines = np.where(categories == 11)[0]
    # print(total_lines)

    div_index = []
    div_ranges = []

    # Identify the real division symbols
    for index in total_lines:
        [x_index, y_index, w_index, h_index] = contours[index]

        # Contours list that surround the symbol in the x-axis
        horizontal_cont_list = np.array([[x, y, w, h] for x, y, w, h in contours if
                                         ((x < x_index < (x + w)) or (x_index < x < (x_index + w_index)))
                                         and (x != x_index and y != y_index)])

        # If there is no contour around, it cannot be a div symbol, so we skip this iteration
        if horizontal_cont_list.shape[0] == 0:
            continue

        # Array formed by the absolute distances between the symbol and the index number
        abs_distance_array = np.array([abs((y_index + h_index) - (y + h)) for _, y, _, h in horizontal_cont_list])
        # Calculate the index of the minimum absolute distance in the original contour list
        min_abs_distance = np.where(np.all(contours == horizontal_cont_list[np.argmin(abs_distance_array)], axis=1))[0][
            0]

        isDiv = False
        if categories[min_abs_distance] != 11:
            # If the min_abs_distance category is not a "-" it means that it is a division
            isDiv = True

        if isDiv:
            div_index.append(index)
            # We take the range of indexes that covers the div symbol
            min_index = np.where(np.all(contours == horizontal_cont_list[0], axis=1))[0][0]
            max_index = np.where(np.all(contours == horizontal_cont_list[-1], axis=1))[0][0]

            div_ranges.append([min_index, max_index])

            # print(f"Div Detected: index = {index} --- min_index = {min_index} --- max_index = {max_index}")

    # Convert to numpy arrays
    div_index = np.array(div_index)
    div_ranges = np.array(div_ranges)

    return div_index, div_ranges


def generate_expression(categories, contours):
    # Generates the final expression given:
    # categories: list that contains the corresponding category to each contour
    # contours: x, y, w, h for each contour in the image

    div_index, div_ranges = identify_div(categories, contours)

    expression = ""

    upper_string = ""
    lower_string = ""

    # Generate the final string that will be evaluated
    for i in range(0, categories.shape[0]):
        # If it is a division symbol, continue
        if np.isin(i, div_index):
            continue

        inRange = False
        isLastIndex = False
        range_div = 0
        while not inRange and range_div < div_ranges.shape[0]:
            # Determine if the index is in any range
            min_index, max_index = div_ranges[range_div]
            if min_index <= i <= max_index:
                inRange = True
                if i == max_index:
                    isLastIndex = True
            else:
                range_div += 1
        # print(f"FOR INDEX = {i} -> (inRange={inRange},isLastIndex={isLastIndex},range_div={range_div})")
        if inRange:
            if contours[i, 1] < contours[div_index[range_div], 1]:
                # If the contour is above the line
                upper_string += category_string_conversion(categories[i])
            else:
                # If the contour is below the line
                lower_string += category_string_conversion(categories[i])
            if isLastIndex:
                # If it is the last element of the range, combine all elements
                expression += "((" + upper_string + ")/(" + lower_string + "))"

                # Reset the strings
                upper_string = ""
                lower_string = ""
        else:
            # If it is not part of any range, just add the number to the string
            expression += category_string_conversion(categories[i])

    return expression


def image_processing(image):
    image = cv2.bitwise_not(image)
    smoothed_image = gaussian_filter(image, 4, 1)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(48, 48))
    enhanced_smoothed = clahe.apply(smoothed_image)

    # enhanced_smoothed = cv2.equalizeHist(smoothed_image)

    _, thresh_img = cv2.threshold(enhanced_smoothed, 170, 255, cv2.THRESH_BINARY)

    if debug:
        plt.subplot(311)
        plt.title("Original Image")
        plt.imshow(image, cmap='grey')

        plt.subplot(312)
        plt.title("Smoothed Image (Gaussian Filter)")
        plt.imshow(smoothed_image, cmap='grey')

        plt.subplot(313)
        plt.title("Enhanced Image (CLAHE)")
        plt.imshow(enhanced_smoothed, cmap='grey')

        plt.show()

    return thresh_img


def contour_extraction_sorting(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a list that contains the x, y, w, h of each contour
    contours_ordered = [cv2.boundingRect(cnt) for cnt in contours]

    # Clear out minor contours produced by noise
    contours_ordered = non_maxima_suppression_contours(contours_ordered)
    contours_ordered = np.array(contours_ordered)

    # print(contours_ordered)
    # print("---------------------------------")

    # Sort the contours from LEFT to RIGHT
    contours_ordered = contours_ordered[np.argsort(contours_ordered[:, 0], kind='mergesort')]

    # print(contours_ordered)

    if debug:
        plt.subplot(2, 1, 1)
        plt.title("Binary Image")
        plt.imshow(image, cmap='grey')

        plt.subplot(2, 1, 2)
        plt.title("Left to Right - Ordered Contours")
        plt.imshow(draw_contours_lr_order(image, contours_ordered))
        plt.show()

    # This array contains the 28x28 images of all the contours in the image
    data = np.zeros((clasiffication.rSize, clasiffication.cSize, len(contours_ordered)))
    count = 0
    for x, y, w, h in contours_ordered:
        # Generates de 28x28 from the 2D array(Bounding Rect of each contour)
        data[:, :, count] = clasiffication.fit_contour(image[y:y + h, x:x + w])
        count += 1

    return data, contours_ordered


def main(imagePath, showImages):
    global debug
    debug = showImages

    try:
        image = cv2.imread("images/" + imagePath, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        print("Image not Found")
        return

    processed_image = image_processing(image)
    data, contours_ordered = contour_extraction_sorting(processed_image)

    print("\n-------------PREDICTION STAGE-----------------")
    res_string, categories = generate_string(data)
    print(f"MODEL PREDICTION: {res_string}\n")

    # Now that we have contours classified we can generate the mathematical expression

    final_expression = generate_expression(categories, contours_ordered)
    print(f"EXPRESSION: {final_expression}")
    try:
        print(f"RESULT: {eval(final_expression)}")
    except Exception:
        print("ERROR: Invalid operation")

