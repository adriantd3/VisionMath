import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import clasiffication
import math

debug = True

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

    # Estabish stopping criteria (either `it` iterations or moving less than `epsilon`)
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
    binarized = flattened_img.reshape((image.shape))

    return binarized


def non_maxima_supression_contours(contours):
    n = len(contours)
    res_contours = []
    for (x, y, w, h) in contours_ordered:
        if w < 30:
            continue
        res_contours.append((x, y, w, h))
    return res_contours


# Draws the bounding rect and order of contours of the input image.
def draw_contours(binary_image, contours_sorted):
    image_drawn = np.copy(binary_image)
    image_drawn = cv2.cvtColor(image_drawn, cv2.COLOR_GRAY2RGB)

    count = 1
    for x, y, w, h in contours_ordered:
        image_drawn = cv2.putText(image_drawn, str(count), (x - 2, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=2, color=(0, 0, 255), thickness=4)
        image_drawn = cv2.rectangle(image_drawn, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 3)
        count += 1

    return image_drawn

def category_toString_conversion(category):
    #returns the string that corresponds to the given category
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
    count = 0

    categories = []

    for i in range(n_elements):
        # Classify each image
        # category = clasiffication.classify_input_weights(data[:,:,i])
        category = clasiffication.classify_input_model(data[:, :, i],debug)
        # Insert category in categories array
        categories.append(category)
        # plt.imshow(data[:,:,i],cmap='grey')
        # plt.show()
        # cv2.imwrite('../../images/datasets/prueba_' + str(count) + '.jpg', data[:, :, i])
        # count += 1

        res_string += category_toString_conversion(category)

    # Convert to numpy array
    categories = np.array(categories)

    return res_string, categories


def generate_expression(categories, contours):
    # We must distinguish when a '-' means subtraction or division
    print(categories)
    total_lines = np.where(categories == 11)[0]
    print(total_lines)
    avg_height = average_height(categories, contours)

    div_index = []
    div_ranges = []

    expression = ""

    # Identify the real division symbols
    for index in total_lines:
        [x_index, y_index, w_index, h_index] = contours[index]

        # Contours list that surround the symbol in the x axis
        horizontal_cont_list = [[x, y, w, h] for x, y, w, h in contours if
                                ((x < x_index < (x + w)) or (x_index < x < (x_index + w_index)))
                                and (x != x_index and y != y_index)]

        horizontal_cont_list = np.array(horizontal_cont_list)

        #Does not have anything
        if horizontal_cont_list.shape[0] == 0:
            continue

        # Vertically sorted array BOTTOM - UP
        print(f"horizontal_cont_list shape = {horizontal_cont_list.shape}")
        vertical_const_list = horizontal_cont_list[np.argsort(horizontal_cont_list[:, 1], kind='mergesort')]

        print(f"VERTICAL CONT LIST FOR INDEX({index})")
        print(vertical_const_list)

        print(f"INDEX INFO: x_index={x_index} -- y_index={y_index} --- w_index={w_index} --- h_index={h_index}")

        # Ordenamos veticalmente (Menor a mayor/Arriba abajo). Si el primer elemento cuya altura sea mayor
        # que la del símbolo (hacemos esto para comprobar el primer contorno mas cercano) no es un '-' quiere decir que
        # es un simbolo de division

        # Equivalente a sacar el menor valor absoluto de la distancia.
        isDiv = False
        i = 0
        while not isDiv and i < vertical_const_list.shape[0]:
            [_, y, _, h] = vertical_const_list[i]

            categorie_index = np.where(np.all(contours == vertical_const_list[i],axis=1))
            # print(np.where(np.all(contours == vertical_const_list[i],axis=1)))
            print(f"ITERATION {i} WITH ({vertical_const_list[i]}) AND CATEGORIE = {categories[categorie_index]}")
            if (y + h) > (y_index + h_index):
                if categories[categorie_index] != 11:
                    isDiv = True
                else:
                    print(f"INDEX({index}) IS NOT A DIVISION")
                    break
            i += 1

        if isDiv:
            div_index.append(index)
            # We take the range of indexes that covers the div symbol
            min_index = np.where(np.all(contours == horizontal_cont_list[0],axis=1))[0][0]
            max_index = np.where(np.all(contours == horizontal_cont_list[-1],axis=1))[0][0]

            div_ranges.append([min_index, max_index])

            print(f"Div Detected: index = {index} --- min_index = {min_index} --- max_index = {max_index}")

    # Convert to numpy arrays
    div_index = np.array(div_index)
    div_ranges = np.array(div_ranges)

    upper_string = ""
    lower_string = ""

    print("----------- ALL DIVS HAVE BEEN DETECTED - WE CREATE THE EXPRESSION ----------------")
    print(f"\ndiv_index={div_index}\ndiv_ranges={div_ranges}\n")

    # Generate the final string that will be evaluated
    for i in range(0,categories.shape[0]):
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
        print(f"FOR INDEX = {i} -> (inRange={inRange},isLastIndex={isLastIndex},range_div={range_div})")
        if inRange:
            if contours[i, 1] < contours[div_index[range_div], 1]:
                # If the contour is above the line
                upper_string += category_toString_conversion(categories[i])
            else:
                # If the contour is below the line
                lower_string += category_toString_conversion(categories[i])
            if isLastIndex:
                # If it is the last element of the range, combine all elements
                expression += "((" + upper_string + ")/(" + lower_string + "))"

                # Reset the strings
                upper_string = ""
                lower_string = ""
        else:
            # If it is not part of any range, simply add the number to the string
            expression += category_toString_conversion(categories[i])

    return expression


def average_height(categories, contours):
    # Calculate the average height of the NUMBERS ONLY contours

    height_list = []
    for i in range(categories.shape[0]):
        if categories[i] < 10:
            height_list.append(contours[i, 3])

    height_list = np.array(height_list)
    return np.average(height_list)


# --------------------------------------- TEST --------------------------------------------------
# LOAD WEIGHTS VECTOR
clasiffication.load_weights()
clasiffication.load_model()

# -----------------------IMAGE PREPROCESSING -----------------------------------
image = cv2.imread('../../images/test16.jpeg', -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
smoothed_image = gaussian_filter(image, 4, 1)

if debug:
    plt.imshow(smoothed_image, cmap='grey')
    plt.show()

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(48, 48))
enhaced_smoothed = clahe.apply(smoothed_image)

_, thresh_img = cv2.threshold(enhaced_smoothed, 170, 255, cv2.THRESH_BINARY)


if debug:
    plt.imshow(enhaced_smoothed, cmap='grey')
    plt.show()

# binary_img = binarize_em(enhaced_smoothed,5)
# binary_img = binarize_kmeans(smoothed_image, 5)

# ------------------------CONTOUR EXTRACTION AND SORTING-----------------------------
contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contours_ordered = [cv2.boundingRect(cnt) for cnt in contours]
contours_ordered = non_maxima_supression_contours(contours_ordered)
contours_ordered = np.array(contours_ordered)

print(contours_ordered)
print("---------------------------------")

# Sort the contours from LEFT to RIGHT
contours_ordered = contours_ordered[np.argsort(contours_ordered[:, 0], kind='mergesort')]

print(contours_ordered)

if True:
    plt.imshow(draw_contours(thresh_img, contours_ordered))
    plt.show()

# This array contains the 28x28 images of all the contours in the image
data = np.zeros((28, 28, len(contours_ordered)))
count = 0
for x, y, w, h in contours_ordered:
    # Generates de 28x28 from the 2D array(Bounding Rect of each contour)
    data[:, :, count] = clasiffication.fit_contour(thresh_img[y:y + h, x:x + w])
    count += 1

# --------------------------CONTOUR CLASSIFICATION------------------------------------------

res_string, categories = generate_string(data)
print(res_string)

# Now that we have contours classified we can generate the mathematical expression
# Now we only have interest

final_expression = generate_expression(categories,contours_ordered)
print(final_expression)
try:
    print(eval(final_expression))
except Exception:
    print("ERROR: La operación no es válida")


debug_image = draw_contours(thresh_img, contours_ordered)

classified_characters = np.zeros(len(contours_ordered))
plt.imshow(debug_image)
# plt.show()
