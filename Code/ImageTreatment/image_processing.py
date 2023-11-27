import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import clasiffication
import math


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


# Bubble-Sort algorithm that sorts the contours from LEFT TO RIGHT
def sort_contours(contours, axis):
    n = len(contours)
    swapped = False
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if contours[j, axis] > contours[j + 1, axis]:
                swapped = True
                contours[j], contours[j + 1] = contours[j + 1], contours[j]
        if not swapped:
            return


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


# Classifies the input data and returns the resultant string
def generate_string(data):
    n_elements = data[0, 0, :].shape[0]
    res_string = ''
    count = 0

    categories = []

    for i in range(n_elements):
        # Classify each image
        # category = clasiffication.classify_input_weights(data[:,:,i])
        category = clasiffication.classify_input_model(data[:, :, i])
        # Insert category in categories array
        categories.append(category)
        # plt.imshow(data[:,:,i],cmap='grey')
        # plt.show()
        # cv2.imwrite('../../images/datasets/prueba_' + str(count) + '.jpg', data[:, :, i])
        # count += 1

        if category == 10:
            res_string += '+'
        elif category == 11:
            res_string += '-'
        elif category == 12:
            res_string += '*'
        else:
            res_string += str(category)

    # Convert to numpy array
    categories = np.array(categories)

    return res_string, categories


def generate_expression(categories, contours):
    # We must distinguish when a '-' means subtraction or division
    total_lines = np.where(categories == '11')

    div_index = []
    div_strings = []
    div_ranges = []

    expression = ""

    #Identify the real division symbols
    for index in total_lines:
        (x_index, y_index, w_index, h_index) = contours[index]
        # Contours list that sorround the symbol in the x axis
        cont_list = [(x, y, w, h) for x, y, w, h in contours if x + w > x_index and x < x_index + w_index and
                     (x != x_index and y != y_index)]
        cont_list = np.array(cont_list)
        cont_list_copy = np.copy(cont_list)

        # Order the cont_list according to the y axis
        sort_contours(cont_list, axis=1)

        ##HAY QUE CAMBIAR ESTOOOOOOO - EL ARRAY DEBERIA ORDENARSE EN SENTIDO OPUESTO PARA QUE FUNCIONE
        isDiv = False
        for (x, y, w, h) in cont_list:
            elem_index = np.where(contours == (x, y, w, h))
            if y < y_index and categories[elem_index[0]] != 11:
                isDiv = True
                break

        if isDiv:
            div_index.append(index)
            min_index = np.where(contours == cont_list_copy[0])
            max_index = np.where(contours == cont_list_copy[-1])

            index_list = np.array(range(min_index[0], max_index[0] + 1))

            upper_string = ""
            lower_string = ""

            for i in index_list:
                if contours[i, 1] > y:
                    upper_string += str(categories[i])
                else:
                    lower_string += str(categories[i])

            div_string = "(" + upper_string + ")/(" + lower_string + ")"

            div_strings.append(div_string)
            div_ranges.append((min_index[0], max_index[0]))

    # Convert to numpy arrays
    div_index = np.array(div_index)
    div_strings = np.array(div_strings)
    div_ranges = np.array(div_ranges)

    # Evaluate the rest of strings and create the final expression
    for i in range(len(categories)):
        inRange = False
        # If it is a division symbol, continue
        if np.isin(i, div_index):
            continue
        # Add the division expression only when the index is the same as the maximum of that range
        cont = 0
        for (min_index, max_index) in div_ranges:
            if min_index <= i <= max_index:
                inRange = True
                if i == max_index:
                    expression += div_strings[cont]
                break
            cont += 1
        #If it is not in any range, it means it is outside a division expression, so simply add it to the final string
        if not inRange:
            expression += str(categories[i])

    return expression


# --------------------------------------- TEST --------------------------------------------------
# LOAD WEIGHTS VECTOR
clasiffication.load_weights()
clasiffication.load_model()

# -----------------------IMAGE PREPROCESSING -----------------------------------
image = cv2.imread('../../images/test11.jpeg', -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
smoothed_image = gaussian_filter(image, 4, 1)

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(48, 48))
enhaced_smoothed = clahe.apply(smoothed_image)

_, thresh_img = cv2.threshold(enhaced_smoothed, 170, 255, cv2.THRESH_BINARY)

plt.imshow(enhaced_smoothed, cmap='grey')
plt.show()

# binary_img = binarize_em(enhaced_smoothed,5)
# binary_img = binarize_kmeans(smoothed_image, 5)

# ------------------------CONTOUR EXTRACTION AND SORTING-----------------------------
contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contours_ordered = [cv2.boundingRect(cnt) for cnt in contours]
contours_ordered = non_maxima_supression_contours(contours_ordered)
contours_ordered = np.array(contours_ordered)
sort_contours(contours_ordered, 0)

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

# Now that we have contours classified we can generate the mathematical expression
# Now we only have interest
final_expression = generate_expression(categories,contours_ordered)
print(final_expression)

debug_image = draw_contours(thresh_img, contours_ordered)

classified_characters = np.zeros(len(contours_ordered))
plt.imshow(debug_image)
# plt.show()
