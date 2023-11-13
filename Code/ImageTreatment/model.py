import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

matplotlib.rcParams['figure.figsize'] = (16.0, 8.0)

rSize = 28
cSize = 28

weights_dataset = None

def generate_dataset(class_character):

    path = '../../images/datasets/dataset1/' + str(class_character) + '/'

    count = 0
    for number in range(0,17):
        #Read image and invert colors
        image = cv2.imread(path + 'image_' + str(class_character) + '_' + str(number) + '.png', 0)
        image = cv2.bitwise_not(image)

        #Binarize the image. (This is due to using the brush on paint)
        _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

        #Find all contours in the image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        print(str(len(contours)) + ' contours in image: ' + str(number))

        #For each contour in the image
        for cnt in contours:
            #Extract de origin and size of the bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            #Subtract that bpunding rect to another image
            specific_contour = image[y:y + h, x:x + w]

            res_image = fit_contour(specific_contour)

            #Save image on path
            cv2.imwrite(path + 'img_' + str(class_character) + '_' + str(count) + '.jpg', res_image)

            count += 1

#Given an rectangle contour, it returns the 28x28 resized version of the template.
def fit_contour(contour_input):

    contour = np.copy(contour_input)
    (h,w) = contour.shape

    aspect_ratio = w / h

    # I use the aspect ratio to adjust the imagen in order to fit it in a 24x24 image.
    scaled_w = int(min(24, 24 * aspect_ratio))
    scaled_h = int(min(24, 24 / aspect_ratio))

    resized_contour = cv2.resize(contour, (scaled_w, scaled_h))
    small_image = np.zeros((24, 24))

    #Calculate the free rows/columns in the image
    free_space = int((24 - min(scaled_w, scaled_h)))
    #Left-Bottom offset
    lb_offset = int(free_space / 2)
    #Right-Top offset
    rt_offset = free_space - lb_offset

    #Depending on the longest axis, we will use verical or horizontal offset
    #Designed to work on square contours too
    if scaled_w == 24:
        small_image[lb_offset:(24 - rt_offset)][:] = resized_contour
    else:
        small_image[:][lb_offset:(24 - rt_offset)] = resized_contour

    # Threshold the image due to the resize algorithm
    _, small_image = cv2.threshold(small_image, 10, 255, cv2.THRESH_BINARY)
    small_image_pad = cv2.copyMakeBorder(small_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    return small_image_pad

def compute_weights():
    dataset = np.zeros((rSize * cSize, 500, 14))

    # For each class
    for number in range(15):
        # For each image in the class
        for i in range(530):
            # Read the image
            path = '../../images/datasets/dataset1/' + str(number) + '/img_' + str(number) + '_' + str(i) + '.jpg'
            image = cv2.imread(path, 0)
            # Binarize it
            vector_image = image.reshape((-1, 1))
            # Reshape it
            dataset[:, i, number] = vector_image[:, 0]

    # Compute probabilities
    probabilities = np.sum(dataset, axis=1) / dataset.shape[1]
    probabilities = np.where(probabilities == 0, 0.001, probabilities)
    probabilities = np.where(probabilities == 1, 0.999, probabilities)

    # Initialize matrix
    weights = np.zeros((rSize * cSize + 1, 14))

    # Compute weights
    weights[:-1, :] = np.log(probabilities / (1 - probabilities))
    weights[-1, :] = np.log(1 / dataset.shape[2]) + np.sum(np.log(1 - probabilities), axis=0)

    #Save the computed weights on a file on directory
    np.save('computed_weights.npy',weights)

def load_weights():
    global weights_dataset
    try:
        weights_dataset = np.load('computed_weights.npy')
    except FileNotFoundError:
        print('Error: EL ARCHIVO NO SE HA ENCONTRADO')


generate_dataset(12)

