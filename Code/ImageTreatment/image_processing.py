import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
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

def binarize_em(image,it):
    cv2.setRNGSeed(5)

    # Define parameters
    n_clusters = 2
    covariance_type = 0  # 0: covariance matrix spherical. 1: covariance matrix diagonal. 2: covariance matrix generic
    epsilon = 0.2

    # Create EM empty object
    em = cv2.ml.EM_create()

    # Set parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, it, epsilon)
    em.setClustersNumber(n_clusters)
    em.setCovarianceMatrixType(covariance_type)
    em.setTermCriteria(criteria)

    # Flatten image
    flattened_img = image.reshape((-1, 1))
    flattened_img = np.float32(flattened_img)

    # Apply EM
    _, _, labels, _ = em.trainEM(flattened_img)

    # Reshape labels to image size (binarization)
    binarized = labels.reshape((image.shape))

    return binarized

#Bubble-Sort algorithm that sorts the contours from LEFT TO RIGHT
def sort_contours(contours):
    n = len(contours)
    swapped = False
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if contours[j][0] > contours[j + 1][0]:
                swapped = True
                contours[j], contours[j + 1] = contours[j + 1], contours[j]
        if not swapped:
            return

image = cv2.imread('../../images/test4.jpeg',-1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)

smoothed_image = gaussian_filter(image,4,1)

#clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(48,48))
#enhaced_smoothed = clahe.apply(smoothed_image)



#binary_img = binarize_em(enhaced_smoothed,5)
binary_img = binarize_kmeans(smoothed_image,5)

contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contours_ordered = [(cv2.boundingRect(cnt),cnt) for cnt in contours]
sort_contours(contours_ordered)

binary_img = cv2.cvtColor(binary_img,cv2.COLOR_GRAY2RGB)

count = 0
color = (0,0,0)
for ((x,y,w,h),cnt) in contours_ordered:
    if count == 0:
        color = (0,0,255)
    elif count == 1:
        color = (0,255,0)
    elif count == 2:
        color = (255,0,0)
    elif count == 3:
        color = (255,255,0)
    elif count == 4:
        color = (0,255,255)

    binary_img = cv2.rectangle(binary_img,(x-1,y-1),(x+w,y+h),color,3)
    count += 1


plt.imshow(binary_img)
plt.show()





