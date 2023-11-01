import numpy as np
import cv2
import matplotlib.pyplot as plt
class ImageEnhaced:
    def __init__(self, originImage):
        self.originImage = originImage

    #Takes a GRAYSCALE image and binzarizes it through K-Means
    def binarizeImage(image):
        # Set random seed for centroids
        cv2.setRNGSeed(124)

        # Flatten image
        flattened_img = image.reshape((-1, 1))
        flattened_img = np.float32(flattened_img)

        # Set epsilon
        epsilon = 0.2

        # Estabish stopping criteria (either `it` iterations or moving less than `epsilon`)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, epsilon)

        # Set K parameter (2 for thresholding)
        K = 2

        # Call kmeans using random initial position for centroids
        _, label, center = cv2.kmeans(flattened_img, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        # Colour resultant labels
        center = np.uint8(center)  # Get center coordinates as unsigned integers
        print(center)
        flattened_img = center[label.flatten()]  # Get the color (center) assigned to each pixel

        # Reshape vector image to original shape
        binarized = flattened_img.reshape((image.shape))

        # Show resultant image
        plt.subplot(2, 1, 1)
        plt.title("Original image")
        plt.imshow(binarized, cmap='gray', vmin=0, vmax=255)

        # Show how original histogram have been segmented
        plt.subplot(2, 1, 2)
        plt.title("Segmented histogram")
        plt.hist([image[binarized == center[0]].ravel(), image[binarized == center[1]].ravel()], 256, [0, 256],
                 color=["black", "gray"], stacked="true")

        return binarized