import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import keras

matplotlib.rcParams['figure.figsize'] = (16.0, 8.0)
images_path = '../../images/datasets/'

rSize = 28
cSize = 28

weights_dataset = None
model = None


def generateData(number):
    num_img = 1
    has_images = True
    count = 1
    while has_images:
        try:
            path = '../../images/datasets/dataset1/' + str(number) + '/'

            image = cv2.imread(path + 'moreData_' + str(number) + "_" + str(num_img) + '.jpeg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.bitwise_not(image)

            # Binarize the image.
            _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

            plt.imshow(image, cmap='grey')
            plt.show()

            # Find all contours in the image
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # print(str(len(contours)) + ' contours in image: ' + str(number))

            for cnt in contours:
                # Extract de origin and size of the bounding rectangle
                x, y, w, h = cv2.boundingRect(cnt)

                # Do not count too small contours
                if w < 5:
                    continue

                # Subtract that bpunding rect to another image
                specific_contour = image[y:y + h, x:x + w]

                res_image = fit_contour(specific_contour)
                # Save image on path
                cv2.imwrite(path + 'img_' + str(number) + '_' + str(count) + '.jpg', res_image)
                count += 1
        except Exception:
            has_images = False
        finally:
            num_img += 1


# Given a rectangle contour, it returns the 28x28 resized version of the template.
def fit_contour(contour_input):
    window_size = 24
    contour = np.copy(contour_input)
    (h, w) = contour.shape

    aspect_ratio = w / h

    # I use the aspect ratio to adjust the imagen in order to fit it in a 24x24 image.
    scaled_w = int(min(window_size, window_size * aspect_ratio))
    scaled_h = int(min(window_size, window_size / aspect_ratio))

    if scaled_w == 0:
        scaled_w = 3
    if scaled_h == 0:
        scaled_h = 3

    resized_contour = cv2.resize(contour, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    # plt.imshow(resized_contour, cmap='grey')
    # plt.show()

    small_image = np.zeros((window_size, window_size))

    # Calculate the free rows/columns in the image
    free_space = int((window_size - min(scaled_w, scaled_h)))
    # Left-Bottom offset
    lb_offset = int(free_space / 2)
    # Right-Top offset
    rt_offset = free_space - lb_offset

    # Depending on the longest axis, we will use verical or horizontal offset
    # Designed to work on square contours too
    if scaled_w == window_size:
        small_image[lb_offset:(window_size - rt_offset), :] = resized_contour
    else:
        small_image[:, lb_offset:(window_size - rt_offset)] = resized_contour

    # Threshold the image due to the resize algorithm
    _, small_image = cv2.threshold(small_image, 20, 255, cv2.THRESH_BINARY)
    # Add padding to match the 28x28 size
    small_image_pad = cv2.copyMakeBorder(small_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # plt.imshow(small_image_pad, cmap='grey')
    # plt.show()

    return small_image_pad


def load_model():
    global model
    try:
        model = keras.models.load_model('../model/visionmath.model')
    except FileNotFoundError:
        print('Error: EL ARCHIVO NO SE HA ENCONTRADO')


def classify_input_model(image):
    global model

    # Adapt the image to (1,28,28) input
    image = np.array([image[:, :]])

    category = model.predict(image)

    return np.argmax(category)


def generate_train_data():
    num_categories = 13
    X_train = []
    Y_train = []

    for category in range(num_categories):
        for i in range(1, 900):
            image = None
            try:
                image = cv2.imread(
                    images_path + 'dataset1/' + str(category) + '/img_' + str(category) + '_' + str(i) + '.jpg')[:, :, 0]
            except Exception:
                break

            _, binarized = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

            # Añadir la imagen y la categoria al dataset
            X_train.append(binarized)
            Y_train.append(category)

    # Normalize data
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # print(X_train.shape)

    X_train = keras.utils.normalize(X_train, axis=1)
    # Train model
    model_fit(X_train, Y_train)


def model_fit(X_train, Y_train):
    global model

    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(13, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10)

    model.save('../model/visionmath.model')
    model.summary()

# path = '../../images/datasets/dataset1/'
#
# for number in range(0,10):
#     for index in range (1,501):
#
#         image = cv2.imread(path + str(number) + '/imagen' + str(number) + '_' + str(index) + '.png')
#         cv2.imwrite(path + str(number) + '/img_' + str(number) + '_' + str(index) + '.jpg',image)

# generate_train_data()
# load_model()

# image = cv2.imread('../../images/prueba_3.png', -1)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.bitwise_not(image)

# print(classify_input_model(image))
# compute_weights()
