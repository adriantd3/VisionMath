import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import keras

matplotlib.rcParams['figure.figsize'] = (16.0, 8.0)
images_path = 'images/datasets/'

rSize = 28
cSize = 28

model = None


def generateData(number):
    num_img = 1
    has_images = True
    count = 1
    while has_images:
        try:
            path = images_path + 'dataset1/' + str(number) + '/'

            image = cv2.imread(path + 'moreData_' + str(number) + "_" + str(num_img) + '.jpg')
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


# Given a rectangle contour, it returns the rSize x cSize resized version of the template.
def fit_contour(contour_input):
    window_size = rSize
    contour = np.copy(contour_input)
    (h, w) = contour.shape

    aspect_ratio = w / h

    # I use the aspect ratio to adjust the imagen in order to fit it in the rSize x cSize image
    scaled_w = int(min(window_size, window_size * aspect_ratio))
    scaled_h = int(min(window_size, window_size / aspect_ratio))

    if scaled_w == 0:
        scaled_w = 3
    if scaled_h == 0:
        scaled_h = 3

    resized_contour = cv2.resize(contour, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    small_image = np.zeros((window_size, window_size))

    # Calculate the free rows/columns in the image
    free_space = int((window_size - min(scaled_w, scaled_h)))
    # Left-Bottom offset
    lb_offset = int(free_space / 2)
    # Right-Top offset
    rt_offset = free_space - lb_offset

    # Depending on the longest axis, we will use vertical or horizontal offset
    # Designed to work on square contours too
    if scaled_w == window_size:
        small_image[lb_offset:(window_size - rt_offset), :] = resized_contour
    else:
        small_image[:, lb_offset:(window_size - rt_offset)] = resized_contour

    # Threshold the image due to the resize algorithm
    _, small_image = cv2.threshold(small_image, 20, 255, cv2.THRESH_BINARY)

    # plt.imshow(small_image_pad, cmap='grey')
    # plt.show()

    return small_image


def load_model():
    global model
    try:
        model = keras.models.load_model('visionmath.model')
    except FileNotFoundError:
        print('Error: EL ARCHIVO NO SE HA ENCONTRADO')


def classify_input_model(image):
    global model

    # Adapt the image to (1,28,28) input
    image = np.array([image[:, :]])

    category = model.predict(image)

    return np.argmax(category)


def generate_xy_train():
    num_categories = 13
    X_train = []
    Y_train = []

    for category in range(num_categories):
        for i in range(1, 900):
            try:
                image = cv2.imread(
                    images_path + 'dataset1/' + str(category) + '/img_' + str(category) + '_' + str(i) + '.jpg')[:, :,
                        0]
            except Exception:
                break

            # Añadir la imagen y la categoria al dataset
            X_train.append(image)
            Y_train.append(category)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # print(X_train.shape)

    # Normalize data
    X_train = keras.utils.normalize(X_train, axis=1)

    # Save train data
    np.save(images_path + 'dataset1/x_train.npy', X_train)
    np.save(images_path + 'dataset1/y_train.npy', Y_train)


def generate_y_test():
    file = open(images_path + 'dataset1/validation_data/validation_classification.txt', 'r')
    lines = file.readlines()

    Y_test = []

    for line in lines:
        Y_test.append(int(line))

    Y_test = np.array(Y_test)
    print(Y_test)
    np.save(images_path + 'dataset1/validation_data/y_test.npy', Y_test)


def generate_x_test():
    cont = 0
    X_test = []
    while True:
        try:
            image = cv2.imread(images_path + 'dataset1/validation_data/val_img_' + str(cont) + '.jpg')[:, :,
                    0]
        except Exception:
            break

        X_test.append(image)
        cont += 1

    X_test = np.array(X_test)
    print(X_test.shape)
    np.save(images_path + 'dataset1/validation_data/x_test.npy', X_test)


def load_train_data():
    x_train = np.load(images_path + 'dataset1/x_train.npy')
    y_train = np.load(images_path + 'dataset1/y_train.npy')

    return x_train, y_train


def load_test_data():
    x_test = np.load(images_path + 'dataset1/validation_data/x_test.npy')
    y_test = np.load(images_path + 'dataset1/validation_data/y_test.npy')

    return x_test, y_test


def model_fit():
    global model

    x_train, y_train = load_train_data()
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=(rSize, cSize)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(13, activation='softmax'))

    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)

    model.save('visionmath.model')
    model.summary()


def model_evaluate():
    # returns the accuracy of the model
    global model
    x_test, y_test = load_test_data()

    total = x_test.shape[0]
    correct = 0
    cont = 0

    for i in range(total):
        if y_test[i] == classify_input_model(x_test[i, :, :]):
            correct += 1

        cont += 1

    return (correct / total) * 100


# Generate the dataset from the moreData_X images
# for i in range(13):
# generateData(i)

# Generate the model
# generate_xy_train()
# model_fit()

# LOAD THE MODEL DATA
load_model()

# Evaluate the model
# generate_x_test()
# generate_y_test()
# print(model_evaluate())
