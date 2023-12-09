import cv2
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib


mnist = keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data("mnist.npz")

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train[0])

#x_train = keras.utils.normalize(x_train,axis=1)
#x_test = keras.utils.normalize(x_test,axis=1)


#model = keras.models.Sequential()
#model.add(keras.layers.Flatten(input_shape=(28,28)))
#model.add(keras.layers.Dense(128,activation='relu'))
#model.add(keras.layers.Dense(128,activation='relu'))
#model.add(keras.layers.Dense(10,activation='softmax'))

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train,epochs=5)

#model.save('handwritting.model')

model = keras.models.load_model('handwritting.model')

#loss,accuracy = model.evaluate(x_test,y_test)

path = '../../images/datasets/'

for i in range(0, 12):
    image = cv2.imread(path + 'prueba_' + str(i) + '.jpg')[:,:,0]
    image = np.array([image])

    print(image.shape)

    prediction = model.predict(image)

    print(f"This digit is a {np.argmax(prediction)}")

    plt.imshow(image[0],cmap='grey')
    plt.show()






