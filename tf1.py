'''
Tensorflow Steps
1. install and check version
2. load data
3. prepare data
3. define NN model
4. train (fit)
5. predict
6. evaluate


'''

import tensorflow as tf

mnist = tf.keras.datasets.mnist

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# convert to range 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())


model.fit(x_train, y_train, epochs=3)
print(model.evaluate(x_test, y_test))


