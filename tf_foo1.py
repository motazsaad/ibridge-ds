'''
Keras Tensorflow Steps
1. install and check version
2. load data
3. prepare data
3. define NN model
4. train (fit)
5. predict
6. evaluate

'''
# 1. import and check version
import tensorflow as tf

print(tf.__version__)

# 2. load and prepare dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# optional
x_train = x_train / 255.0  # normalize values between 0-1
x_test = x_test / 255.0  # normalize values between 0-1

# step 3 define model tf.keras.models
model = tf.keras.models.Sequential()

# add layers
# input layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # input layer
# hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# define optimizer
# compile model
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy', 'mae'])
print(model.summary())
# 4. train NN
model.fit(x=x_train, y=y_train, epochs=3, batch_size=100)
# 5. evaluate
result = model.evaluate(x=x_test, y=y_test)
print(result)