import numpy as np
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for purposes of reproducibility
seed = 21

from keras.datasets import cifar10

# Loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]






# Initialize model
model = keras.Sequential()

## FIRST LAYER #############################################################################

# First convolutional layer
model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=X_train.shape[1:], padding='same'))

#  First dropout layer
model.add(keras.layers.Dropout(0.2))

# First batch normalizacion layer
model.add(keras.layers.BatchNormalization())

## SECOND LAYER ##############################################################################

# Second convolutional layer
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))

# Second activation layer
model.add(keras.layers.MaxPooling2D(2))

# Second dropout 
model.add(keras.layers.Dropout(0.2))

# Second batch normalization
model.add(keras.layers.BatchNormalization())

## THIRD LAYER ##############################################################################

# Second convolutional layer
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))

# Second activation layer
model.add(keras.layers.MaxPooling2D(2))

# Second dropout 
model.add(keras.layers.Dropout(0.2))

# Second batch normalization
model.add(keras.layers.BatchNormalization())

## FORTH LAYER ###############################################################################

# Third convolutional layer
model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))

# Third dropout 
model.add(keras.layers.Dropout(0.2))

# Third batch normalization
model.add(keras.layers.BatchNormalization())




## FLATTEN LAYER ###########################################################################################

model.add(keras.layers.Flatten())

# Forth dropout
model.add(keras.layers.Dropout(0.2))

## DENSE LAYERS ###########################################################################################

# First dense layer
model.add(keras.layers.Dense(32, activation='relu'))

# Fifth dropout
model.add(keras.layers.Dropout(0.3))

# Fifth batch normalization
model.add(keras.layers.BatchNormalization())

# Second dense layer
model.add(keras.layers.Dense(class_num, activation='softmax'))


## COMPILATION ############################################################################################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## PLOTEAR MODEL SUMMARY ##################################################################################
print(model.summary())

### TRAINING THE MODEL ###################################################################################

np.random.seed(seed)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64)

model.save("C:/Users/marit/Documents/red neuronal/red_neuronal.h5py")

print("Model saved in C:/Users/marit/Documents/red neuronal/red_neuronal.h5py")

test_eval = model.evaluate(X_test, y_test, verbose=1)
 
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

pd.DataFrame(history.history).plot()
plt.show()