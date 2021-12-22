import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt

modelo_red_neuronal = load_model("C:/Users/marit/Documents/red neuronal/red_neuronal.h5py")

# summarize model
modelo_red_neuronal.summary()

# load dataset
from keras.datasets import cifar10

# Loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# evaluate the model
score = modelo_red_neuronal.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (modelo_red_neuronal.metrics_names[1], score[1]*100))


y_pred = modelo_red_neuronal.predict(X_test)
print(y_pred[5:6])
print(y_pred.shape)

y_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

imagen = 6528;

for i in range(imagen,imagen+1):
    x = np.argmax(y_pred[i:i+1])
    print("the picture", i,  "is a", y_classes[x], "with a probability of:", y_pred[i,x]*100)
    #print(y_pred[i,:]*100)
    my_image = X_test[i]
    plt.imshow(my_image)
    plt.show()
