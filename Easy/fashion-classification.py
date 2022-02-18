import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255
test_images = test_images / 255

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10)])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model



model=get_model()
history=model.fit(train_images, train_labels, epochs=20)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose =2)
print('\nTest accuracy:', test_acc)


#       Plotting
loss=history.history['loss']
accuracy=history.history['accuracy']
plt.plot(loss)
plt.plot(accuracy)
plt.show()





































