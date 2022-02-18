import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras.datasets import imdb

# Load and vectorise data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Create model
def get_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam',
            loss='BinaryCrossentropy',
            metrics=['accuracy'])    
    return model

# Seperate data into train and validation sets
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train
model = get_model()       
history=model.fit(partial_x_train, partial_y_train, epochs=13)

# Compute and report accuracy
test_loss, test_acc =model.evaluate(x_val, y_val, verbose=2)
print('\nTest Accuracy:', test_acc)

# Plotting 
loss=history.history['loss']
accuracy=history.history['accuracy']
plt.plot(loss, label='loss')
plt.plot(accuracy, label='accuracy')
plt.legend()
plt.show()
