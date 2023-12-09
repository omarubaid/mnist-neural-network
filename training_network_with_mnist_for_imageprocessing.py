from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Training data
print('Training samples shape:', x_train.shape)
print('Number of training labels:', len(y_train))
print('Training labels:', y_train)

# Test data
print('\nTest samples shape:', x_test.shape)
print('Number of test labels:', len(y_test))
print('Test labels:', y_test)

from keras import models
from keras import layers

network= models.Sequential()
network.add(layers.Dense(units=256, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(units=128, activation='relu'))
network.add(layers.Dense(units=10, activation= 'softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    ##preprocess data to be in interval [0, 1]##
x_train= x_train.reshape((60000, 28 * 28))
x_train= x_train.astype('float32') / 255

x_test= x_test.reshape((10000, 28 * 28))
x_test= x_test.astype('float32') / 255

    ##categorically encode the labels##
from keras.utils import to_categorical

y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

    ##train the network##
history= network.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

    ##Evaluate the model##
test_loss, test_acc = network.evaluate(x_test, y_test)
print('Test Accuracy:', test_acc)

# Plot the training history (loss and accuracy)
import matplotlib.pyplot as plt
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

# Plot training and validation loss
plt.plot(epochs, train_loss, 'o-', color='orange', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(epochs, train_acc, 'o-', color='orange', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()