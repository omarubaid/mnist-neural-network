from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train= x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test= x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

y_train= to_categorical(y_train)
y_test= to_categorical(y_test)

network= models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPooling2D(2, 2))
network.add(layers.Flatten())
network.add(layers.Dense(units=128, activation='relu'))
network.add(layers.Dropout(0.3))
network.add(layers.Dense(units=10, activation='softmax'))

network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#network.save('mnist_model5.h5')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history= network.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), callbacks=[early_stopping])

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'o-', color='orange', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, train_acc, 'o-', color='orange', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()