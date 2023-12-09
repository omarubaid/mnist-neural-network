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