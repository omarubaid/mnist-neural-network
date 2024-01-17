# MNIST Digit Classification with Keras
MNIST ConvNet implementation, training, evaluation, and prediction.


## Overview

This project implements a Convolutional Neural Network (CNN) using Keras to recognize handwritten digits from the MNIST dataset. The trained model is saved and later used to make predictions on user-provided images.

----------
## Project Structure

- `mnist_cnn_with_dropout.py`: Contains the code to train the CNN on the MNIST dataset and save the model as 'mnist_model.h5'.
- `loaded_mnist_model_digit_predictor.py`: Loads the trained model and makes predictions on user-provided images.

-----------
## Model Architecture

- Input Layer: Accepts 28x28 grayscale images.
- Convolutional Layers: Utilizes 32 and 64 filters of size (3, 3) with ReLU activation.
- Max Pooling Layers: Downsamples feature maps with a pool size of (2, 2).
- Flatten Layer: Converts 2D feature maps to a 1D vector.
- Dense Layers: Consists of two hidden layers with 256 and 64 units, respectively, using ReLU activation.
- Dropout Layers: 30% applied for regularization to prevent overfitting.
- Output Layer: Has 10 units representing digits 0-9 with softmax activation.
-----------
## Data Preprocessing

- Images are normalized to a range between 0 and 1.
- Training data is split into training and validation sets using a 90-10 split.
- Batch normalization is applied to improve training speed and stability.
----------
## Model Training

- The model is trained for 10 epochs with a batch size of 128.
- Early stopping is employed with a patience of 3 to monitor validation loss and restore the best weights.
-------------
## Model Evaluation

- The trained model is evaluated on the test set to assess its accuracy.
--------
## Visualizing Results

The test accuracy on 10000 samples was recorded as:
- Test Accuracy: 0.9902999997138977 (**99.03 %**)

---------
## Dependencies

- Python 3.x
- Keras
- Matplotlib
- NumPy
- PIL (Python Imaging Library)
- scikit-learn

Install dependencies using:

```bash
pip install keras matplotlib numpy pillow scikit-learn
```
-------------

## Project Considerations

- The model achieves high accuracy on the MNIST dataset, demonstrating its effectiveness in digit recognition.
- Employing early stopping helps prevent overfitting and ensures the model generalizes well.
- Model predictions can be further improved with experimentation in image pre-processing techniques.
--------------------------------



   





