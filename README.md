# MNIST Digit Classification with Keras
MNIST ConvNet implementation, training, evaluation, and prediction.


## Overview

This project implements a Convolutional Neural Network (CNN) using Keras to recognize handwritten digits from the MNIST dataset. The trained model is saved and later used to make predictions on user-provided images.


## Project Structure

- `mnist_cnn_with_dropout.py`: Contains the code to train the CNN on the MNIST dataset and save the model as 'mnist_model.h5'.
- `loaded_mnist_model_digit_predictor.py`: Loads the trained model and makes predictions on user-provided images.


## Dependencies

- Python 3.x
- Keras
- Matplotlib
- NumPy
- PIL (Python Imaging Library)

Install dependencies using:

```bash
pip install keras matplotlib numpy pillow
```

## Usage
   ### Training the Model

Run mnist_cnn_with_dropout.py to train the CNN on the MNIST dataset and the graph curves for loss and accuracy will be plotted. The trained model will be saved as 'mnist_model5.h5'.


   ### Making Predictions

Provide your handwritten digit image in the image_path variable in loaded_mnist_model_digit_predictor.py. Run predict_digit.py to load the trained model and make predictions on the provided image.


## File Descriptions

mnist_model5.h5: The saved Keras model file after training.
handwritten_digit.png: Example image for making prediction of the digit in the image.


## Visualizing Results

![Screenshot from 2023-12-16 21-58-49](https://github.com/omarubaid/mnist-neural-network/assets/142675270/77a35d3d-3150-4b91-bea5-3ac29102ec0b) 

------

![Screenshot from 2023-12-16 21-59-06](https://github.com/omarubaid/mnist-neural-network/assets/142675270/6675e9c6-d0fd-4695-945e-caa34b8a9735)


## Acknowledgment

This project utilizes the MNIST dataset provided by the Keras datasets module for training the model.


## Project Considerations

While the project successfully demonstrates handwritten digit recognition, it's important to note potential area for improvement:


   ### Model Predictions

The model's predictions are functional but may not be optimal in all cases. If you encounter instances of incorrect predictions, consider the following:

**Image Pre-processing:** Ensure input images conform to the specified dimensions (28x28 pixels) and are in grayscale. Experiment with different pre-processing techniques, including color inversion, for potential improvements.


--------------------------------



   





