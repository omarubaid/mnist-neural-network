from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

model = load_model('mnist_model.h5')

image_path = '/Downloads/handwritten_digit.png'
original_img = Image.open(image_path)

plt.imshow(original_img, cmap='gray')
plt.title('Original Image')
plt.show()

resized_img = original_img.resize((28, 28))
grayscale_img = ImageOps.grayscale(resized_img)
inverted_img = ImageOps.invert(grayscale_img)

plt.imshow(grayscale_img, cmap='gray')
plt.title('Preprocessed Image')
plt.show()

img_array = np.array(inverted_img)
img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

prediction = model.predict(img_array)

predicted_digit = np.argmax(prediction)
print(f'\nThe predicted digit is: {predicted_digit}')