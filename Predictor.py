from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('sign_lang_model.h5')
# model.summary()

image = plt.imread('testimage.jpg')

# print(image.shape)
image = np.expand_dims(image, -1)
image = np.expand_dims(image, 0)
# print(image.shape)

preds = model.predict(image)
prednum = np.argmax(preds)
print('Predicted letter is ', chr(prednum+97))
