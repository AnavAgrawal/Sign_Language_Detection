from tensorflow.keras.models import load_model

model = load_model('sign_lang_model.h5')
model.summary()
