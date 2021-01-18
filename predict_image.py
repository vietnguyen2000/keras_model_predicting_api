from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import urllib.request

image_shape = 240

model_path = "Trained_model"
CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']
# load the model we saved
model = load_model(model_path)

def download_image_ipg(url, file_path, file_name):
    fullpath=file_path+file_name+".png"
    urllib.request.urlretrieve(url,fullpath)

def predict(url):
    # download the image first
    download_image_ipg(url, 'image_to_predict/', 'data')
    # predicting images
    img = image.load_img('image_to_predict/data.png', target_size=(image_shape, image_shape))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255

    images = np.vstack([x])
    classes = model.predict(images)
    pred_name = CATEGORIES[np.argmax(classes)]
    return pred_name