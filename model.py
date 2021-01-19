from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import requests

# NOTE: uncomment this if train using GPU
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_shape = 240

model_path = "Trained_model"
CATEGORIES = ['bathroom', 'bedroom', 'dining room', 'exterior', 'interior', 'kitchen', 'living room']
# load the model we saved
image_path = 'image_to_predict/'

class Model():
    def __init__(self):
        self.model = load_model(model_path)

    def download_image_ipg(self, urls, file_path):
        for i, url in enumerate(urls):
            full_path = file_path + format(i, '04d') + '.png'
            url = 'http://papers.xtremepapers.com/CIE/Cambridge%20IGCSE/Mathematics%20(0580)/0580_s03_qp_1.pdf'
            r = requests.get(url)
            with open('0580_s03_qp_1.pdf', 'wb') as outfile:
                outfile.write(r.content)

    def predict(self, urls):
        # download the images first
        self.download_image_ipg(urls, image_path)
        images = []
        # predicting images
        for i in range(len(urls)):
            img = image.load_img(image_path + format(i, '04d') + '.png', target_size=(image_shape, image_shape))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255
            images.append(x)

        images = np.vstack(images)
        classes = self.model.predict(images)
        # pred_name = CATEGORIES[np.argmax(classes, axis = 1)]
        pred_name = [CATEGORIES[i] for i in np.argmax(classes, axis = 1)]
        return pred_name
