import os

import keras
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        ## load model

        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model: keras.Model = load_model(os.path.join("model", "model.h5"))  # type: ignore[assignment]

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        # same preprocessing as training, otherwise predictions are unreliable
        test_image = preprocess_input(test_image)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = "Normal"
            return [{"image": prediction}]
        else:
            prediction = "Adenocarcinoma Cancer"
            return [{"image": prediction}]
