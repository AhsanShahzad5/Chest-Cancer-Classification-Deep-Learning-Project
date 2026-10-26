import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from cnnClassifier.constants import CLASS_NAMES


class PredictionPipeline:
    """Turns an image file into a label and a confidence score. It receives an
    already-loaded model, so it never touches the disk or MLflow to get one."""

    def __init__(self, model):
        self.model = model

    def predict(self, image_path: str) -> dict:
        img = image.load_img(image_path, target_size=(224, 224))
        batch = np.expand_dims(image.img_to_array(img), axis=0)
        # same preprocessing as training, otherwise predictions are unreliable
        batch = preprocess_input(batch)
        probabilities = self.model.predict(batch)[0]
        class_index = int(np.argmax(probabilities))
        return {
            "prediction": CLASS_NAMES[class_index],
            "confidence": round(float(probabilities[class_index]), 4),
        }
