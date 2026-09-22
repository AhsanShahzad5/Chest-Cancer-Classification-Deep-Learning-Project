from pathlib import Path
from urllib.parse import urlparse

import mlflow
import tensorflow as tf
from mlflow.keras import log_model as log_keras_model

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        dataflow_kwargs = dict(  # noqa: C408
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data, shuffle=False, **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(dict(self.config.all_params))
            mlflow.log_metrics(
                {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
            )

            if tracking_url_type_store != "file":
                log_keras_model(
                    model=self.model,
                    name="model",
                    registered_model_name="VGG16Model",
                )
            else:
                log_keras_model(model=self.model, name="model")
