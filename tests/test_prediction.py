from cnnClassifier.pipeline.prediction import PredictionPipeline


def test_class_1_is_normal(png_path, fake_model):
    pipeline = PredictionPipeline(fake_model([0.1, 0.9]))
    assert pipeline.predict(png_path) == {"prediction": "Normal", "confidence": 0.9}


def test_class_0_is_adenocarcinoma(png_path, fake_model):
    pipeline = PredictionPipeline(fake_model([0.75, 0.25]))
    assert pipeline.predict(png_path) == {
        "prediction": "Adenocarcinoma Cancer",
        "confidence": 0.75,
    }
