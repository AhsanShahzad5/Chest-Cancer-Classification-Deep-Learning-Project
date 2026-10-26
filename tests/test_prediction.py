from cnnClassifier.pipeline.prediction import PredictionPipeline


def test_class_1_is_normal(png_path, use_fake_model):
    use_fake_model([0.1, 0.9])
    assert PredictionPipeline(png_path).predict() == [{"image": "Normal"}]


def test_class_0_is_adenocarcinoma(png_path, use_fake_model):
    use_fake_model([0.9, 0.1])
    assert PredictionPipeline(png_path).predict() == [
        {"image": "Adenocarcinoma Cancer"}
    ]
