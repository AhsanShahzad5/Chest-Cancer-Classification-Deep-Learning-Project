from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# --- model serving ---
# the registry model the API serves: "VGG16Model" version that has the "champion" alias
DEFAULT_MODEL_URI = "models:/VGG16Model@champion"
# where the training pipeline writes the freshly trained model
TRAINED_MODEL_PATH = Path("artifacts/training/model.h5")
# local fallback copy, used only when MODEL_SOURCE=local
LOCAL_MODEL_PATH = Path("model/model.h5")

# index in the model's output = class name (folders are sorted alphabetically)
CLASS_NAMES = ["Adenocarcinoma Cancer", "Normal"]
