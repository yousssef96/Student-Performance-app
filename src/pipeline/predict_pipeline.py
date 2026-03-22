import pandas as pd
from pathlib import Path
from src.utils import load_object
from src.logger import logger

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = Path("artifacts") / "model.pkl"
            preprocessor_path = Path("artifacts") / "preprocessor.pkl"

            logger.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logger.info("Model and preprocessor loaded successfully")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            logger.exception(f"Error in prediction pipeline: {e}")
            raise