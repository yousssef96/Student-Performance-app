from src.logger import logger
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.components.data_validation import DataValidation

@dataclass
class DataIngestionConfig:
    train_data_path: Path = Path("artifacts") / "train.csv"
    test_data_path: Path = Path("artifacts") / "test.csv"
    raw_data_path: Path = Path("artifacts") / "raw.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logger.info("Read the dataset as dataframe")

            validator = DataValidation(df)
            is_valid = validator.run_validation()

            if not is_valid:
                raise ValueError("Data validation failed — check logs for details")

            self.ingestion_config.train_data_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logger.exception(f"Error in data ingestion: {e}")
            raise

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))




