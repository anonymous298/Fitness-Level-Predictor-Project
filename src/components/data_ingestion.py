from operator import index
import os
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger('data-ingestion')

@dataclass
class DataPaths:
    clean_data_path: str = r'F:\Projects\Fitness-Level-Predictor-Project\research\data\clean\clean_data.csv'
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_paths = DataPaths()

    def load_data(self):
        '''
        This method will load data and return train and test path

        Returns:
            train and test path.
        '''

        try:
            logger.info('Data Ingestion started')
            data = pd.read_csv(self.data_paths.clean_data_path)
            logger.info('Clean Data Loaded')

            logger.info('Splitting our data to train and test')
            train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

            logger.info('Creating our Artifacts Folder if not exists')
            os.makedirs(os.path.dirname(self.data_paths.train_data_path), exist_ok=True)

            logger.info('Saving our train data to Artifacts')
            train_data.to_csv(self.data_paths.train_data_path, index=False)

            logger.info('Saving our test data to Artifacts')
            test_data.to_csv(self.data_paths.test_data_path, index=False)

            logger.info('Data Ingestion Completed')

            return (
                self.data_paths.train_data_path,
                self.data_paths.test_data_path
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.load_data()
