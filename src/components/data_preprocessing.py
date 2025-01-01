from math import e
import os
import sys

import pandas as pd

from dataclasses import dataclass

from sqlalchemy.sql.ddl import exc

from src.components import data_ingestion
from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import save_model

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = get_logger('data-preprocessing')

@dataclass
class PreprocessorPath:
    preprocessor_path: str = os.path.join('model', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.preprocessor_path = PreprocessorPath()

    def get_preprocessor(self, independent_data):
        '''
        This method will create our preprocessor object and returns it.

        Args:
            independent_data: our independent data use to extract columns

        Returns:
            preprocessor object
        '''

        try:
            logger.info('Creating our Preprocessor object')

            logger.info('Extracting columns')
            numerical_columns = independent_data.select_dtypes('number').columns
            categorical_columns = independent_data.select_dtypes('object').columns

            logger.info('Creating our pipeline')
            numerical_pipeline = Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scaling', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder(sparse_output=False)),
                ('scaling', StandardScaler())
            ])
            logger.info('Pipeline Created Successfully')

            logger.info('Creating our ColumnTransformer Object')

            preprocessor = ColumnTransformer([
                ('numerical_trf', numerical_pipeline, numerical_columns),
                ('categorical_trf', categorical_pipeline, categorical_columns)
            ])
            
            logger.info('Preprocessor Object Created Successfully')

            return preprocessor

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def load_train_test_data(self, train_path, test_path):
        '''
        This method will take train and test path and returns dependent and independent features.

        Args:
            train_path (string): saved training data path.
            test_path (string): saved testing data path.

        Returns:
            X_train, X_test, y_train, y_test.
        '''

        try:
            logger.info('Loading our train and test data from path')
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logger.info('Data Loaded')

            logger.info('Splitting our data to Independent and Dependent Features')
            
            DEPENDENT_FEATURE  = 'fitness_level'

            X_train, X_test, y_train, y_test = (
                train_data.drop(DEPENDENT_FEATURE, axis=1),
                test_data.drop(DEPENDENT_FEATURE, axis=1),
                train_data[DEPENDENT_FEATURE],
                test_data[DEPENDENT_FEATURE]
            )

            logger.info('Data Splitted successfully')

            return (
                X_train, 
                X_test,
                y_train, 
                y_test
            )
            
            
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        

    def preprocess_data(self, train_path, test_path):
        '''
        This method will take train and test path and returns preprocessed X_train, X_test, y_train, y_test.

        Args:
            train_path (string): saved training data path.
            test_path (string): saved testing data path.

        Returns:
            X_train, X_test, y_train, y_test.
        '''

        try:
            logger.info('Data Preprocessing Started')

            X_train, X_test, y_train, y_test = self.load_train_test_data(train_path, test_path)

            preprocessor = self.get_preprocessor(X_train)

            logger.info('Applying Transformation')
            
            logger.info('Transforming Train data')
            X_train = preprocessor.fit_transform(X_train)

            logger.info('Transforming Test data')
            X_test = preprocessor.transform(X_test)

            logger.info('Data Preprocessing Completed')

            save_model(
                self.preprocessor_path.preprocessor_path,
                preprocessor
            )

            return (
                X_train, 
                X_test, 
                y_train, 
                y_test
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.load_data()

    data_preprocessing = DataPreprocessing()
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(train_path, test_path)