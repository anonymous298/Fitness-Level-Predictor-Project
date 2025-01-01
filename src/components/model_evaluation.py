import os
import sys

import dagshub
import mlflow

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from tensorflow.keras.models import load_model

logger = get_logger('model-evaluation')

class ModelEvaluation:
    def evaluate(self, model_path, X_train, X_test, y_train, y_test):
        '''
        This method will evaluate our model on training and testing data.

        Args:
            model_path: our saved model path
            X_train: our X-Training data.
            X_test: X-Testing data.
            y_train: y_training data.
            y_test: y_testing data.

        Returns:
            None
        '''

        try:
            logger.info('Model Evaluation Started')

            logger.info('Loading our saved Model')
            model = load_model(model_path)

            logger.info('Evaluating Model For Training Data')
            _, train_r2 = model.evaluate(X_train, y_train)

            logger.info('Evaluating Model For Testing Data')
            _, test_r2 = model.evaluate(X_test, y_test)

            logger.info('Connecting to DagsHub Client')
            dagshub.init(repo_owner='anonymous298', repo_name='Fitness-Level-Predictor-Project', mlflow=True)

            logger.info('Experiment Tracking Started')
            
            mlflow.set_experiment('Fitness-Level-Predictor')

            with mlflow.start_run():
                mlflow.log_metric('Training R2_Score', train_r2)
                mlflow.log_metric('Testing R2_Score', test_r2)

                mlflow.tensorflow.log_model(model, 'model', registered_model_name='Fitness-Predictor-Model')

            logger.info('Experiment Tracking Completed Successfully')


        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
