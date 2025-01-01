from math import e
import os
import sys

from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import get_callbacks

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import R2Score

logger = get_logger('model-trainer')

@dataclass
class ModelPaths:
    model_path: str = os.path.join('model', 'model.h5')

class ModelTrainer:
    def __init__(self):
        self.model_path = ModelPaths()

    def get_model_architecture(self, input_shape):
        '''
        This method will build our Neural Network and returns it.

        Args:
            input_shape: input shape required for our model building

        Returns:
            Neural Network Object.
        '''

        try:
            logger.info('Building our Neural Network Architecture')

            model = Sequential()

            model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
            model.add(Dropout(0.5))

            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))

            model.add(Dense(1))

            logger.info('Compiling Our Model')

            model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[R2Score()])

            logger.info('Model Builded Successfully')

            return model
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def start_training(self, X_train, X_test, y_train, y_test):
        '''
        This method will intiate training of our model and returns us the trained saved model path.

        Args:
            X_train: our X-Training data.
            X_test: X-Testing data.
            y_train: y_training data.
            y_test: y_testing data.

        Returns:
            Trained Saved model path.
        '''

        try:
            logger.info('Model Training Started')

            input_shape = X_train.shape[1]

            model = self.get_model_architecture(input_shape)

            es_callback, tb_callback = get_callbacks()

            logger.info('Model Fitting Started')

            model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=10,
                callbacks=[es_callback, tb_callback]
            )

            logger.info('Model Training Completed Sucessfully')

            logger.info('Saving our model')
            model.save(self.model_path.model_path)
            logger.info('Model Saved Sucessfully')

            return self.model_path.model_path
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)