import os
import sys

import dill
from sqlalchemy import exc

from src.utils.exception import CustomException
from src.utils.logger import get_logger

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

logger = get_logger('utils')

def save_model(file_path: str, model):
    '''
    saves the model to desired file path

    Parameters:
        file_path (str): Path where the model have to be saved
        model (model): Model which have to be saved

    Returns:
        None
    '''

    try:
        logger.info('Saving Model to path')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(model, file)

        logger.info('Model Saved Successfully')
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def load_object(file_path):
    '''
    This function will take the file path and returns the loaded model.

    Args:
        file_path: file path where the model have to be loaded.

    Returns:
        loaded model.
    '''

    try:
        logger.info('Loading our model from path')
        
        with open(file_path, 'rb') as file:
            model = dill.load(file)

        logger.info('Model Loaded Successfully')

        return model
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def get_callbacks():
    '''
    This method will return callbacks for our Neural Network model.

    Returns:
        EarlyStopping and TensorBoard callbacks.
    '''

    try:
        logger.info('Initiating Callbacks for our model')

        TB_PATH = os.path.join('NN_Logs')

        os.makedirs(TB_PATH, exist_ok=True)

        es_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        tb_callback = TensorBoard(log_dir='NN_Logs/training_2', histogram_freq=1)

        logger.info('Callbacks Created')

        return (
            es_callback,
            tb_callback
        )

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
