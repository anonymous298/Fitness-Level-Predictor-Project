import os
import sys

import pandas as pd

from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import load_object

from tensorflow.keras.models import load_model

logger = get_logger('prediction-pipeline')

@dataclass
class PredictionPipePaths:
    preprocessor_path: str = os.path.join('model', 'preprocessor.pkl')
    model_path: str = os.path.join('model', 'model.h5')

class CustomData:
    def __init__(self, 
                 age: int,
                 gender: str,
                 height_cm: float,
                 weight_kg: float,
                 bmi: float,
                 activity_type: str,
                 duration_minutes: int,
                 intensity: str,
                 calories_burned: float,
                 avg_heart_rate: int,
                 resting_heart_rate: int,
                 blood_pressure_systolic: int,
                 blood_pressure_diastolic: int,
                 hours_sleep: float,
                 stress_level: int,
                 daily_steps: int,
                 hydration_level: int,
                 smoking_status: str):
        
        self.age = age
        self.gender = gender
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.bmi = bmi
        self.activity_type = activity_type
        self.duration_minutes = duration_minutes
        self.intensity = intensity
        self.calories_burned = calories_burned
        self.avg_heart_rate = avg_heart_rate
        self.resting_heart_rate = resting_heart_rate
        self.blood_pressure_systolic = blood_pressure_systolic
        self.blood_pressure_diastolic = blood_pressure_diastolic
        self.hours_sleep = hours_sleep
        self.stress_level = stress_level
        self.daily_steps = daily_steps
        self.hydration_level = hydration_level
        self.smoking_status = smoking_status

    def get_data_as_dataframe(self):
        '''
        This method will convert all the columns to a dataframe.

        Returns:
            DataFrame
        '''
        try:
            
            custom_data_input_dict = {
                'age': [self.age],
                'gender': [self.gender],
                'height_cm': [self.height_cm],
                'weight_kg': [self.weight_kg],
                'bmi': [self.bmi],
                'activity_type': [self.activity_type],
                'duration_minutes': [self.duration_minutes],
                'intensity': [self.intensity],
                'calories_burned': [self.calories_burned],
                'avg_heart_rate': [self.avg_heart_rate],
                'resting_heart_rate': [self.resting_heart_rate],
                'blood_pressure_systolic': [self.blood_pressure_systolic],
                'blood_pressure_diastolic': [self.blood_pressure_diastolic],
                'hours_sleep': [self.hours_sleep],
                'stress_level': [self.stress_level],
                'daily_steps': [self.daily_steps],
                'hydration_level': [self.hydration_level],
                'smoking_status': [self.smoking_status]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logger.info('DataFrame Gathered')
            return df

        except Exception as e:
            logger.error('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)

class PredictionPipeline:
    def __init__(self):
        self.prediction_paths = PredictionPipePaths()

    def predict(self, dataframe):
        '''
        This method will take dataframe and returns us the response from the model.

        Args:
            dataframe: our input querie

        Returns:
            model prediction.
        '''

        try:
            logger.info('Prediction Started')

            logger.info('Loading Preprocessor')
            preprocessor = load_object(self.prediction_paths.preprocessor_path)

            logger.info('Loading our model')
            model = load_model(self.prediction_paths.model_path)

            logger.info('Applying Transformation on our Input qurie')
            transformed_input = preprocessor.transform(dataframe)

            logger.info('Creating Prediction')
            
            prediction = model.predict(transformed_input)

            logger.info('Prediction Completed')

            return prediction

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
