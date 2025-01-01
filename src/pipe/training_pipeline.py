from src.utils.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger('training-pipeline')

def main():
    logger.info('Training Pipeline started')

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.load_data()

    data_preprocessing = DataPreprocessing()
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(train_path, test_path)

    model_trainer = ModelTrainer()
    model_path = model_trainer.start_training(X_train, X_test, y_train, y_test)

    model_evaluation = ModelEvaluation()
    model_evaluation.evaluate(model_path, X_train, X_test, y_train, y_test)

    logger.info('Training Pipeline Completed Successfully')


if __name__ == '__main__':
    main()