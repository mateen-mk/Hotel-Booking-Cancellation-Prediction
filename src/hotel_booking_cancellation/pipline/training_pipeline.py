import sys
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.components.data_ingestion import DataIngestion
from src.hotel_booking_cancellation.components.data_validation import DataValidation
from src.hotel_booking_cancellation.components.data_preprocessing import DataPreprocessing
from src.hotel_booking_cancellation.components.model_trainer import ModelTrainer

from src.hotel_booking_cancellation.entity.config_entity import (DataIngestionConfig,
                                                                 DataValidationConfig,
                                                                 DataPreprocessingConfig,
                                                                 ModelTrainerConfig)

from src.hotel_booking_cancellation.entity.artifact_entity import (DataIngestionArtifact, 
                                                                   DataValidationArtifact,
                                                                   DataPreprocessingArtifact,
                                                                   ModelTrainerArtifact)

class TrainingPipeline:
    def __init__(self):
        logging.info("* "*50)
        logging.info("- - - - - Started Training Pipeline - - - - -")
        logging.info("* "*50)
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()


    # Data Ingestion Function
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("")
            logging.info("")
            logging.info("$ Entered start_data_ingestion method of TrainingPipline Class:")
            
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("- "*50)
            logging.info("- - - Data Ingested Successfully! - - -")

            logging.info("")
            logging.info("! ! ! Exited the start_data_ingestion method of TrainingPipeline class:")
            logging.info("_"*100)
            return data_ingestion_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e


    # Data Validation Function
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data validation component
        """
        try:
            logging.info("")
            logging.info("")
            logging.info("$ Entered start_data_validation method of TrainingPipline Class:")

            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("- "*50)
            logging.info("- - - Data Validated Successfully! - - -")

            logging.info("")
            logging.info("! ! ! Exited the start_data_validation method of TrainingPipeline class:")
            logging.info("_"*100)

            return data_validation_artifact

        except Exception as e:
            raise HotelBookingException(e, sys) from e


    # Data Preprocessing Function
    def start_data_preprocessing(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataPreprocessingArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data preprocessing component
        """        
        try:
            logging.info("")
            logging.info("")
            logging.info("$ Entered start_data_preprocessing method of TrainingPipline Class:")

            data_preprocessing = DataPreprocessing(data_ingestion_artifact=data_ingestion_artifact,
                                                   data_validation_artifact=data_validation_artifact,
                                                   data_preprocessing_config=self.data_preprocessing_config)
            data_preprocessing_artifact = data_preprocessing.initiate_data_preprocessing()

            logging.info("- "*50)
            logging.info("- - - Data Preprocessed Successfully! - - -")

            logging.info("")
            logging.info("! ! ! Exited the start_data_preprocessing method of TrainingPipeline class:")
            logging.info("_"*100)
            return data_preprocessing_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e
        


    # Model Training Function
    def start_model_trainer(self, data_preprocessing_artifact: DataPreprocessingArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            logging.info("")
            logging.info("")
            logging.info("$ Entered start_model_trainer method of TrainingPipline Class:")
        
            model_trainer = ModelTrainer(data_preprocessing_artifact=data_preprocessing_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("- "*50)
            logging.info("- - - Model Trained Successfully! - - -")

            logging.info("")
            logging.info("! ! ! Exited the start_model_trainer method of TrainingPipeline class:")
            logging.info("_"*100)
            return model_trainer_artifact

        except Exception as e:
            raise HotelBookingException(e, sys)


    # Run the training pipeline
    def run_pipeline(self, ) -> None:
        """
        This method of TrainingPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_preprocessing_artifact = self.start_data_preprocessing(data_ingestion_artifact, data_validation_artifact)
            model_training_artifact = self.start_model_trainer(data_preprocessing_artifact)
        except Exception as e:
            raise HotelBookingException(e, sys) from e