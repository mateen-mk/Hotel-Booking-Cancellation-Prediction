import sys
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.components.data_ingestion import DataIngestion
from src.hotel_booking_cancellation.components.data_validation import DataValidation
from src.hotel_booking_cancellation.components.data_preprocessing import DataPreprocessing

from src.hotel_booking_cancellation.entity.config_entity import (DataIngestionConfig,
                                                                 DataValidationConfig,
                                                                 DataPreprocessingConfig)

from src.hotel_booking_cancellation.entity.artifact_entity import (DataIngestionArtifact, 
                                                                   DataValidationArtifact,
                                                                   DataPreprocessingArtifact)

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()


    # Data Ingestion Function
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
            logging.info("Getting the data from MySQL Database")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from MySQL Database")
            logging.info(
                "Exited the start_data_ingestion method of TrainingPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e


    # Data Validation Function
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data validation component
        """
        logging.info("Entered the start_data_validation method of TrainingPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainingPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise HotelBookingException(e, sys) from e


    # Data Preprocessing Function
    def start_data_preprocessing(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataPreprocessingArtifact:
        """
        This method of TrainingPipeline class is responsible for starting data preprocessing component
        """
        logging.info("Entered the start_data_preprocessing method of TrainingPipeline class")
        
        try:
            data_preprocessing = DataPreprocessing(data_ingestion_artifact=data_ingestion_artifact,
                                                   data_validation_artifact=data_validation_artifact,
                                                   data_preprocessing_config=self.data_preprocessing_config)
            data_preprocessing_artifact = data_preprocessing.initiate_data_preprocessing()
            return data_preprocessing_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e
        


    # Run the training pipeline
    def run_pipeline(self, ) -> None:
        """
        This method of TrainingPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_preprocessing_artifact = self.start_data_preprocessing(data_ingestion_artifact, data_validation_artifact)
        except Exception as e:
            raise HotelBookingException(e, sys) from e