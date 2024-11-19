import sys
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.components.data_ingestion import DataIngestion

from src.hotel_booking_cancellation.entity.config_entity import DataIngestionConfig
from src.hotel_booking_cancellation.entity.artifact_entity import DataIngestionArtifact



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


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
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e



    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise HotelBookingException(e, sys) from e