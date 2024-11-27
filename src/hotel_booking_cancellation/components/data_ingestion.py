import os
import sys
from pandas import DataFrame

from src.hotel_booking_cancellation.constants import DATASET_NAME
from src.hotel_booking_cancellation.constants import SCHEMA_FILE_PATH
from src.hotel_booking_cancellation.utils.main_utils import read_yaml_file
from src.hotel_booking_cancellation.entity.config_entity import DataIngestionConfig
from src.hotel_booking_cancellation.entity.artifact_entity import DataIngestionArtifact
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.data_access.hotel_booking_data import HotelBookingData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize the DataIngestion class with the provided configuration.

        :param data_ingestion_config: Configuration for data ingestion. If not provided, default configuration is used.

        Raises:
            HotelBookingException: If an error occurs during initialization. The exception message and the original error are provided.
        """
        logging.info("_"*100)
        logging.info("| | Started Data Ingestion Stage:")
        logging.info("- "*50)
        try:

            self.dataset_name = DATASET_NAME
            self.data_ingestion_config = data_ingestion_config
            # Read the schema configuration for sensitive columns and other details
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise HotelBookingException(f"Error during DataIngestion initialization: {str(e)}", sys) from e

    def export_data_into_artifact_data(self) -> DataFrame:
        try:
            logging.info("Exporting data from MySQL Database")
            hotel_booking_data = HotelBookingData()
            dataframe = hotel_booking_data.export_data_as_dataframe(dataset_name=self.dataset_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            artifact_data_file_path = self.data_ingestion_config.data_file_path
            dir_path = os.path.dirname(artifact_data_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving exported data into feature store file path: {artifact_data_file_path}")
            dataframe.to_csv(artifact_data_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise HotelBookingException(f"Error in export_data_into_feature_store: {str(e)}", sys) from e



    def drop_sensitive_columns(self, dataframe: DataFrame) -> DataFrame:
        """
        Method Name :   drop_sensitive_columns
        Description :   This method drops sensitive columns from the dataframe.

        Output      :   DataFrame with sensitive columns removed.
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered drop_sensitive_columns method of DataIngestion class")

        try:
            # Retrieve sensitive columns from schema config
            sensitive_columns = self._schema_config.get("sensitive_columns", [])
            if sensitive_columns:
                logging.info(f"Removing sensitive columns: {sensitive_columns}")
                dataframe = dataframe.drop(columns=sensitive_columns, errors="ignore")

            logging.info("Exited drop_sensitive_columns method of DataIngestion class")
            return dataframe
        except Exception as e:
            raise HotelBookingException(e, sys) from e
        


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline

        Output      :   Ingested data is saved as a CSV file.
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.export_data_into_artifact_data()
            logging.info("Got the data from MySQL Database")

            dataframe = self.drop_sensitive_columns(dataframe)
            logging.info("Dropped sensitive columns from the dataframe")

            ingested_data_file_path = self.data_ingestion_config.ingested_file_path
            dir_path = os.path.dirname(ingested_data_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving ingested data into file path: {ingested_data_file_path}")
            dataframe.to_csv(ingested_data_file_path, index=False, header=True)

            logging.info("Exited initiate_data_ingestion method of DataIngestionClass")

            data_ingestion_artifact = DataIngestionArtifact(ingest_file_path=ingested_data_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e