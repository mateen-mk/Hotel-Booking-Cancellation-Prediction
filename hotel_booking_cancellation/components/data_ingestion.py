import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from hotel_booking_cancellation.constants import DATASET_NAME
from hotel_booking_cancellation.entity.config_entity import DataIngestionConfig
from hotel_booking_cancellation.entity.artifact_entity import DataIngestionArtifact
from hotel_booking_cancellation.exception import HotelBookingException
from hotel_booking_cancellation.logger import logging
from hotel_booking_cancellation.data_access.hotel_booking_data import HotelBookingData



class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize the DataIngestion class with the provided configuration.

        :param data_ingestion_config: Configuration for data ingestion. If not provided, default configuration is used.

        Raises:
            USvisaException: If an error occurs during initialization. The exception message and the original error are provided.
        """
        try:
            self.dataset_name = DATASET_NAME
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HotelBookingException(e, sys)


    def export_data_into_feature_store(self)-> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from MySQL Database to csv file
        
        Output      :   Data is returned as a pandas DataFrame and saved to a CSV file.
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from MySQL Database")
            hotel_booking_data = HotelBookingData()
            dataframe = hotel_booking_data.export_data_as_dataframe(dataset_name=
                                                                   self.dataset_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path  = self.data_ingestion_config.data_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e:
            raise HotelBookingException(e,sys)


    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Train and test datasets are saved as CSV files.
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise HotelBookingException(e, sys) from e
        

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from MySQL Database")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of DataIngestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e