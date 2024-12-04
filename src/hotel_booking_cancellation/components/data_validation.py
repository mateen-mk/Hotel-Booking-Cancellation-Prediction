import sys
import json

from pandas import DataFrame

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.exception import HotelBookingException

from src.hotel_booking_cancellation.entity.config_entity import DataValidationConfig
from src.hotel_booking_cancellation.entity.artifact_entity import (DataIngestionArtifact, 
                                                                   DataValidationArtifact)

from src.hotel_booking_cancellation.utils.main_utils import (YamlUtils, 
                                                             DataUtils, 
                                                             TrainTestSplitUtils)
from src.hotel_booking_cancellation.constants import (SCHEMA_FILE_PATH, 
                                                      TRAIN_TEST_SPLIT_RATIO)




class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            logging.info("_"*100)
            logging.info("")
            logging.info("| | Started Data Validation Stage:")
            logging.info("- "*50)

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = YamlUtils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            logging.error(f"Error in DataValidation initialization: {str(e)}")
            raise HotelBookingException(f"Error during DataValidation initialization: {str(e)}", sys) from e


    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validates if the number of required columns (excluding sensitive columns) in the DataFrame 
        matches the schema configuration. Logs missing columns for better traceability.
        """
        try:
            # Retrieve schema and sensitive columns from the configuration
            numerical_columns = self._schema_config.get("numerical_columns", [])
            categorical_columns = self._schema_config.get("categorical_columns", [])
            sensitive_columns = self._schema_config.get("sensitive_columns", [])
            
            
            # Combine numerical and categorical columns into required columns
            required_columns = numerical_columns + categorical_columns
            required_columns = [col for col in required_columns if col not in sensitive_columns]

            
            # Identify any missing columns
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            

            # Validation status
            status = len(missing_columns) == 0


            # Logging the validation result and details
            if status:
                logging.info("All required columns are present (excluding sensitive columns).")
            
            else:
                logging.error(f"Missing required columns: {missing_columns}")
                logging.info(f"Columns in DataFrame: {list(dataframe.columns)}")
                logging.info(f"Expected required columns: {required_columns}")
            
            
            return status
        
        except Exception as e:
            logging.error(f"Error in validate_number_of_columns: {str(e)}")
            raise HotelBookingException(f"Error in validate_number_of_columns: {str(e)}", sys) from e



    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            
            for column in self._schema_config.get("numerical_columns",[]):
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            
            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config.get("categorical_columns", []):
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)


            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")


            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True

        except Exception as e:
            logging.error(f"Error in is_column_exist: {str(e)}")
            raise HotelBookingException(f"Error in is_column_exist: {str(e)}", sys) from e        



    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame, ) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df)


            report = data_drift_profile.json()
            json_report = json.loads(report)


            YamlUtils.write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)


            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]


            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]

            return drift_status

        except Exception as e:
            logging.error(f"Error in detect_data_drift: {str(e)}")
            raise HotelBookingException(f"Error in detect_data_drift: {str(e)}", sys) from e



    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline.
        
        Output      :   Returns a DataValidationArtifact object based on validation results.
        On Failure  :   Writes an exception log and then raises an exception.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation process.")


            # Reading dataset
            df = DataUtils.read_data(file_path=self.data_ingestion_artifact.ingest_file_path)
            logging.info("Training and testing datasets loaded successfully.")


            # Step 1: Validate number of columns in training and testing datasets
            status = self.validate_number_of_columns(dataframe=df)
            logging.info(f"All required columns present in dataframe: {status}")
            if not status:
                validation_error_msg += "Required columns are missing in dataframe.\n"


            # Step 2: Validate column existence in training and testing datasets
            status = self.is_column_exist(df=df)
            logging.info(f"Validation of column existence in dataframe: {status}")
            if not status:
                validation_error_msg += "Required or sensitive columns validation failed for dataframe.\n"


            # Consolidate validation status
            validation_status = len(validation_error_msg) == 0


            # Step 3: Split dataset into training and testing datasets
            train_df, test_df = TrainTestSplitUtils.train_test_split_for_data_validation(dataframe=df, test_size=TRAIN_TEST_SPLIT_RATIO)


            # Step 4: Detect dataset drift if validation passes
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)

                if drift_status:
                    logging.warning("Drift detected between training and testing datasets.")
                    validation_error_msg += "Drift detected between training and testing datasets.\n"

                else:
                    logging.info("No drift detected between training and testing datasets.")

            else:
                logging.warning(f"Validation failed with the following errors: {validation_error_msg}")


            # Create and return DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg.strip(),
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data validation artifact generated: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            logging.error(f"Error in initiate_data_validation: {str(e)}")
            raise HotelBookingException(f"Error in initiate_data_validation: {str(e)}", sys) from e