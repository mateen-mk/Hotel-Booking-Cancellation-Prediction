import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.hotel_booking_cancellation.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.hotel_booking_cancellation.entity.config_entity import DataPreprocessingConfig
from src.hotel_booking_cancellation.entity.artifact_entity import DataPreprocessingArtifact, DataIngestionArtifact, DataValidationArtifact
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file




class DataPreprocessing:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_preprocessing_config: DataPreprocessingConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_preprocessing_config: configuration for data preprocessing
        """
        try:

            logging.info("- "*50)
            logging.info("- - - - - Started Data Preprocessing Stage - - - - -")
            logging.info("- "*50)

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_preprocessing_config = data_preprocessing_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise HotelBookingException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HotelBookingException(e, sys)
        



    # Function for Dropping Directly Related Features
    def drop_directly_related_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name :   drop_directly_related_features
        Description :   Drops columns that are directly related to the target or cause data leakage,
                        as defined in the schema.yaml configuration.
        
        Output      :   Returns a DataFrame with specified columns removed.
        """
        try:
            logging.info("Fetching directly related features to drop from schema.yaml")
            drop_columns = self._schema_config.get('drop_columns', [])
            logging.info(f"Columns to drop identified: {drop_columns}")

            # Dropping the columns
            df = df.drop(columns=drop_columns, errors='ignore')
            logging.info(f"Successfully dropped directly related columns: {drop_columns}")
            return df

        except Exception as e:
            raise HotelBookingException(f"Error in drop_directly_related_features: {str(e)}",sys) from e
                


    # Function for Handling Missing Values
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name :   handle_missing_values
        Description :   Handles missing values by logging columns with missing data 
                        and filling them with 0.
        
        Output      :   Returns a DataFrame with missing values handled.
        """
        try:
            # Check for missing values
            missing_columns = df.columns[df.isnull().any()].tolist()
            
            if missing_columns:
                # Log the columns with missing values
                logging.info(f"Missing values found in columns: {missing_columns}")
                
                # Fill missing values with 0
                df[missing_columns] = df[missing_columns].fillna(0)
                logging.info(f"Filled missing values in columns: {missing_columns} with 0")
            
            return df
        
        except Exception as e:
            raise HotelBookingException(f"Error while handling missing values: {str(e)}", sys) from e



    # Function for Handling Noisy Data
    def handle_noisy_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name :   handle_noisy_data
        Description :   Identifies and handles noisy data dynamically using schema.yaml configuration.
        
        Output      :   Returns a cleaned DataFrame with noisy data handled appropriately.
        """
        try:
            logging.info("Fetching noisy value columns from schema.yaml")
            noisy_columns = self._schema_config.get('noisy_values_columns', [])
            logging.info(f"Noisy value columns identified: {noisy_columns}")

            # Initialize a dictionary to track noisy data conditions and actions
            noisy_conditions = {
                'adr':      df['adr'] < 0,
                'adults':   df['adults'] == 0,
                'children': df['children'] == 10,
                'babies':   df['babies'] == 10,
            }

            # Filter noisy_conditions to only include columns present in schema.yaml
            noisy_conditions = {col: condition for col, condition in noisy_conditions.items() if col in noisy_columns}

            # Log the counts of noisy data
            noisy_data_count = {key: df[condition].shape[0] for key, condition in noisy_conditions.items()}
            for feature, count in noisy_data_count.items():
                if count > 0:
                    logging.info(f"Found {count} noisy rows in '{feature}'")

            # Handle noisy data based on schema definitions
            if 'adr' in noisy_columns and noisy_data_count.get('adr', 0) > 0:
                median_adr = df[df['adr'] >= 0]['adr'].median()
                df.loc[df['adr'] < 0, 'adr'] = median_adr
                logging.info(f"     Replaced negative ADR values with median: {median_adr}")

            if 'adults' in noisy_columns and noisy_data_count.get('adults', 0) > 0:
                df = df[df['adults'] > 0]
                logging.info(f"     Removed rows where adults == 0")

            if 'children' in noisy_columns and noisy_data_count.get('children', 0) > 0:
                df = df[df['children'] != 10]
                logging.info(f"     Removed rows where children == 10")

            if 'babies' in noisy_columns and noisy_data_count.get('babies', 0) > 0:
                df = df[df['babies'] != 10]
                logging.info(f"     Removed rows where babies == 10")
            
            logging.info("Noisy data handling completed successfully")
            return df

        except Exception as e:
            raise HotelBookingException(f"Error in handle_noisy_data: {str(e)}", sys) from e



    # Function for Getting Data Preprocessing object
    def get_data_preprocessing_object(self) -> Pipeline:
        """
        Method Name :   get_data_preprocessing_object
        Description :   This method creates and returns a data preprocessing object for the data
        
        Output      :   data preprocessing object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_preprocessing_object method of DataPreprocessing class")

        try:

            onehot_encoding_columns = self._schema_config.get('onehot_encoding_columns',[])
            ordinal_encoding_columns = self._schema_config.get('ordinal_encoding_columns',[])
            scaling_features = self._schema_config.get('scaling_columns',[])
            
            logging.info('Preprocessing columns are fetched from schema.yaml')

            # month_order for ordinal encoding so it can be encoded in proper order
            month_order = ['January', 'February', 'March', 'April', 
                            'May', 'June', 'July', 'August', 
                            'September', 'October', 'November', 'December']
            
            scaler = MinMaxScaler()
            onehot_encoder = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder(categories=[month_order])

            logging.info("Initialized MinMaxScaler, OneHotEncoder, OrdinalEncoder")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", onehot_encoder, onehot_encoding_columns),
                    ("OrdinalEncoder", ordinal_encoder, ordinal_encoding_columns),
                    ("MinMaxScaler", scaler, scaling_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info("Exited get_data_preprocessor_object method of DataPreprocessor class")
            return preprocessor

        except Exception as e:
            raise HotelBookingException(f"Error in get_data_preprocessor_object: {str(e)}", sys) from e



    # Funcion for Separating Target feature from Dataset
    def separate_features_and_target(self, df: pd.DataFrame, target_column: str) -> tuple:
        """
        Method Name :   separate_features_and_target
        Description :   Separates independent features and dependent (target) feature from the DataFrame.
        
        Input       :   df            -> The input DataFrame (train/test).
                    target_column  -> The name of the target column in the DataFrame.
        
        Output      :   tuple         -> A tuple containing the independent features DataFrame and the target feature series.
        """
        try:
            logging.info(f"Separating independent features and target feature for column: {target_column}")
            
            # Separating independent features (X) and target feature (y)
            input_features_df = df.drop(columns=[target_column], axis=1)
            target_feature_df = df[target_column]
            
            logging.info("Independent features and target feature separated successfully")
            return input_features_df, target_feature_df

        except Exception as e:
            raise HotelBookingException(f"Error in separate_features_and_target: {str(e)}", sys) from e



    def initiate_data_preprocessing(self, ) -> DataPreprocessingArtifact:
        """
        Method Name :   initiate_data_preprocessing
        Description :   This method initiates the data preprocessing component for the pipeline 
        
        Output      :   data preprocessing steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:

                # Initialize preprocessor object
                logging.info("Starting data preprocessing")
                preprocessor = self.get_data_preprocessing_object()
                logging.info("Got the preprocessor object")


                # Fetching Train and Test datasets
                logging.info("Start Fetching Train dataset")
                train_df = DataPreprocessing.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                
                logging.info("Start Fetching Test dataset")
                test_df = DataPreprocessing.read_data(file_path=self.data_ingestion_artifact.test_file_path)
                logging.info("Got the Train and Test data")


                # Drop directly related features
                logging.info("Start Dropping directly related features from Train dataset")
                train_df = self.drop_directly_related_features(train_df)

                logging.info("Start Dropping directly related features from Test dataset")
                test_df = self.drop_directly_related_features(test_df)
                logging.info("Dropped directly related features from Train and Test datasets")


                # Handle missing values in Train and Test datasets
                logging.info("Start Handling missing values in Training dataset")
                train_df = self.handle_missing_values(train_df)
                
                logging.info("Start Handling missing values in Testing dataset")
                test_df = self.handle_missing_values(test_df)
                logging.info("Handled Missing values in Train and Test datasets")


                # Handle noisy data in Train and Test datasets
                logging.info("Start Handling noisy data in Train and dataset")
                train_df = self.handle_noisy_data(train_df)

                logging.info("Start Handling noisy data in Test and dataset")
                test_df = self.handle_noisy_data(test_df)
                logging.info("Handled Noisy data in Train and Test datasets")
                

                # Seprating Independent Features and Dependent Feature 
                logging.info("Start Seprating Independent Features and Dependent Feature for Train dataset")
                input_feature_train_df, target_feature_train_df = self.separate_features_and_target(train_df, target_column=TARGET_COLUMN)

                logging.info("Start Seprating Independent Features and Dependent Feature for Test dataset")
                input_feature_test_df, target_feature_test_df = self.separate_features_and_target(test_df, target_column=TARGET_COLUMN)
                logging.info("Independent Features and Dependent Feature are separated from Train and Test datasets")



                # Appliying Preprocessing object on Training and Testing Datasets
                logging.info("Start Appliying Preprocessing object on Train dataset")
                preprocessed_input_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info("Start Appliying Preprocessing object on Test dataset")
                preprocessed_input_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Applied Preprocessing object on Train and Test datasets")


                # Combining train and test arrays with target feature (train_arr, test_arr)
                logging.info("Start Combining train features and target feature as train array")
                train_arr = np.c_[preprocessed_input_train_arr, np.array(target_feature_train_df)]

                logging.info("Start Combining test features and target feature as test array")
                test_arr = np.c_[preprocessed_input_test_arr, np.array(target_feature_test_df)]
                logging.info("Train and test arrays created successfully")


                # Saving preprocessor and train, test arrays
                logging.info("Start Saving preprocessor object and train, test arrays")
                save_object(self.data_preprocessing_config.preprocessed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_preprocessing_config.preprocessed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_preprocessing_config.preprocessed_test_file_path, array=test_arr)
                logging.info("Saved the preprocessor object and train, test arrays")

                logging.info(
                    "Exited initiate_data_preprocessor method of DataPreprocessor class"
                )

                data_preprocessor_artifact = DataPreprocessingArtifact(
                    preprocessed_object_file_path=self.data_preprocessing_config.preprocessed_object_file_path,
                    preprocessed_train_file_path=self.data_preprocessing_config.preprocessed_train_file_path,
                    preprocessed_test_file_path=self.data_preprocessing_config.preprocessed_test_file_path
                )
                return data_preprocessor_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise HotelBookingException(e, sys) from e