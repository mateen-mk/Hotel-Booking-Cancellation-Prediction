import os
import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.hotel_booking_cancellation.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.hotel_booking_cancellation.entity.config_entity import DataPreprocessingConfig
from src.hotel_booking_cancellation.entity.artifact_entity import DataPreprocessingArtifact, DataIngestionArtifact, DataValidationArtifact
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.utils.main_utils import read_data, separate_features_and_target, save_object, read_yaml_file




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
        


    # Add the target feature in dataset
    @staticmethod
    def add_target_to_preprocessed_features(preprocessed_features: np.ndarray, target_feature: pd.Series) -> pd.DataFrame:
        """
        Method Name: add_target_to_preprocessed_features
        Description: Converts the preprocessed features into a DataFrame, appends the target feature, 
                    and returns the complete DataFrame.
        
        Input:
            - preprocessed_features: Numpy array of preprocessed input features.
            - target_feature: Pandas Series containing target labels.
            - feature_columns: List of column names corresponding to the input features.
        
        Output:
            - DataFrame with preprocessed input features and target feature appended.
        """
        try:
            logging.info("Converting preprocessed features into a DataFrame")
            preprocessed_df = pd.DataFrame(preprocessed_features)
            
            logging.info("Adding target feature to the preprocessed DataFrame")
            preprocessed_df[TARGET_COLUMN] = target_feature.values
            
            logging.info("Successfully added target feature to the preprocessed DataFrame")
            return preprocessed_df
        
        except Exception as e:
            raise HotelBookingException(f"Error while adding target to preprocessed features: {str(e)}", sys) from e




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
    def get_preprocessing_pipeline(self) -> Pipeline:
        logging.info("Entered get_data_preprocessing_object method of DataPreprocessing class")
        try:
            # Fetch schema config
            onehot_encoding_columns = self._schema_config.get('onehot_encoding_columns', [])
            label_encoding_columns = self._schema_config.get('label_encoding_columns', [])
            scaling_columns = self._schema_config.get('scaling_columns', [])
            logging.info('Preprocessing columns fetched from schema.yaml')

            # Define individual transformer functions
            def label_encoding_function(data: pd.DataFrame) -> pd.DataFrame:
                month_order = ['January', 'February', 'March', 'April', 
                            'May', 'June', 'July', 'August', 
                            'September', 'October', 'November', 'December']
                columns = label_encoding_columns if isinstance(label_encoding_columns, list) else [label_encoding_columns]
                for col in columns:
                    data[col] = data[col].apply(lambda x: month_order.index(x) + 1)
                logging.info(f"Label encoding applied on columns: {columns}")
                return data

            def onehot_encoding_function(data: pd.DataFrame) -> pd.DataFrame:
                data = pd.get_dummies(data, columns=onehot_encoding_columns, drop_first=True)
                logging.info(f"One-hot encoding applied on columns: {onehot_encoding_columns}")
                return data

            # Initialize transformers
            label_encoder = FunctionTransformer(label_encoding_function)
            onehot_encoder = FunctionTransformer(onehot_encoding_function)
            scaler = MinMaxScaler()
            logging.info("Initialized MinMaxScaler, Custom One-hot Encoder, Label Encoder")

            # Combine transformers in ColumnTransformer
            transformers = []
            if label_encoding_columns:
                transformers.append(('label_encoder', label_encoder, label_encoding_columns))
            if onehot_encoding_columns:
                transformers.append(('onehot_encoder', onehot_encoder, onehot_encoding_columns))
            if scaling_columns:
                transformers.append(('scaler', scaler, scaling_columns))
            
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

            # Create pipeline
            data_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            logging.info("Pipeline created successfully.")
            return data_pipeline

        except Exception as e:
            logging.error(f"Error in get_data_preprocessor_object: {str(e)}")
            raise HotelBookingException(f"Error in get_data_preprocessor_object: {str(e)}", sys) from e



    # Function to export preprocessed data
    def export_preprocessed_data(self, preprocessed_dataframe: pd.DataFrame, preprocessed_file_path) -> pd.DataFrame:
        """
        Saves the preprocessed data into a specified file path.
        Args:
            preprocessed_dataframe (DataFrame): Preprocessed DataFrame to be saved.
        Returns:
            DataFrame: The preprocessed DataFrame after being saved.
        """
        try:
            logging.info("Exporting preprocessed data to a file.")
            
            # Ensure the directory exists
            dir_path = os.path.dirname(preprocessed_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving preprocessed data to file path: {preprocessed_file_path}")
            preprocessed_dataframe.to_csv(preprocessed_file_path, index=False, header=True)

            return preprocessed_dataframe
        except Exception as e:
            raise HotelBookingException(f"Error in export_preprocessed_data: {str(e)}", sys) from e




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
                preprocessor = self.get_preprocessing_pipeline()
                logging.info("Got the preprocessor object")


                # Fetching Train and Test datasets
                logging.info("Start Fetching Train dataset")
                train_df = read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                
                logging.info("Start Fetching Test dataset")
                test_df = read_data(file_path=self.data_ingestion_artifact.test_file_path)
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
                train_input_feature, train_target_feature = separate_features_and_target(train_df, target_column=TARGET_COLUMN)

                logging.info("Start Seprating Independent Features and Dependent Feature for Test dataset")
                test_input_feature, test_target_feature = separate_features_and_target(test_df, target_column=TARGET_COLUMN)
                logging.info("Independent Features and Dependent Feature are separated from Train and Test datasets")
                

                # Appliying Preprocessing object on Training and Testing Datasets
                logging.info("Start Appliying Preprocessing object on Train dataset")
                train_preprocessed_input_feature = preprocessor.fit_transform(train_input_feature)

                logging.info("Start Appliying Preprocessing object on Test dataset")
                test_preprocessed_input_feature = preprocessor.fit_transform(test_input_feature)
                logging.info("Applied Preprocessing object on Train and Test datasets")
                

                # Adding the target column back to the preprocessed DataFrame
                logging.info("Start Adding target feature to preprocessed Train dataset")
                train_preprocessed_df = self.add_target_to_preprocessed_features(train_preprocessed_input_feature, 
                                                                                 train_target_feature)

                logging.info("Start Adding target feature to preprocessed Test dataset")
                test_preprocessed_df = self.add_target_to_preprocessed_features(test_preprocessed_input_feature, 
                                                                                 test_target_feature)
                logging.info("Added target feature to preprocessed Train and Test datasets")
                

                # Saving preprocessor and train, test datasets
                logging.info("Start Saving preprocessor object and train, test datasets")
                save_object(self.data_preprocessing_config.preprocessed_object_file_path, preprocessor)
                self.export_preprocessed_data(train_preprocessed_df, self.data_preprocessing_config.preprocessed_train_file_path)
                self.export_preprocessed_data(test_preprocessed_df, self.data_preprocessing_config.preprocessed_test_file_path)
                logging.info("Saved the preprocessor object and train, test datasets")

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