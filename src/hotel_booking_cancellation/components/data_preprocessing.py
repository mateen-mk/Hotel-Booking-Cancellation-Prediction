import sys

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.exception import HotelBookingException

from src.hotel_booking_cancellation.entity.config_entity import DataPreprocessingConfig
from src.hotel_booking_cancellation.entity.artifact_entity import (DataPreprocessingArtifact,
                                                                   DataIngestionArtifact,
                                                                   DataValidationArtifact)

from src.hotel_booking_cancellation.constants import SCHEMA_FILE_PATH
from src.hotel_booking_cancellation.constants import MONTH_ORDER
from src.hotel_booking_cancellation.utils.main_utils import (DataUtils, 
                                                             ObjectUtils, 
                                                             YamlUtils,
                                                             TrainTestSplitUtils)




class DataPreprocessing:
    """
    Class Name: DataPreprocessing
    Description: Performs Data preprocessing and returns preprocessing object (preprocessing.pkl) and preprocessed data (preprocessed.csv)
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_preprocessing_config: DataPreprocessingConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_preprocessing_config: configuration for data preprocessing
        """
        try:

            logging.info("_"*100)
            logging.info("\n| | Started Data Preprocessing Stage:")
            logging.info("- "*50)


            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_preprocessing_config = data_preprocessing_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = YamlUtils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
     
        except Exception as e:
            logging.error(f"Error in DataPreprocessing initialization: {str(e)}")
            raise HotelBookingException(f"Error during DataPreprocessing initialization: {str(e)}", sys) from e



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
            logging.error(f"Error in drop_directly_related_features: {str(e)}")
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
            logging.error(f"Error in handle_missing_values: {str(e)}")
            raise HotelBookingException(f"Error in handle_missing_values: {str(e)}", sys) from e



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
                logging.info("     Removed rows where adults == 0")

            if 'children' in noisy_columns and noisy_data_count.get('children', 0) > 0:
                df = df[df['children'] != 10]
                logging.info("     Removed rows where children == 10")

            if 'babies' in noisy_columns and noisy_data_count.get('babies', 0) > 0:
                df = df[df['babies'] != 10]
                logging.info("     Removed rows where babies == 10")
            
            logging.info("Noisy data handling completed successfully")
  
            return df

        except Exception as e:
            logging.error(f"Error in handle_noisy_data: {str(e)}")
            raise HotelBookingException(f"Error in handle_noisy_data: {str(e)}", sys) from e



    # Function for Getting Data Preprocessing fucntions and saving preprocessing object (prerprocessing.pkl)
    def get_preprocessing_functions(self) -> Pipeline: #tuple:
        logging.info("Entered get_data_preprocessing_functions method of DataPreprocessing class")
        try:
            # Fetch schema config
            onehot_encoding_columns = self._schema_config.get('onehot_encoding_columns', [])
            label_encoding_columns = self._schema_config.get('label_encoding_columns', [])
            scaling_columns = self._schema_config.get('scaling_columns', [])
            logging.info('Preprocessing columns fetched from schema.yaml')

            # Define individual transformer functions
            def label_encoding_function(data: pd.DataFrame) -> pd.DataFrame:
                month_order = MONTH_ORDER
                columns = label_encoding_columns if isinstance(label_encoding_columns, list) else [label_encoding_columns]
                for col in columns:
                    data[col] = data[col].apply(lambda x: month_order.index(x) + 1)
                logging.info(f"Label encoding applied on columns: {columns}")
                return data

            def onehot_encoding_function(data: pd.DataFrame) -> pd.DataFrame:
                data = pd.get_dummies(data, columns=onehot_encoding_columns, drop_first=True)
                data = data.astype(int)
                logging.info(f"One-hot encoding applied on columns: {onehot_encoding_columns}")
                return data
            
            def scaling_function(data: pd.DataFrame) -> pd.DataFrame:
                data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
                logging.info(f"Scaling applied on columns: {scaling_columns}")
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


            # Save the pipeline
            ObjectUtils.save_object(self.data_preprocessing_config.preprocessed_object_file_path, data_pipeline)
            logging.info("Preprocessing object (preprocessing.pkl) saved successfully.")

            return data_pipeline
            # return label_encoding_function, onehot_encoding_function, scaling_function

        except Exception as e:
            logging.error(f"Error in get_data_preprocessor_object: {str(e)}")
            raise HotelBookingException(f"Error in get_data_preprocessor_object: {str(e)}", sys) from e
        


    # Putting all together and initializing the data preprocessing function
    def initiate_data_preprocessing(self) -> DataPreprocessingArtifact:
        """
        Method Name :   initiate_data_preprocessing
        Description :   This method initiates the data preprocessing component for the pipeline 
        
        Output      :   data preprocessing steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:

                # Initialize preprocessing functions
                logging.info("Start initializing data preprocessing functions")
                preprocessor = self.get_preprocessing_functions()
                # label_enocder, onehot_enoder, scaler = self.get_preprocessing_functions() 
                logging.info("Initialized data preprocessing functions")


                # Fetching dataset
                logging.info("Start Fetching dataset")
                df = DataUtils.read_data(file_path=self.data_ingestion_artifact.data_file_path)
                logging.info("Fetched dataset")


                # Drop directly related features
                logging.info("Start Dropping directly related features from dataset")
                df = self.drop_directly_related_features(df)
                logging.info("Dropped directly related features from dataset")


                # Handle missing values in Train and Test datasets
                logging.info("Start Handling missing values in dataset")
                df = self.handle_missing_values(df)
                logging.info("Handled Missing values in dataset")

                
                # Handle noisy data in Train and Test datasets
                logging.info("Start Handling noisy data in dataset")
                df = self.handle_noisy_data(df)
                logging.info("Handled Noisy data in dataset")
  
                
                # Appliying Data Preprocessing fucntions on Dataset
                logging.info("Start Appliying Data Preprocessing fucntions on Dataset")
                
                # label encoding
                logging.info("\tAppliying label_encoding") 
                df = label_enocder(df)
                logging.info("Applied label_encoding") 
                
                # onehot encoding
                logging.info("\tAppliying onehot_encoding") 
                df = onehot_enoder(df)
                logging.info("Applied onehot_encoding")  

                # scaler
                logging.info("\tAppliying scaler") 
                df = scaler(df)
                logging.info("Applied scaler") 


                # Splitting dataset into train, test, validation datasets
                logging.info("Start Splitting dataset into train, test, validation datasets")
                train_df, test_df, validation_df = TrainTestSplitUtils.split_data(df)
                logging.info("Splitted dataset into train, test, validation datasets")
                
                # Logging the dataset sizes
                logging.info(f"\tTrain data size: {len(train_df)}")
                logging.info(f"\tTest data size: {len(test_df)}")
                logging.info(f"\tValidation data size: {len(validation_df)}")


                # Saving preprocessed train dataset
                logging.info("Start Saving preprocessed train dataset")
                DataUtils.save_data(train_df, self.data_preprocessing_config.train_data_file_path)
                logging.info("Saved preprocessed train dataset")

                # Saving preprocessed test dataset
                logging.info("Start Saving preprocessed test dataset")
                DataUtils.save_data(test_df, self.data_preprocessing_config.test_data_file_path)
                logging.info("Saved preprocessed test dataset")

                # Saving preprocessed validation dataset
                logging.info("Start Saving preprocessed validation dataset")
                DataUtils.save_data(validation_df, self.data_preprocessing_config.validation_data_file_path)
                logging.info("Saved preprocessed validation dataset")


                logging.info("Exited initiate_data_preprocessor method of DataPreprocessor class")

                data_preprocessor_artifact = DataPreprocessingArtifact(
                    preprocessed_object_file_path=self.data_preprocessing_config.preprocessed_object_file_path,
                    train_data_file_path=self.data_preprocessing_config.train_data_file_path,
                    test_data_file_path=self.data_preprocessing_config.test_data_file_path,
                    validation_data_file_path=self.data_preprocessing_config.validation_data_file_path
                )

                return data_preprocessor_artifact

            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            logging.error(f"Error in initialize_data_preprocessing: {str(e)}")
            raise HotelBookingException(f"Error in initialize_data_preprocessing: {str(e)}", sys) from e