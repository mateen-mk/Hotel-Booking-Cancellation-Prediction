import os
import sys

import dill
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging


class YamlUtils:
    """
    Class Name :   YamlUtils
    Description :   This class provides utility methods for reading and writing YAML files.
    """

    @staticmethod
    def read_yaml_file(file_path: str) -> dict:
        """
        Read a YAML file and return its content as a dictionary.

        Parameters:
        file_path (str): The path to the YAML file to be read.

        Returns:
        dict: The content of the YAML file as a dictionary.

        Raises:
        USvisaException: If an error occurs while reading the YAML file.
        """
        try:
            with open(file_path, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise HotelBookingException(e, sys) from e
        

    @staticmethod
    def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
        """
        Write a dictionary to a YAML file.
        
        Parameters:
        file_path (str): The path to the YAML file to be written.
        content (object): The dictionary to be written to the YAML file.
        replace (bool, optional): If True, overwrite the file if it already exists. Defaults to False.
        
        Raises:
        USvisaException: If an error occurs while writing the YAML file.
        """
        try:
            if replace:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                yaml.dump(content, file)
        except Exception as e:
            raise HotelBookingException(e, sys) from e
        


class DataUtils:
    """
    Class Name :   DataUtils
    Description :  This class contains methods to perform data operations.
    """

    # Function for Reading data from a file
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read data from a CSV file and return it as a DataFrame.

        :param file_path: The file path of the CSV file to read.
        :return: A DataFrame containing the data from the CSV file.
        :raises HotelBookingException: If an error occurs while reading the data.
        """
        try:
            logging.info(f"Reading data from {file_path}")
            dataframe = pd.read_csv(file_path)
            logging.info(f"Successfully read data from {file_path}")
            return dataframe
        except Exception as e:
            raise HotelBookingException(f"Error reading data from {file_path}: {str(e)}", sys) from e


    # Function for saving data to a file
    @staticmethod
    def save_data(dataframe: pd.DataFrame, file_path: str) -> None:
        """
        Save the given DataFrame to a CSV file at the specified file path.

        :param dataframe: The DataFrame to save.
        :param file_path: The file path where the DataFrame will be saved.
        :raises HotelBookingException: If an error occurs while saving the data.
        """
        try:
            logging.info(f"Saving data to {file_path}")
            # Ensure the directory exists
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the DataFrame to a CSV file
            dataframe.to_csv(file_path, index=False, header=True)
            logging.info(f"Successfully saved data to {file_path}")
        except Exception as e:
            raise HotelBookingException(f"Error saving data to {file_path}: {str(e)}", sys) from e
    


class TrainTestSplitUtils:
    """
    Class Name: TrainTestSplitUtils
    Description :  This class contains methods to perform train-test split operations.
    """

    # split data into three datasets train, test and validation datasets
    @staticmethod
    def split_data(dataframe: pd.DataFrame) -> tuple:
        """
        Method Name: split_data
        Description :   Splits the given DataFrame into three datasets: train, test, and validation.
        
        Input       :   dataframe      -> The input DataFrame (train/test).
        
        Output      :   tuple         -> A tuple containing the training DataFrame, the testing DataFrame, and the validation DataFrame.
        """ 
        try:
            # Splitting and saving the preprocessed dataset
            logging.info("Start splitting the preprocessed dataset")

            # Split into train and remaining (test + validation)
            train_data, temp_data = train_test_split(
                dataframe, 
                test_size=0.30,  # 30% for test + validation
                random_state=42, 
                shuffle=True
            )

            # Split remaining into test and validation
            test_data, validation_data = train_test_split(
                temp_data, 
                test_size=0.50,  # Split 30% into 50% test and 50% validation
                random_state=42, 
                shuffle=True
            )

            return train_data, test_data, validation_data
        
        except Exception as e:
            raise HotelBookingException(f"Error in split_data: {str(e)}", sys) from e


    # Funcion for Separating Target feature from Dataset
    @staticmethod
    def separate_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple:
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
            X = dataframe.drop(columns=[target_column], axis=1)
            y = dataframe[target_column]
            
            logging.info("Independent features and target feature separated successfully")
            return X, y

        except Exception as e:
            raise HotelBookingException(f"Error in separate_features_and_target: {str(e)}", sys) from e
    

    @staticmethod
    def train_test_split_for_data_validation(dataframe: pd.DataFrame, test_size: float) -> tuple:
        """
        Perform train-test split on the given DataFrame for data validation.

        :param dataframe: The DataFrame to split.
        :param test_size: The proportion of the dataset to include in the test split.
        :return: A tuple containing the training set and the testing set.
        """
        try:
            train_set, test_set = train_test_split(dataframe, test_size=test_size, random_state=42)
            return train_set, test_set
        except Exception as e:
            raise HotelBookingException(e, sys) from e


    @staticmethod
    def train_test_split_for_model_building(dataframe: pd.DataFrame, test_size: float, target_column: str) -> tuple:
        """
        Perform train-test split on the given DataFrame with stratification for model training.

        :param dataframe: The DataFrame to split.
        :param test_size: The proportion of the dataset to include in the test split.
        :param target_column: The name of the target column to stratify on.
        :return: A tuple containing the training set and the testing set.
        """
        try:
            # Separate the features and the target column
            X = dataframe.drop(columns=[target_column])
            y = dataframe[target_column]

            # Perform train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

            # # Combine the features and target columns back into DataFrames
            # train_set = pd.concat([X_train, y_train], axis=1)
            # test_set = pd.concat([X_test, y_test], axis=1)

            return X_train, X_test, y_train, y_test 
        except Exception as e:
            raise HotelBookingException(e, sys) from e


    @staticmethod
    def train_test_split_for_tuning(X_train: pd.DataFrame, y_train: pd.Series, test_size: float) -> tuple:
        """
        Perform train-test split on the given training data for hyperparameter tuning.

        :param X_train: The training features DataFrame.
        :param y_train: The training target Series.
        :param test_size: The proportion of the dataset to include in the test split.
        :return: A tuple containing the tuning set features and target.
        """
        try:
            # Perform train-test split for hyperparameter tuning
            X_tune, _, y_tune, _ = train_test_split(X_train, y_train, test_size=test_size)
            return X_tune, y_tune
        except Exception as e:
            raise HotelBookingException(e, sys) from e



class ObjectUtils:
    """
    Class Name: ObjectUtils
    Description :  This class contains utility methods for working with objects.
    """
    
    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """
        Save an object to a file using the dill module.
        
        Parameters:
        file_path (str): The path to the file to be written.
        obj (object): The object to be written to the file.
        
        Raises:
        USvisaException: If an error occurs while saving the object.
        """
        logging.info("Entered the save_object method of utils")

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

            logging.info("Exited the save_object method of utils")

        except Exception as e:
            raise HotelBookingException(e, sys) from e
        

    @staticmethod
    def load_object(file_path: str) -> object:
        """
        Load an object from a file using the dill module.
        
        Parameters:
        file_path (str): The path to the file containing the object to be loaded.
        
        Returns:
        object: The loaded object from the file.
        
        Raises:
        USvisaException: If an error occurs while loading the object.
        """
        logging.info("Entered the load_object method of utils")

        try:

            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)

            logging.info("Exited the load_object method of utils")

            return obj

        except Exception as e:
            raise HotelBookingException(e, sys) from e
    
    

