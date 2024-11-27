import sys
from typing import Tuple

import importlib
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf  import ModelFactory

from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.utils.main_utils import read_data, separate_features_and_target, load_numpy_array_data, read_yaml_file, load_object, save_object
from src.hotel_booking_cancellation.entity.config_entity import ModelTrainerConfig
from src.hotel_booking_cancellation.entity.artifact_entity import DataPreprocessingArtifact, ModelTrainerArtifact, ClassificationMetrixArtifact
from src.hotel_booking_cancellation.entity.estimator import HotelBookingModel
from src.hotel_booking_cancellation.constants import TARGET_COLUMN

class ModelTrainer:
    def __init__(self, data_preprocessing_artifact: DataPreprocessingArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_preprocessing_config: Configuration for data preprocessing
        """
        logging.info("- "*50)
        logging.info("- - - - - Started Data Preprocessing Stage - - - - -")
        logging.info("- "*50)

        self.data_preprocessing_artifact = data_preprocessing_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: DataFrame, test: DataFrame) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            

            # Load model configuration
            logging.info("Loading model configuration and data")
            model_config = read_yaml_file(self.model_trainer_config.model_config_file_path)


            # Split data into features (X) and target (y)
            X_train, y_train = separate_features_and_target(train, target_column=TARGET_COLUMN)
            X_test, y_test = separate_features_and_target(test, target_column=TARGET_COLUMN)


            best_model = None
            best_accuracy = 0
            metric_artifact = None

            # Iterate through each model in the configuration
            for model_name, model_details in model_config['model_selection'].items():
                model_class_name = model_details['class']
                model_module = model_details['module']
                params = model_details['params']
                search_param_grid = model_details['search_param_grid']

                # Dynamically load the model class
                module = importlib.import_module(model_module)
                model_class = getattr(module, model_class_name)
                
                # Initialize the model
                model = model_class(**params)

                # Apply grid search for hyperparameter tuning
                grid_search = GridSearchCV(model, search_param_grid, cv=model_config['grid_search']['params']['cv'], 
                                        scoring=model_config['grid_search']['params']['scoring'], 
                                        n_jobs=model_config['grid_search']['params']['n_jobs'], 
                                        verbose=model_config['grid_search']['params']['verbose'])
                
                grid_search.fit(X_train, y_train)

                # Get the best model from grid search
                best_model_candidate = grid_search.best_estimator_

                # Evaluate the model
                y_pred = best_model_candidate.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)


                # Save the best model based on accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = best_model_candidate
                    metric_artifact = ClassificationMetrixArtifact(accuracy=accuracy, f1_score=f1, 
                                                                precision_score=precision, recall_score=recall)
                    
                
            if best_model is None:
                raise HotelBookingException("No suitable model found.", sys)


            return best_model, metric_artifact
        
        except Exception as e:
            logging.error(f"Error in get_model_object_and_report: {str(e)}")
            raise HotelBookingException(e, sys) from e





        #     best_model_detail = model_factory.get_best_model(
        #         X=X_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
        #     )
        #     model_obj = best_model_detail.best_model

        #     y_pred = model_obj.predict(X_test)
            
        #     accuracy = accuracy_score(y_test, y_pred) 
        #     f1 = f1_score(y_test, y_pred)  
        #     precision = precision_score(y_test, y_pred)  
        #     recall = recall_score(y_test, y_pred)
        #     metric_artifact = ClassificationMetrixArtifact(accuracy=accuracy, f1_score=f1, precision_score=precision, recall_score=recall)
            
        #     return best_model_detail, metric_artifact
        
        # except Exception as e:
        #     raise HotelBookingException(e, sys) from e
        

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_df = read_data(file_path=self.data_preprocessing_artifact.preprocessed_train_file_path)
            test_df = read_data(file_path=self.data_preprocessing_artifact.preprocessed_test_file_path)

            
            best_model_detail ,metric_artifact = self.get_model_object_and_report(train=train_df, test=test_df)
            
            preprocessing_obj = load_object(file_path=self.data_preprocessing_artifact.preprocessed_object_file_path)


            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            usvisa_model = HotelBookingModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e