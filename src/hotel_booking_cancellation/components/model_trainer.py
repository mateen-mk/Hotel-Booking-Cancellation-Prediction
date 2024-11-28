import sys

import importlib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.hotel_booking_cancellation.logger import logging
from src.hotel_booking_cancellation.entity.config_entity import ModelTrainerConfig
from src.hotel_booking_cancellation.entity.artifact_entity import DataPreprocessingArtifact, ModelTrainerArtifact, ClassificationMetrixArtifact
from src.hotel_booking_cancellation.entity.estimator import HotelBookingModel
from src.hotel_booking_cancellation.exception import HotelBookingException

from src.hotel_booking_cancellation.utils.main_utils import DataUtils, TrainTestSplitUtils, YamlUtils, ObjectUtils
from src.hotel_booking_cancellation.constants import TARGET_COLUMN, TRAIN_TEST_SPLIT_RATIO



class ModelTrainer:
    """
    Class Name: ModelTrainer
    Description: Trains a model using neuro_mf and returns trained model object and classification metrics
    """
    def __init__(self, data_preprocessing_artifact: DataPreprocessingArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_preprocessing_config: Configuration for data preprocessing
        """
        try:
            logging.info("_"*100)
            logging.info("")
            logging.info("| | Started Model Trainer Stage:")
            logging.info("- "*50)

            self.data_preprocessing_artifact = data_preprocessing_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            logging.error(f"Error in ModelTrainer initialization: {str(e)}")
            raise HotelBookingException(f"Error during ModelTrainer initialization: {str(e)}", sys) from e
                


    def metrics_calculator(self, clf, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> pd.DataFrame:
        '''
        This function calculates all desired performance metrics for a given model on test data.
        The metrics are calculated specifically for class 1.
        '''
        try:
            logging.info(f"Calculating performance metrics for model: {model_name}")
            y_pred = clf.predict(X_test)
            result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                         precision_score(y_test, y_pred, pos_label=1),
                                         recall_score(y_test, y_pred, pos_label=1),
                                         f1_score(y_test, y_pred, pos_label=1),
                                         roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])],
                                   index=['Accuracy', 'Precision (Class 1)', 'Recall (Class 1)', 'F1-score (Class 1)', 'AUC (Class 1)'],
                                   columns=[model_name])
            result = (result * 100).round(2).astype(str) + '%'

            logging.info(f"Metrics for {model_name}: {result.to_dict()}")
            return result
        except Exception as e:
            raise HotelBookingException(e, sys) from e


    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        logging.info("Starting hyperparameter tuning...")
        try:

            # Use 30% of the dataset for hyperparameter tuning
            X_tune, y_tune = TrainTestSplitUtils.train_test_split_for_tuning(X_train, y_train, test_size=0.7)
            logging.info(f"Tuning dataset size: {X_tune.shape}, {y_tune.shape}")


            # Load model configuration
            model_config = YamlUtils.read_yaml_file(self.model_trainer_config.model_config_file_path)
            models = model_config['model_selection']
            grid_search_params = model_config['grid_search']['params']
            logging.info("Model configurations and hyperparameter grids loaded successfully")


            # Dictionary to store best models after tuning
            best_models = {}

            # Loop through each model and its parameters for tuning
            for model_name, model_info in models.items():
                logging.info(f"Tuning hyperparameters for model: {model_name}")

                print(f"Tuning hyperparameters for {model_name}...")
                model_class = getattr(importlib.import_module(model_info['module']), model_info['class'])
                model = model_class(**model_info['params'])
                params = model_info['search_param_grid']

                # Use GridSearchCV for tuning
                search = GridSearchCV(
                    model,
                    params,
                    **grid_search_params
                )
                logging.info(f"GridSearchCV initialized for {model_name} with parameters: {params}")

                # Fit the model using the subset of data
                search.fit(X_tune, y_tune)
                logging.info(f"Hyperparameter tuning completed for {model_name}")

                # Save the best model
                best_models[model_name] = search.best_estimator_

                # Print best parameters and best score
                print(f"Best parameters for {model_name}: {search.best_params_}")
                print(f"Best cross-validation score for {model_name}: {search.best_score_}\n")
                logging.info(f"Best parameters for {model_name}: {search.best_params_}")
                logging.info(f"Best cross-validation score for {model_name}: {search.best_score_}\n")


            logging.info("Hyperparameter tuning completed successfully")

            return best_models
        except Exception as e:
            raise HotelBookingException(e, sys) from e



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Starting model training process...")
        try:
            # Load data
            data = DataUtils.read_data(self.data_preprocessing_artifact.preprocessed_data_file_path)
            logging.info(f"Loaded preprocessed data from: {self.data_preprocessing_artifact.preprocessed_data_file_path}")
            logging.info(f"Data shape: {data.shape}")


            # Perform train-test split
            X_train, X_test, y_train, y_test = TrainTestSplitUtils.train_test_split_for_model_building(data, TRAIN_TEST_SPLIT_RATIO, TARGET_COLUMN)
            logging.info("Train-test split completed successfully")
            logging.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")


            # Tune hyperparameters and get best models
            best_models = self.tune_hyperparameters(X_train, y_train)


            # Evaluate models
            best_model = None
            best_recall = 0
            metric_artifact = None


            logging.info("Evaluating models on the test set")
            for model_name, model in best_models.items():
                logging.info(f"Evaluating model: {model_name}")

                result = self.metrics_calculator(model, X_test, y_test, model_name)
                recall = float(result.loc['Recall (Class 1)'].values[0].replace('%', ''))


                if recall > best_recall:
                    logging.info(f"Model {model_name} has recall: {recall}%")

                    best_recall = recall
                    best_model = model
                    metric_artifact = ClassificationMetrixArtifact(
                        accuracy=float(result.loc['Accuracy'].values[0].replace('%', '')),
                        precision_score=float(result.loc['Precision (Class 1)'].values[0].replace('%', '')),
                        recall_score=recall,
                        f1_score=float(result.loc['F1-score (Class 1)'].values[0].replace('%', '')),
                        auc=float(result.loc['AUC (Class 1)'].values[0].replace('%', ''))
                    )

            if best_model is None:
                raise HotelBookingException("No suitable model found.", sys)
            

            # Save the best model
            preprocessing_obj = ObjectUtils.load_object(file_path=self.data_preprocessing_artifact.preprocessed_object_file_path)
            hotel_booking_model = HotelBookingModel(preprocessing_object=preprocessing_obj, trained_model_object=best_model)
            ObjectUtils.save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=hotel_booking_model)
            logging.info("Best model saved successfully")


            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")

            return model_trainer_artifact
        except Exception as e:
            raise HotelBookingException(e, sys) from e
