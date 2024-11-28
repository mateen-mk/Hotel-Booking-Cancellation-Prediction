import os
from dataclasses import dataclass
from src.hotel_booking_cancellation.constants import *



# Training Configuration
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()



# Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    data_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_DATA_DIR, DSTA_INGESTION_DATA_FILE_NAME)
    ingested_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, DATA_INGESTION_INGESTED_FILE_NAME)
    # training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    # testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    # train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    


# Data Validation Configuration
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, 
                                               DATA_VALIDATION_DRIFT_REPORT_DIR, 
                                               DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    


# Data Preprocessing Configuration
@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_DIR_NAME)
    preprocessed_data_file_path: str = os.path.join(data_preprocessing_dir, 
                                                     DATA_PREPROCESSING_PREPROCESSED_DATA_DIR, 
                                                     DATA_PREPROCESSING_PREPROCESSED_DATA_FILE_NAME)
    preprocessed_object_file_path: str = os.path.join(data_preprocessing_dir,
                                                     DATA_PREPROCESSING_PREPROCESSED_OBJECT_DIR,
                                                     DATA_PREPROCESSING_PREPROCESSED_OBJECT_FILE_NAME)
    


# Model Training Configuration
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH