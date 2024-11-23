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
    data_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_DATA_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    


# Data Validation Configuration
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    


# Data Preprocessing Configuration
@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_DIR_NAME)
    preprocessed_train_file_path: str = os.path.join(data_preprocessing_dir, DATA_PREPROCESSING_PREPROCESSED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    preprocessed_test_file_path: str = os.path.join(data_preprocessing_dir, DATA_PREPROCESSING_PREPROCESSED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    preprocessed_object_file_path: str = os.path.join(data_preprocessing_dir,
                                                     DATA_PREPROCESSING_PREPROCESSED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_NAME)
    