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
    raw_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_RAW_DIR, DSTA_INGESTION_RAW_FILE_NAME)
    data_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_DATA_DIR, DATA_INGESTION_DATA_FILE_NAME)
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
    train_data_file_path: str = os.path.join(data_preprocessing_dir,DATA_PREPROCESSING_TRAIN_DATA_FILE_NAME)
    test_data_file_path: str = os.path.join(data_preprocessing_dir,DATA_PREPROCESSING_TEST_DATA_FILE_NAME)
    validation_data_file_path: str = os.path.join(data_preprocessing_dir,DATA_PREPROCESSING_VALIDATION_DATA_FILE_NAME)
    preprocessed_object_file_path: str = os.path.join(data_preprocessing_dir,DATA_PREPROCESSING_PREPROCESSED_OBJECT_FILE_NAME)
    


# Model Training Configuration
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_FILE_NAME)
    metric_article_file_path: str = os.path.join(model_trainer_dir,MODEL_TRAINER_METRIC_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH



# Model Evaluation Configuration
@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME)
    evaluation_report_file_path: str = os.path.join(model_evaluation_dir, 
                                               MODEL_EVALUATION_REPORT_DIR, 
                                               MODEL_EVALUATION_REPORT_FILE_NAME)
        
    
    # changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    # bucket_name: str = MODEL_BUCKET_NAME
    # s3_model_key_path: str = MODEL_FILE_NAME