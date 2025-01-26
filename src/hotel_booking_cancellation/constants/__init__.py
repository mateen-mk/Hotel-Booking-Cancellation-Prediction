import os
from datetime import datetime, date



TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


DATABASE_NAME = "projects_db"
MYSQL_ENGINE_URL = "MYSQL_ENGINE_URL"

PIPELINE_NAME: str = "hotelbooking"
ARTIFACT_DIR: str = "artifact"

# TRAIN_FILE_NAME: str = "train.csv"
# TEST_FILE_NAME: str = "test.csv"

# dataset name for saving it in 'artifact/data' after importing it from MySQL
# FILE_NAME: str = "hotel_booking.csv"
MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "is_canceled"
CURRENT_YEAR = date.today().year
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
TRAIN_TEST_SPLIT_RATIO: float = 0.2
MONTH_ORDER: list =['January', 'February', 'March', 'April', 
                    'May', 'June', 'July', 'August', 
                    'September', 'October', 'November', 'December']


# # AWS Constants
# AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
# AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
# REGION_NAME = "us-east-1"


# Data Ingestion related Constants
DATASET_NAME = "hotel_booking"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_DIR: str = "raw"
DSTA_INGESTION_RAW_FILE_NAME: str = "raw.csv"
DATA_INGESTION_DATA_DIR: str = "data"
DATA_INGESTION_DATA_FILE_NAME: str = "data.csv"

# # Data Ingestion related Constants
# DATASET_NAME = "hotel_booking"
# DATA_INGESTION_DIR_NAME: str = "data_ingestion"
# DATA_INGESTION_DATA_DIR: str = "data"
# DSTA_INGESTION_DATA_FILE_NAME: str = "data.csv"
# DATA_INGESTION_INGESTED_DIR: str = "ingested"
# DATA_INGESTION_INGESTED_FILE_NAME: str = "ingested.csv"


# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# Data Preprocessing related constants
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
DATA_PREPROCESSING_TRAIN_DATA_FILE_NAME: str = "train.csv"
DATA_PREPROCESSING_TEST_DATA_FILE_NAME: str = "test.csv"
DATA_PREPROCESSING_VALIDATION_DATA_FILE_NAME: str = "val.csv"
DATA_PREPROCESSING_PREPROCESSED_OBJECT_FILE_NAME = "preprocessing.pkl"

# # Data Preprocessing related constants
# DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
# DATA_PREPROCESSING_PREPROCESSED_DATA_DIR: str = "preprocessed"
# DATA_PREPROCESSING_PREPROCESSED_DATA_FILE_NAME: str = "preprocessed.csv"
# DATA_PREPROCESSING_PREPROCESSED_OBJECT_DIR: str = "preprocessed_object"
# DATA_PREPROCESSING_PREPROCESSED_OBJECT_FILE_NAME = "preprocessing.pkl"


# Model Training related constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
# MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "trained_model.pkl"
MODEL_TRAINER_METRIC_FILE_NAME: str = "metrics.yaml"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.75
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# # Model Training related constants
# MODEL_TRAINER_DIR_NAME: str = "model_trainer"
# MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
# MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
# MODEL_TRAINER_EXPECTED_SCORE: float = 0.75
# MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


# Model Evaluation related constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_DIR: str = "evaluation_report"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "report.json"


# MODEL_BUCKET_NAME = "AWS_BUCKET_NAME"
# # MODEL_PUSHER_S3_KEY = "model-registry"
# MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02