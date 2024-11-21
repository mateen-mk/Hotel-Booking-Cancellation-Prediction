import os
from datetime import datetime, date
from dataclasses import dataclass



TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


DATABASE_NAME = "projects_db"
MYSQL_ENGINE_URL = "MYSQL_ENGINE_URL"

PIPELINE_NAME: str = "hotelbooking"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# dataset name for saving it in 'artifact/data' after importing it from MySQL
FILE_NAME: str = "hotel_booking.csv"
MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "is_canceled"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")



# Data Ingestion related Constants
DATASET_NAME = "hotel_booking"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_DATA_DIR: str = "data"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


