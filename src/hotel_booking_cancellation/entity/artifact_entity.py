from dataclasses import dataclass



# Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    ingest_file_path: str
    # trained_file_path:str 
    # test_file_path:str 


# Data Validation Artifact
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    drift_report_file_path: str


# Data Preprocessing Artifact
@dataclass
class DataPreprocessingArtifact:
    preprocessed_object_file_path:str    # file path to preprocessing.pkl
    preprocessed_data_file_path:str     # file path to preprocessed data
    # preprocessed_test_file_path:str      # file path to test data in numpy array format (test.npy)


# Classification Matrix Artifact
@dataclass
class ClassificationMetrixArtifact:
    accuracy: str
    f1_score:float
    precision_score:float
    recall_score:float


# Model Training Artifacts
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetrixArtifact