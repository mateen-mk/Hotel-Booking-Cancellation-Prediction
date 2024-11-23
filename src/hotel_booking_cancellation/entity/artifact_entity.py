from dataclasses import dataclass



# Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 


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
    preprocessed_train_file_path:str     # file path to trained data in numpy array format (train.npy)
    preprocessed_test_file_path:str      # file path to test data in numpy array format (test.npy)