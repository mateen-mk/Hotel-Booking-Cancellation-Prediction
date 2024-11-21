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

