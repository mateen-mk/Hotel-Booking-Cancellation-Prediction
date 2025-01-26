from dataclasses import dataclass



# Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    data_file_path: str
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
    train_data_file_path:str                # file path to preprocessed train data
    test_data_file_path:str                 # file path to preprocessed test data 
    validation_data_file_path:str           # file path to preprocessed validation data 
    preprocessed_object_file_path:str       # file path to preprocessing.pkl

# Classification Matrix Artifact
@dataclass
class ClassificationMetrixArtifact:
    accuracy: str
    f1_score:float
    precision_score:float
    recall_score:float
    auc: float


# Model Training Artifacts
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetrixArtifact


# Model Evaluation Artifacts
@dataclass
class ModelEvaluationArtifact:
    evaluation_report_file_path:str


    # is_model_accepted:bool
    # changed_accuracy:float
    # s3_model_path:str 
    # trained_model_path:str