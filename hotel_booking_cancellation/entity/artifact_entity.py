from dataclasses import dataclass



# Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 



