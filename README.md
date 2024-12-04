# Hotel Booking Cancellation Prediction

## **Overview**  
This project aims to predict hotel booking cancellations based on historical booking data. The system implements a robust machine learning pipeline, addressing data ingestion, preprocessing, validation, and model training, while ensuring reproducibility and maintainability. The project integrates features like data drift detection, versioning, and modular design, making it scalable for real-world deployment.

---

## **Problem Statement**  
Hotels often face uncertainty due to booking cancellations, which affect revenue and resource planning. The client needs a predictive system that can classify whether a booking is likely to be canceled, allowing for better inventory management and dynamic pricing strategies.

---

## **Solution Approach**  
This project builds an end-to-end machine learning pipeline that:  
1. **Ingests** raw booking data from a MySQL database.  
2. **Validates** data to ensure schema and quality consistency.  
3. **Preprocesses** the data using transformations like encoding, normalization, and missing value handling.  
4. **Detects data drift** to ensure that the model performs reliably over time.  
5. **Trains and evaluates** machine learning models to predict cancellations.  
6. **Deploys** the trained model, enabling real-time predictions via a web application.  

---

## **Features**  
- **Data Validation:** Ensures integrity with schema checks and drift detection.  
- **Pipeline Modularity:** Components are reusable and designed for scalability.  
- **Model Training:** Employs various machine learning algorithms and selects the best-performing model.  
- **Versioning:** Artifacts, models, and configurations are version-controlled for reproducibility.  
- **Interactive Notebooks:** Jupyter notebooks for experimentation, visualization, and debugging.  
- **Web Interface:** Enables users to upload booking data and receive predictions.  
- **Docker Support:** Simplifies deployment with containerization.  

---

## **Folder Structure**  
```plaintext
.
├── artifact/                     # Stores versioned data and model artifacts
├── config/                       # Configuration files for schema and model settings
├── logs/                         # Log files for debugging and monitoring
├── notebooks/                    # Jupyter notebooks for EDA and experimentation
├── src/                          # Source code for the pipeline and utility functions
├── app.py                        # Flask/Streamlit app for deployment
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker containerization setup
├── README.md                     # Project documentation
```

---

## **Notebooks**  
The `notebooks` folder contains Jupyter notebooks that demonstrate key steps of the project:  
1. **01_fetch_data_mysql.ipynb**: Demonstrates how data is fetched from the MySQL database.  
2. **02_EDA.ipynb**: Explores the data through visualizations and descriptive statistics.  
3. **03_data_preprocessing.ipynb**: Applies preprocessing techniques like handling missing values, encoding, and normalization.  
4. **04_model_building.ipynb**: Builds and evaluates machine learning models.  
5. **data_drift_evidently.ipynb**: Detects data drift using EvidentlyAI and generates drift reports.  
6. **HotelBooking_data_drift_dashboard.html**: An interactive dashboard for visualizing data drift.  
7. **trainl.ipynb**: A consolidated notebook to train the final model.  

These notebooks are essential for prototyping, experimentation, and understanding intermediate steps in the project.

---

## **Pipeline Details**  
### **1. Data Ingestion**  
- Fetches raw data from MySQL and saves it locally.  
- Example artifact: `artifact/<timestamp>/data_ingestion/data.csv`.  

### **2. Data Validation**  
- Checks schema consistency and generates a drift report.  
- Example artifact: `artifact/<timestamp>/data_validation/drift_report.yaml`.  

### **3. Data Preprocessing**  
- Handles missing values, encodes categorical features, and normalizes numerical features.  
- Example artifact: `artifact/<timestamp>/data_preprocessing/preprocessed.csv`.  

### **4. Model Training**  
- Trains models such as Logistic Regression, Random Forest, and XGBoost.  
- Stores the trained model: `artifact/<timestamp>/model_trainer/trained_model/model.pkl`.  

### **5. Deployment**  
- A Flask/Streamlit app allows users to interact with the system.  

---

## **Technologies Used**  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, EvidentlyAI, Flask/Streamlit  
- **Database:** MySQL  
- **Visualization:** Matplotlib, Seaborn  
- **Versioning:** Git  
- **Containerization:** Docker  

---

## **How to Run the Project**  
1. Clone the repository:  
   ```bash
   git clone <repo_url>
   cd hotel_booking_cancellation_prediction
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the database connection in `config/mysql_connection.py`.  
4. Run the training pipeline:  
   ```bash
   python src/hotel_booking_cancellation/pipline/training_pipeline.py
   ```
5. Start the web application:  
   ```bash
   python app.py
   ```
6. Open the application in your browser at `http://localhost:5000`.

---

## **Future Enhancements**  
- Integration of deep learning models for better accuracy.  
- Real-time data ingestion from live APIs.  
- Advanced data visualization using dashboards like Tableau or Power BI.  

---

## **Contributors**  
- **Mateen Khan**  

For any questions or suggestions, feel free to contact **[Your Email/LinkedIn](#)**.  



--- 

#### Action Plan
1. Data Ingestion
    - Drop Sensitive Columns: `name`, `email`, `phone-number` and `credit_card`.
2. Data Validation
3. Feature Engineering
    - Remove Directly related features: `reservation_status`, `reservation_status_date`, etc.
4. Data Cleaning
    - Handle missing values
    - Handle noisy data
4. Data Transformation
    - Encoding Categorical features
    - Normalize/Scale features



#### to do
- Go through same as notebooks
    1. Data ingestion:
        - fetch data from MySQL
        - remove sensitive features
        - save data in feature_store: without splitting it
    2. Data validation:
        - validate whole data: instead of train and test differntly
    3. Data Preprocessing:
    - functions
        - read data from artifact/data_ingestion/ingested/ingested.csv
        - remove directly related features
        - handle missing values
        - handle noisy data
        - and go with notebooks
    4. Model Building:
        - read data from artifact/data_prprocessing/preprocessed/preprocessed.csv
        - perform train-test split
        - Now look into notebook and figure out


### Things we have done till now:
1. Introduction of project and setup
2. database: MySQL connection, how to get connection url.
3. Notebooks:
    - fetch data from MySQL
    - EDA
    - Data Preprocessing
    - model building
4. Modular programming:
    - created some utility functions that are used commonly in code
    - Data ingestion:
        - created Data ingestion related constants
        - created Data ingestion related configurations
        - created Data ingestion related artifacts
        - created components (data_ingestion.py)
        - created Data ingestion pipeline in training_pipeline.py
    - Data Validation:
        - created Data validation related constants
        - created Data validation related configurations
        - created Data validation related artifacts
        - created components (data_validation.py)
        - created Data validation pipeline in training_pipeline.py
    - Data Preprocessing:
        - created Data Preprocessing related constants
        - created Data Preprocessing related configurations
        - created Data Preprocessing related artifacts
        - created components (data_preprocessing.py)
        - created Data Preprocessing pipeline in training_pipeline.py
    - Model Trainer:
        - created Model Trainer related constants
        - created Model Trainer related configurations
        - created Model Trainer related artifacts
        - created components (model_trainer.py)
        - created Model Trainer pipeline in training_pipeline.py