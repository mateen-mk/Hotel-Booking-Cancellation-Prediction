# Hotel-Booking-Cancellation-Prediction



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