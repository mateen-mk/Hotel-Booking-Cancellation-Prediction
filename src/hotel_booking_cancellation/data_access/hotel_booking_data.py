import sys
import pandas as pd
from typing import Optional

from src.hotel_booking_cancellation.configuration.mysql_connection import MySQLConnect
from src.hotel_booking_cancellation.exception import HotelBookingException
from src.hotel_booking_cancellation.constants import DATABASE_NAME


class HotelBookingData:
    """
    This class helps to export entire MySQL table data as a pandas DataFrame.
    """

    def __init__(self):
        """
        Initializes the MySQL client connection.
        """
        try:
            self.mysql_connect = MySQLConnect()
        except Exception as e:
            raise HotelBookingException(e, sys)

    def export_data_as_dataframe(self, dataset_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports the entire table as a pandas DataFrame.
        
        :param dataset_name: Name of the dataset to export.
        :param database_name: Name of the database (optional, defaults to the connection's database).
        :return: pd.DataFrame containing table data.
        """
        try:
            # Use the default database if none is provided
            database_name = database_name or DATABASE_NAME
            
            # Construct the SQL query
            query = f"SELECT * FROM {database_name}.{dataset_name}"
            
            # Fetch data using SQLAlchemy
            with self.mysql_connect.engine.connect() as connection:
                df = pd.read_sql(query, connection)
            
            # Replace placeholder values (e.g., "na") with NaN
            df.replace({"na": pd.NA}, inplace=True)
            
            return df
        except Exception as e:
            raise HotelBookingException(e, sys)
