import sys
import pandas as pd
from typing import Optional

from hotel_booking_cancellation.configuration.mysql_connection import MySQLConnect
from hotel_booking_cancellation.exception import HotelBookingException
from hotel_booking_cancellation.constants import DATABASE_NAME


class USvisaData:
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

    def export_table_as_dataframe(self, table_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports the entire table as a pandas DataFrame.
        
        :param table_name: Name of the table to export.
        :param database_name: Name of the database (optional, defaults to the connection's database).
        :return: pd.DataFrame containing table data.
        """
        try:
            # Use the default database if none is provided
            database_name = database_name or DATABASE_NAME
            
            # Construct the SQL query
            query = f"SELECT * FROM {database_name}.{table_name}"
            
            # Fetch data using SQLAlchemy
            with self.mysql_connect.engine.connect() as connection:
                df = pd.read_sql(query, connection)
            
            # Replace placeholder values (e.g., "na") with NaN
            df.replace({"na": pd.NA}, inplace=True)
            
            return df
        except Exception as e:
            raise HotelBookingException(e, sys)
