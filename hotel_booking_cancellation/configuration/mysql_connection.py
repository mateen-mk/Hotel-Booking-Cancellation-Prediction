import sys
from hotel_booking_cancellation.exception import HotelBookingException
from hotel_booking_cancellation.logger import logging
from hotel_booking_cancellation.constants import MYSQL_ENGINE_URL

import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

class MySQLConnect:
    """
    Class Name :   MySQLConnect
    Description :   This class establishes a connection to the MySQL database using SQLAlchemy.

    Output      :   Connection to the MySQL database
    On Failure  :   Raises an exception
    """
    engine = None

    def __init__(self) -> None:
        try:
            # Get SQLAlchemy engine URL from environment variable
            sqlalchemy_url = os.getenv(MYSQL_ENGINE_URL)
            
            if not sqlalchemy_url:
                raise Exception("Environment variable 'MYSQL_ENGINE_URL' is not set.")
            
            if MySQLConnect.engine is None:
                # Initialize the SQLAlchemy engine
                MySQLConnect.engine = create_engine(sqlalchemy_url)
            
            self.engine = MySQLConnect.engine
            logging.info("MySQL connection successful")
        
        except SQLAlchemyError as e:
            raise HotelBookingException(f"MySQL connection error: {e}", sys)
        except Exception as e:
            raise HotelBookingException(e, sys)
