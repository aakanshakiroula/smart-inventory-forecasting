
import pandas as pd
from db_connection import create_connection

connection = create_connection()

if connection:
    try:
        query = "SELECT * FROM sales"

        df = pd.read_sql(query, connection)

        print("Data fetched successfully!")
        print("\nFirst 5 records:")
        print(df.head())

        print("\nTotal records:", len(df))

    except Exception as error:
        print("Error:", error)

    finally:
        connection.close()
