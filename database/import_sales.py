
import pandas as pd
from db_connection import create_connection

# Step 1: Read the CSV file
df = pd.read_csv("data/raw/inventory_sales.csv")

# Step 2: Convert date column into Python dates
df["date"] = pd.to_datetime(df["date"]).dt.date

# Step 3: Connect to MySQL
connection = create_connection()

if connection:
    try:
        cursor = connection.cursor()

        # Step 4: Check whether data already exists
        cursor.execute("SELECT COUNT(*) FROM sales")
        existing_records = cursor.fetchone()[0]

        if existing_records > 0:
            print("Sales table already contains data.")
            print("Import skipped to prevent duplicate records.")

        else:
            # Step 5: Prepare SQL query
            query = """
            INSERT INTO sales
            (date, product_id, product_name, category,
             units_sold, unit_price, revenue)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            # Step 6: Convert DataFrame into rows
            rows = list(df.itertuples(index=False, name=None))

            # Step 7: Insert all records
            cursor.executemany(query, rows)

            # Step 8: Save changes
            connection.commit()

            print(len(rows), "sales records imported successfully!")

        cursor.close()

    except Exception as error:
        connection.rollback()
        print("Import failed:", error)

    finally:
        connection.close()
        print("Database connection closed.")
