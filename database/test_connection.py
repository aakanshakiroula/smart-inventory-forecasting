from db_connection import create_connection

connection = create_connection()

if connection:
    print("Python is successfully connected to MySQL!")

    connection.close()
    print("Database connection closed.")
else:
    print("Connection failed.")