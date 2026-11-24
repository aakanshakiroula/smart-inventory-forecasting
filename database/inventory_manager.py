
from db_connection import create_connection


# Function 1: Fetch all inventory records
def get_inventory():
    connection = create_connection()

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)

            cursor.execute("SELECT * FROM inventory")

            records = cursor.fetchall()

            cursor.close()

            return records

        except Exception as error:
            print("Error fetching inventory:", error)
            return []

        finally:
            connection.close()

    return []


# Function 2: Update product stock
def update_stock(product_id, new_stock):

    if new_stock < 0:
        print("Stock cannot be negative.")
        return

    connection = create_connection()

    if connection:
        try:
            cursor = connection.cursor()

            query = """
            UPDATE inventory
            SET current_stock = %s
            WHERE product_id = %s
            """

            cursor.execute(query, (new_stock, product_id))

            if cursor.rowcount == 0:
                print("Product ID not found.")
            else:
                connection.commit()
                print("Stock updated successfully!")

            cursor.close()

        except Exception as error:
            connection.rollback()
            print("Error updating stock:", error)

        finally:
            connection.close()


# Main program
if __name__ == "__main__":

    print("\n===== INVENTORY MANAGEMENT =====")

    # Display inventory
    inventory = get_inventory()

    for item in inventory:
        print(
            item["product_id"],
            item["product_name"],
            "Stock:", item["current_stock"],
            "Lead Time:", item["lead_time_days"]
        )

    # Update stock
    try:
        product_id = int(input("\nEnter Product ID to update: "))
        new_stock = int(input("Enter new stock quantity: "))

        update_stock(product_id, new_stock)

    except ValueError:
        print("Please enter valid whole numbers.")
