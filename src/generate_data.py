
import pandas as pd
import numpy as np

# Make random results reproducible
np.random.seed(42)

# Product information
products = [
    {"product_id": 101, "product_name": "Laptop", "category": "Electronics", "price": 45000, "avg_demand": 8},
    {"product_id": 102, "product_name": "Mobile", "category": "Electronics", "price": 15000, "avg_demand": 15},
    {"product_id": 103, "product_name": "Tablet", "category": "Electronics", "price": 25000, "avg_demand": 10},
    {"product_id": 104, "product_name": "Mouse", "category": "Accessories", "price": 800, "avg_demand": 20},
    {"product_id": 105, "product_name": "Keyboard", "category": "Accessories", "price": 1500, "avg_demand": 12}
]

# Generate dates for one year
dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")

# Store generated sales records
sales_data = []

# Generate daily sales for every product
for product in products:
    for date in dates:
        units_sold = np.random.poisson(product["avg_demand"])

        sales_data.append({
            "date": date,
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "category": product["category"],
            "units_sold": units_sold,
            "unit_price": product["price"],
            "revenue": units_sold * product["price"]
        })

# Convert records into a Pandas DataFrame
df = pd.DataFrame(sales_data)

# Save the dataset as a CSV file
df.to_csv("data/raw/inventory_sales.csv", index=False)

print("Sales dataset generated successfully!")
print("Total records:", len(df))
print(df.head())
