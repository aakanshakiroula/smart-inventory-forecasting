import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_connection import create_connection

from src.inventory_logic import (
    calculate_reorder_point,
    calculate_reorder_quantity,
    get_stock_status
)



def get_inventory_from_db():

    connection = create_connection()

    if connection is None:
        raise ConnectionError("Could not connect to MySQL database.")

    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT product_id, current_stock, lead_time_days
        FROM inventory
        """

        cursor.execute(query)
        records = cursor.fetchall()

        if not records:
            raise ValueError("No inventory records found in MySQL.")

        return pd.DataFrame(records)

    finally:
        if cursor:
            cursor.close()
        connection.close()


# Step 1: Load processed data and trained model
df = pd.read_csv("data/processed/cleaned_inventory_sales.csv")
model = joblib.load("models/inventory_demand_model.pkl")

df["date"] = pd.to_datetime(df["date"])

# Step 2: Sort data and find the latest record per product
df = df.sort_values(["product_id", "date"])

latest_data = (
    df.groupby("product_id")
    .tail(1)
    .copy()
    .reset_index(drop=True)
)

# Step 3: Prepare features for the next day
next_date = df["date"].max() + pd.Timedelta(days=1)

latest_data["month"] = next_date.month
latest_data["day_of_week"] = next_date.day_of_week
latest_data["is_weekend"] = int(next_date.day_of_week in [5, 6])

for index, row in latest_data.iterrows():

    product_history = df[
        df["product_id"] == row["product_id"]
    ].sort_values("date")

    latest_data.loc[index, "lag_1"] = (
        product_history["units_sold"].iloc[-1]
    )

    latest_data.loc[index, "lag_7"] = (
        product_history["units_sold"].iloc[-7]
    )

    latest_data.loc[index, "rolling_mean_7"] = (
        product_history["units_sold"].tail(7).mean()
    )

# Step 4: Select model input features
features = [
    "product_id",
    "month",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

# Predict next-day demand
latest_data["predicted_demand"] = model.predict(
    latest_data[features]
)

latest_data["predicted_demand"] = (
    latest_data["predicted_demand"].clip(lower=0)
)

# Step 5: Fetch live inventory information from MySQL
inventory = get_inventory_from_db()

# Step 6: Combine demand and inventory data
result = latest_data.merge(
    inventory,
    on="product_id"
)

# Step 7: Calculate average daily demand and safety stock
result["average_daily_demand"] = (
    df.groupby("product_id")["units_sold"]
    .mean()
    .reindex(result["product_id"])
    .values
)

result["demand_std"] = (
    df.groupby("product_id")["units_sold"]
    .std()
    .reindex(result["product_id"])
    .values
)

result["safety_stock"] = (
    1.65
    * result["demand_std"]
    * (result["lead_time_days"] ** 0.5)
)


# Step 8: Calculate reorder point
result["reorder_point"] = result.apply(
    lambda row: calculate_reorder_point(
        predicted_demand=row["predicted_demand"],
        lead_time_days=row["lead_time_days"],
        safety_stock=row["safety_stock"]
    ),
    axis=1
)

# Step 9: Calculate recommended order quantity
result["reorder_quantity"] = result.apply(
    lambda row: calculate_reorder_quantity(
        predicted_demand=row["predicted_demand"],
        lead_time_days=row["lead_time_days"],
        safety_stock=row["safety_stock"],
        current_stock=row["current_stock"]
    ),
    axis=1
)

# Step 10: Generate inventory status
result["status"] = result.apply(
    lambda row: get_stock_status(
        current_stock=row["current_stock"],
        reorder_point=row["reorder_point"]
    ),
    axis=1
)


# Step 10.1: Validate inventory recommendations

print("\n===== INVENTORY VALIDATION =====")

# Check that all products have inventory records
missing_products = set(latest_data["product_id"]) - set(result["product_id"])

if missing_products:
    raise ValueError(
        f"Missing inventory records for products: {missing_products}"
    )

# Check for missing values
if result[
    ["predicted_demand", "current_stock",
     "reorder_point", "reorder_quantity"]
].isnull().any().any():
    raise ValueError("Missing values found in inventory calculations.")

# Check for negative stock or quantities
if (result["current_stock"] < 0).any():
    raise ValueError("Negative current stock found.")

if (result["reorder_quantity"] < 0).any():
    raise ValueError("Negative reorder quantity found.")

# Check consistency of reorder status
expected_reorder = result["current_stock"] <= result["reorder_point"]
actual_reorder = result["status"] == "REORDER"

if not (expected_reorder == actual_reorder).all():
    raise ValueError("Inventory status does not match reorder point.")

print("All inventory validation checks passed!")


# Step 11: Display recommendations
columns = [
    "product_name",
    "predicted_demand",
    "current_stock",
    "reorder_point",
    "reorder_quantity",
    "status"
]

print("\nINVENTORY RECOMMENDATIONS")
print(result[columns].round(2).to_string(index=False))

# Step 12: Save recommendations
result[columns].to_csv(
    "data/processed/inventory_recommendations.csv",
    index=False
)

print("\nInventory recommendations saved successfully!")