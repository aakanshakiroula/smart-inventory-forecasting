
import pandas as pd
import matplotlib.pyplot as plt

# Load the sales dataset
df = pd.read_csv("data/raw/inventory_sales.csv")



# Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

print("\n----- UPDATED DATA TYPES -----")
print(df.dtypes)

# Total units sold by each product
print("\n----- PRODUCT-WISE SALES -----")
product_sales = df.groupby("product_name")["units_sold"].sum()
print(product_sales)

# Total revenue by each product
print("\n----- PRODUCT-WISE REVENUE -----")
product_revenue = df.groupby("product_name")["revenue"].sum()
print(product_revenue)

# Category-wise revenue
print("\n----- CATEGORY-WISE REVENUE -----")
category_revenue = df.groupby("category")["revenue"].sum()
print(category_revenue)


# Display the first 5 rows
print("\n----- FIRST 5 ROWS -----")
print(df.head())

# Display the number of rows and columns
print("\n----- DATASET SHAPE -----")
print(df.shape)

# Display column names and data types
print("\n----- DATASET INFORMATION -----")
df.info()

# Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Check duplicate records
print("\n----- DUPLICATE RECORDS -----")
print(df.duplicated().sum())

# Statistical summary
print("\n----- STATISTICAL SUMMARY -----")
print(df.describe())


# 1. Product-wise units sold
product_sales = df.groupby("product_name")["units_sold"].sum()

plt.figure(figsize=(9, 5))
product_sales.plot(kind="bar")

plt.title("Total Units Sold by Product")
plt.xlabel("Product Name")
plt.ylabel("Units Sold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Product-wise revenue
product_revenue = df.groupby("product_name")["revenue"].sum()

plt.figure(figsize=(9, 5))
product_revenue.plot(kind="bar")

plt.title("Total Revenue by Product")
plt.xlabel("Product Name")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. Category-wise revenue
category_revenue = df.groupby("category")["revenue"].sum()

plt.figure(figsize=(7, 5))
category_revenue.plot(kind="pie", autopct="%1.1f%%")

plt.title("Revenue Contribution by Category")
plt.ylabel("")
plt.tight_layout()
plt.show()



# 4. Monthly sales analysis

# Extract month from date and calculate monthly sales
df["month"] = df["date"].dt.to_period("M")

monthly_sales = df.groupby("month")["units_sold"].sum()
monthly_revenue = df.groupby("month")["revenue"].sum()

print("\nMonthly Units Sold:")
print(monthly_sales)

print("\nMonthly Revenue:")
print(monthly_revenue)

# Monthly units sold chart
plt.figure(figsize=(12, 5))
monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Units Sold Trend")
plt.xlabel("Month")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Monthly revenue chart
plt.figure(figsize=(12, 5))
monthly_revenue.plot(kind="line", marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()