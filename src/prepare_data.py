import pandas as pd

# Step 1: Load raw sales data
df = pd.read_csv("data/raw/inventory_sales.csv")

# Step 2: Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

# Step 3: Check missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Step 4: Check duplicate product-date records
duplicates = df.duplicated(
    subset=["date", "product_id"]
).sum()

print("Duplicate product-date records:", duplicates)

df = df.drop_duplicates(
    subset=["date", "product_id"]
)

# Step 5: Handle missing values
df = df.dropna()

# Step 6: Create date-related features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.day_of_week
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Step 7: Sort records by product and date
df = df.sort_values(by=["product_id", "date"])



# Step 8: Create lag features

df["lag_1"] = df.groupby("product_id")["units_sold"].shift(1)

df["lag_7"] = df.groupby("product_id")["units_sold"].shift(7)

df["rolling_mean_7"] = (
    df.groupby("product_id")["units_sold"]
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)

# Step 9: Save processed data
df.to_csv("data/processed/cleaned_inventory_sales.csv", index=False)

# Step 10: Display final information
print("\nData preparation completed!")
print("Final dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

print("\nLag feature columns:")
print(df[["date", "product_name", "units_sold",
          "lag_1", "lag_7", "rolling_mean_7"]].head(12))