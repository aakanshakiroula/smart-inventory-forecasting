import streamlit as st
import pandas as pd
import altair as alt
import joblib
import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_connection import create_connection


# Load the trained machine learning model
MODEL_PATH = PROJECT_ROOT / "models" / "inventory_demand_model.pkl"

if not MODEL_PATH.exists():
    st.error("Trained ML model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)



# Page configuration
st.set_page_config(
    page_title="Smart Inventory System",
    page_icon="📦",
    layout="wide"
)

# Dashboard title
st.title("📦 Smart Inventory & Demand Forecasting System")

if "stock_update_message" in st.session_state:
    st.success(st.session_state.pop("stock_update_message"))

st.write(
    "Analyze product demand, monitor stock levels "
    "and generate inventory recommendations."
)

# Load data
# Connect to MySQL database
connection = create_connection()

if connection:
    try:
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT date, product_id, product_name, category,
               units_sold, unit_price, revenue
        FROM sales
        """

        cursor.execute(query)
        records = cursor.fetchall()

        df = pd.DataFrame(records)
        df["revenue"] = pd.to_numeric(df["revenue"])
        df["unit_price"] = pd.to_numeric(df["unit_price"])
        df["units_sold"] = pd.to_numeric(df["units_sold"])
        
        cursor.close()

    finally:
        connection.close()

else:
    st.error("Unable to connect to MySQL database.")
    st.stop()

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Refresh ML recommendations using the latest MySQL inventory
if st.button("🔄 Refresh ML Recommendations"):

    script_path = PROJECT_ROOT / "src" / "inventory_recommendations.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        st.success("ML recommendations updated successfully!")

        with st.expander("View refresh details"):
            st.code(result.stdout)

    else:
        st.error("Unable to refresh recommendations.")
        st.code(result.stderr)

# Load the latest recommendations
recommendations_path = (
    PROJECT_ROOT / "data" / "processed" / "inventory_recommendations.csv"
)

recommendations = pd.read_csv(recommendations_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Dashboard metrics
total_revenue = df["revenue"].sum()
total_units = df["units_sold"].sum()
total_products = df["product_id"].nunique()
reorder_count = (
    recommendations["status"] == "REORDER"
).sum()

# Display metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Total Units Sold", f"{total_units:,}")
col3.metric("Total Products", total_products)
col4.metric("Products to Reorder", reorder_count)

# Inventory recommendations
st.subheader("Inventory Recommendations")

st.dataframe(
    recommendations,
    width="stretch"
)


# Product filter
st.subheader("Product-wise Sales Analysis")

products = ["All Products"] + sorted(
    df["product_name"].unique().tolist()
)

selected_product = st.selectbox(
    "Select a product",
    products
)

# Filter sales data
if selected_product == "All Products":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["product_name"] == selected_product
    ]



# Monthly sales
st.subheader("📈 Monthly Units Sold")

monthly_sales_df = (
    filtered_df.assign(
        Month=filtered_df["date"].dt.to_period("M").dt.to_timestamp()
    )
    .groupby("Month", as_index=False)["units_sold"]
    .sum()
)

sales_chart = alt.Chart(monthly_sales_df).mark_line(
    point=True
).encode(
    x=alt.X("Month:T", title="Month"),
    y=alt.Y("units_sold:Q", title="Units Sold"),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),
        alt.Tooltip("units_sold:Q", title="Units Sold")
    ]
).properties(
    height=350
)

st.altair_chart(
    sales_chart,
    width="stretch"
)

# Monthly revenue
st.subheader("💰 Monthly Revenue")

monthly_revenue_df = (
    filtered_df.assign(
        Month=filtered_df["date"].dt.to_period("M").dt.to_timestamp()
    )
    .groupby("Month", as_index=False)["revenue"]
    .sum()
)

revenue_chart = alt.Chart(monthly_revenue_df).mark_bar().encode(
    x=alt.X("Month:T", title="Month"),
    y=alt.Y("revenue:Q", title="Revenue (INR)"),
    tooltip=[
        alt.Tooltip("Month:T", title="Month", format="%B %Y"),
        alt.Tooltip("revenue:Q", title="Revenue", format=",.0f")
    ]
).properties(
    height=350
)

st.altair_chart(
    revenue_chart,
    width="stretch"
)



# Reorder alerts
st.subheader("⚠️ Stock Reorder Alerts")

reorder_products = recommendations[
    recommendations["status"] == "REORDER"
]

if not reorder_products.empty:
    st.warning(
        f"{len(reorder_products)} products need restocking!"
    )

    st.dataframe(
        reorder_products[
            ["product_name", "current_stock",
             "reorder_point", "reorder_quantity"]
        ],
        width="stretch"
    )
else:
    st.success("All products have sufficient stock!")


st.write("---")
st.header("Download Reports")

# Download filtered sales data
sales_csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sales Report",
    data=sales_csv,
    file_name="sales_report.csv",
    mime="text/csv"
)

# Download inventory recommendations
recommendations_csv = recommendations.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Inventory Recommendations",
    data=recommendations_csv,
    file_name="inventory_recommendations.csv",
    mime="text/csv"
)


st.write("---")
st.header("Product Revenue Comparison")

product_revenue = (
    filtered_df
    .groupby("product_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(product_revenue)


st.write("---")
st.header("Business Insights")

# Highest selling product by quantity
best_selling_product = (
    filtered_df.groupby("product_name")["units_sold"]
    .sum()
    .idxmax()
)

# Highest revenue generating product
top_revenue_product = (
    filtered_df.groupby("product_name")["revenue"]
    .sum()
    .idxmax()
)

st.write("Best Selling Product:", best_selling_product)

st.write("Highest Revenue Product:", top_revenue_product)



# --------------------------------
# Demand Forecasting Section
# --------------------------------

st.write("---")
st.header("Demand Forecasting")

# Load historical sales data
forecast_data = df.sort_values(["product_id", "date"])

# Product selection
product_list = sorted(forecast_data["product_name"].unique())

selected_forecast_product = st.selectbox(
    "Select product for demand prediction",
    product_list
)

# Get selected product's sales history
product_history = forecast_data[
    forecast_data["product_name"] == selected_forecast_product
].copy()

product_history = product_history.sort_values("date")

# Get latest sales record
latest = product_history.iloc[-1]

# Prepare next-day date
next_date = latest["date"] + pd.Timedelta(days=1)

# Prepare model input
prediction_input = pd.DataFrame([{
    "product_id": latest["product_id"],
    "month": next_date.month,
    "day_of_week": next_date.day_of_week,
    "is_weekend": int(next_date.day_of_week in [5, 6]),
    "lag_1": latest["units_sold"],
    "lag_7": product_history["units_sold"].iloc[-7],
    "rolling_mean_7": product_history["units_sold"].tail(7).mean()
}])

# Predict demand
predicted_demand = model.predict(prediction_input)[0]

# Display prediction
st.subheader(f"Forecast for {selected_forecast_product}")

st.metric(
    "Predicted Next-Day Demand",
    f"{max(0, predicted_demand):.2f} units"
)

st.write("Forecast date:", next_date.strftime("%d %B %Y"))




# ==========================================
# INVENTORY MANAGEMENT
# ==========================================

st.divider()

st.header("📦 Inventory Management")

def fetch_inventory():
    connection = create_connection()

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM inventory")
            records = cursor.fetchall()
            cursor.close()
            return records
        except Exception as error:
            st.error(f"Error fetching inventory: {error}")
            return []
        finally:
            connection.close()

    return []


# Display current inventory
inventory_records = fetch_inventory()

if inventory_records:

    inventory_df = pd.DataFrame(inventory_records)
    

    st.subheader("Current Stock Overview")

    st.dataframe(
        inventory_df,
        width="stretch",
        hide_index=True
    )

    st.subheader("Update Product Stock")

    product_options = {
        item["product_name"]: item["product_id"]
        for item in inventory_records
    }

    selected_product = st.selectbox(
        "Select product",
        list(product_options.keys()),
        key="inventory_product"
    )

    selected_id = product_options[selected_product]

    current_stock = next(
        item["current_stock"]
        for item in inventory_records
        if item["product_id"] == selected_id
    )

    st.metric("Current Stock", current_stock)

    new_stock = st.number_input(
        "Enter new stock quantity",
        min_value=0,
        step=1,
        value=int(current_stock),
        key="new_stock_quantity"
    )

    
    if st.button("Update Stock"):

        connection = create_connection()

        if connection:
            cursor = None
            stock_saved = False

            try:
                cursor = connection.cursor()

                query = """
                UPDATE inventory
                SET current_stock = %s
                WHERE product_id = %s
                """

                cursor.execute(
                    query,
                    (int(new_stock), selected_id)
                )

                connection.commit()
                stock_saved = True

            except Exception as error:
                connection.rollback()
                st.error(f"Stock update failed: {error}")

            finally:
                if cursor is not None:
                    cursor.close()
                connection.close()

            # Refresh recommendations only after a successful update
            if stock_saved:

                script_path = (
                    PROJECT_ROOT / "src" / "inventory_recommendations.py"
                )

                refresh_result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True
                )

                if refresh_result.returncode == 0:
                    st.session_state["stock_update_message"] = (
                        "Stock updated and ML recommendations refreshed!"
                    )
                    st.rerun()

                else:
                    st.warning(
                        "Stock updated in MySQL, but recommendations "
                        "could not be refreshed."
                    )
                    st.code(refresh_result.stderr)

        else:
            st.error("Unable to connect to MySQL database.")

else:
    st.warning("No inventory records found.")



# ==========================================
# MODEL EVALUATION DASHBOARD
# ==========================================

st.header("🤖 ML Model Evaluation")

# Load evaluation files
metrics_path = PROJECT_ROOT / "data" / "processed" / "model_metrics.csv"
predictions_path = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
importance_path = PROJECT_ROOT / "data" / "processed" / "feature_importance.csv"

metrics_df = pd.read_csv(metrics_path)
predictions_df = pd.read_csv(predictions_path)
importance_df = pd.read_csv(importance_path)

# Load time-series cross-validation results
cv_path = (
    PROJECT_ROOT / "data" / "processed"
    / "time_series_cv_results.csv"
)

cv_df = pd.read_csv(cv_path)

# Display cross-validation results
st.subheader("Time-Series Cross-Validation")

col1, col2 = st.columns(2)

col1.metric(
    "Average CV MAE",
    round(cv_df["MAE"].mean(), 2)
)

col2.metric(
    "Average CV RMSE",
    round(cv_df["RMSE"].mean(), 2)
)

# Display fold-wise performance
st.write("Performance Across Five Folds")

st.dataframe(
    cv_df,
    width="stretch"
)

# Visualize fold-wise MAE
st.write("MAE Across Folds")

st.bar_chart(
    cv_df.set_index("Fold")[["MAE"]]
)


# 1. Display model performance
st.subheader("Model Performance Metrics")

# Convert metric values from text to numbers
metrics_df["Value"] = pd.to_numeric(
    metrics_df["Value"],
    errors="coerce"
)

metric_values = dict(
    zip(metrics_df["Metric"], metrics_df["Value"])
)


col1, col2, col3, col4 = st.columns(4)

col1.metric("Model MAE", round(metric_values["Model MAE"], 2))
col2.metric("Model RMSE", round(metric_values["Model RMSE"], 2))
col3.metric(
    "MAE Improvement",
    f'{metric_values["MAE Improvement (%)"]:.2f}%'
)
col4.metric(
    "RMSE Improvement",
    f'{metric_values["RMSE Improvement (%)"]:.2f}%'
)

# 2. Actual vs Predicted chart
st.subheader("Actual vs Predicted Demand")

predictions_df["date"] = pd.to_datetime(
    predictions_df["date"]
)

selected_product = st.selectbox(
    "Select Product for Evaluation",
    sorted(predictions_df["product_name"].unique()),
    key="evaluation_product"
)

product_predictions = predictions_df[
    predictions_df["product_name"] == selected_product
].sort_values("date")

chart_data = product_predictions.set_index("date")[
    ["Actual", "Predicted"]
]

st.line_chart(chart_data)

# 3. Feature importance
st.subheader("Feature Importance")

importance_chart = importance_df.set_index("Feature")

st.bar_chart(importance_chart)

# 4. Product-wise prediction error
st.subheader("Product-wise Prediction Performance")

product_errors = (
    predictions_df.groupby("product_name")
    .agg(
        MAE=("Absolute_Error", "mean"),
        Records=("Actual", "count")
    )
    .reset_index()
)

product_errors["MAE"] = product_errors["MAE"].round(2)

st.dataframe(product_errors, width="stretch")

# 5. Actual prediction records
with st.expander("View Actual vs Predicted Records"):
    st.dataframe(
        predictions_df,
        width="stretch"
    )