
# ==================================================
# SMART INVENTORY FORECASTING SYSTEM
# Public Interactive Demo
# ==================================================

# 1. Import libraries

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import sys

from pathlib import Path


# 2. Configure project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.inventory_logic import (
    calculate_reorder_point,
    calculate_reorder_quantity,
    get_stock_status
)

DATA_PATH = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "inventory_demand_model.pkl"
)


# 3. Page configuration

st.set_page_config(
    page_title="Smart Inventory Forecasting",
    page_icon="📦",
    layout="wide"
)


# 4. Load project datasets

@st.cache_data
def load_data():

    sales = pd.read_csv(
        DATA_PATH / "cleaned_inventory_sales.csv"
    )

    sales["date"] = pd.to_datetime(sales["date"])

    inventory = pd.read_csv(
        DATA_PATH / "inventory_recommendations.csv"
    )

    metrics = pd.read_csv(
        DATA_PATH / "model_metrics.csv"
    )

    comparison = pd.read_csv(
        DATA_PATH / "model_comparison.csv"
    )

    predictions = pd.read_csv(
        DATA_PATH / "test_predictions.csv"
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    importance = pd.read_csv(
        DATA_PATH / "feature_importance.csv"
    )

    return (
        sales,
        inventory,
        metrics,
        comparison,
        predictions,
        importance
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:

    (
        sales,
        inventory,
        metrics,
        comparison,
        predictions,
        importance
    ) = load_data()

    model = load_model()

except Exception as error:

    st.error(f"Unable to load project files: {error}")

    st.stop()


# 5. Inventory configuration

LEAD_TIMES = {
    "Laptop": 5,
    "Mobile": 4,
    "Tablet": 3,
    "Mouse": 3,
    "Keyboard": 4
}

FEATURES = [
    "product_id",
    "month",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]


# 6. Initialize simulated inventory

if "demo_stock" not in st.session_state:

    st.session_state.demo_stock = {
        row["product_name"]: int(row["current_stock"])
        for _, row in inventory.iterrows()
    }


# 7. Forecast next-day demand

def forecast_demand(product_name):

    product_data = sales[
        sales["product_name"] == product_name
    ].sort_values("date")

    if len(product_data) < 7:
        raise ValueError(
            "At least seven historical records are required."
        )

    forecast_date = (
        product_data["date"].max()
        + pd.Timedelta(days=1)
    )

    feature_data = pd.DataFrame([{

        "product_id": int(
            product_data["product_id"].iloc[-1]
        ),

        "month": forecast_date.month,

        "day_of_week": forecast_date.day_of_week,

        "is_weekend": int(
          forecast_date.day_of_week >= 5
    ),

        "lag_1": product_data["units_sold"].iloc[-1],

        "lag_7": product_data["units_sold"].iloc[-7],

        "rolling_mean_7": (
            product_data["units_sold"]
            .tail(7)
            .mean()
        )

    }])

    prediction = model.predict(
        feature_data[FEATURES]
    )[0]

    predicted_demand = max(0, float(prediction))

    return forecast_date, predicted_demand


# 8. Generate inventory recommendations

def generate_recommendations():

    results = []

    for product in sorted(
        sales["product_name"].unique()
    ):

        forecast_date, demand = forecast_demand(
            product
        )

        product_sales = sales[
            sales["product_name"] == product
        ]

        lead_time = LEAD_TIMES[product]

        current_stock = (
            st.session_state.demo_stock[product]
        )

        demand_std = (
            product_sales["units_sold"].std()
        )

        safety_stock = (
            1.65 * demand_std * np.sqrt(lead_time)
        )

        reorder_point = calculate_reorder_point(
            demand,
            lead_time,
            safety_stock
        )

        reorder_quantity = calculate_reorder_quantity(
            demand,
            lead_time,
            safety_stock,
            current_stock
        )

        status = get_stock_status(
            current_stock,
            reorder_point
        )

        results.append({

            "Product": product,

            "Forecast Date": (
                forecast_date.strftime("%Y-%m-%d")
            ),

            "Predicted Demand": round(demand, 2),

            "Current Stock": current_stock,

            "Lead Time (Days)": lead_time,

            "Safety Stock": round(safety_stock, 2),

            "Reorder Point": round(reorder_point, 2),

            "Reorder Quantity": reorder_quantity,

            "Status": status

        })

    return pd.DataFrame(results)


# 9. Dashboard heading

st.title("📦 Smart Inventory Forecasting System")

st.caption(
    "Machine Learning | Sales Analytics | "
    "Inventory Optimization"
)

st.info(
    "Public interactive demo using synthetic historical "
    "sales data and simulated inventory. The original "
    "application uses MySQL for persistent inventory "
    "management. Demo stock changes are temporary."
)


# 10. Sidebar navigation

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select a section",

    [
        "Dashboard Overview",
        "Demand Forecasting",
        "Inventory Management",
        "ML Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Historical dataset: "
    f"{sales['date'].min():%d %b %Y} to "
    f"{sales['date'].max():%d %b %Y}"
)

st.sidebar.markdown(
    "[View GitHub Repository]"
    "(https://github.com/aakanshakiroula/"
    "smart-inventory-forecasting)"
)


# ==================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==================================================

if page == "Dashboard Overview":

    st.header("Sales and Inventory Overview")

    selected_categories = st.multiselect(

        "Select Categories",

        options=sorted(
            sales["category"].unique()
        ),

        default=sorted(
            sales["category"].unique()
        )
    )

    category_data = sales[
        sales["category"].isin(
            selected_categories
        )
    ]

    available_products = sorted(
        category_data["product_name"].unique()
    )

    selected_products = st.multiselect(

        "Select Products",

        options=available_products,

        default=available_products
    )

    filtered = category_data[
        category_data["product_name"].isin(
            selected_products
        )
    ].copy()

    if filtered.empty:

        st.warning("Please select at least one product.")

        st.stop()

    recommendations = generate_recommendations()

    filtered_recommendations = recommendations[
        recommendations["Product"].isin(
            selected_products
        )
    ]

    total_revenue = filtered["revenue"].sum()

    total_units = filtered["units_sold"].sum()

    total_products = (
        filtered["product_name"].nunique()
    )

    reorder_count = (
        filtered_recommendations["Status"]
        .eq("REORDER")
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Revenue",
        f"₹{total_revenue / 10000000:,.2f} Cr"
    )

    c2.metric(
        "Units Sold",
        f"{total_units:,}"
    )

    c3.metric(
        "Total Products",
        total_products
    )

    c4.metric(
        "Products to Reorder",
        reorder_count
    )

    st.divider()

    filtered["Month"] = (
        filtered["date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (

        filtered.groupby("Month", as_index=False)

        .agg(
            Units_Sold=("units_sold", "sum"),
            Revenue=("revenue", "sum")
        )

    )

    left, right = st.columns(2)

    with left:

        st.subheader("Monthly Sales")

        chart = alt.Chart(monthly).mark_line(
            point=True
        ).encode(

            x=alt.X("Month:O", title="Month"),

            y=alt.Y(
                "Units_Sold:Q",
                title="Units Sold"
            ),

            tooltip=[
                "Month",
                "Units_Sold"
            ]

        ).properties(height=320)

        st.altair_chart(
            chart,
            width="stretch"
        )

    with right:

        st.subheader("Monthly Revenue")

        chart = alt.Chart(monthly).mark_bar(
            color="#2563eb"
        ).encode(

            x=alt.X("Month:O", title="Month"),

            y=alt.Y(
                "Revenue:Q",
                title="Revenue (INR)"
            ),

            tooltip=[
                "Month",
                "Revenue"
            ]

        ).properties(height=320)

        st.altair_chart(
            chart,
            width="stretch"
        )

    st.subheader("Product-wise Revenue")

    product_revenue = (

        filtered.groupby(
            "product_name",
            as_index=False
        )["revenue"].sum()

    )

    chart = alt.Chart(
        product_revenue
    ).mark_bar().encode(

        x=alt.X(
            "revenue:Q",
            title="Revenue (INR)"
        ),

        y=alt.Y(
            "product_name:N",
            sort="-x",
            title="Product"
        ),

        color=alt.Color(
            "product_name:N",
            legend=None
        ),

        tooltip=[
            "product_name",
            "revenue"
        ]

    ).properties(height=300)

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.subheader("Inventory Alerts")

    st.dataframe(
        filtered_recommendations,
        width="stretch",
        hide_index=True
    )


# ==================================================
# PAGE 2: DEMAND FORECASTING
# ==================================================

elif page == "Demand Forecasting":

    st.header("Machine Learning Demand Forecasting")

    product = st.selectbox(

        "Select Product",

        sorted(
            sales["product_name"].unique()
        )
    )

    forecast_date, demand = forecast_demand(
        product
    )

    product_data = sales[
        sales["product_name"] == product
    ].sort_values("date")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted Daily Demand",
        f"{demand:.2f} units"
    )

    c2.metric(
        "Historical Average Demand",
        f"{product_data['units_sold'].mean():.2f}"
    )

    c3.metric(
        "Forecast Date",
        forecast_date.strftime("%d %b %Y")
    )

    st.caption(
        "Forecast is for the day immediately after "
        "the last available historical sales record, "
        "not today's real-world demand."
    )

    st.subheader("Recent Sales and Forecast")

    history = product_data[
        ["date", "units_sold"]
    ].tail(30).copy()

    history = history.rename(
        columns={"units_sold": "Demand"}
    )

    historical_chart = alt.Chart(
        history
    ).mark_line(
        point=True,
        color="#2563eb"
    ).encode(

        x=alt.X("date:T", title="Date"),

        y=alt.Y(
            "Demand:Q",
            title="Units"
        ),

        tooltip=[
            "date",
            "Demand"
        ]
    )

    forecast_data = pd.DataFrame({

        "date": [forecast_date],

        "Demand": [demand]

    })

    forecast_chart = alt.Chart(
        forecast_data
    ).mark_point(

        color="#ef4444",

        size=180,

        filled=True

    ).encode(

        x="date:T",

        y="Demand:Q",

        tooltip=[
            "date",
            "Demand"
        ]
    )

    st.altair_chart(

        (
            historical_chart + forecast_chart
        ).properties(height=400),

        width="stretch"
    )

    st.caption(
        "Blue line: historical sales. "
        "Red point: predicted next-day demand."
    )


# ==================================================
# PAGE 3: INVENTORY MANAGEMENT
# ==================================================

elif page == "Inventory Management":

    st.header("Interactive Inventory Simulation")

    st.write(
        "Change stock quantities to see how the "
        "inventory recommendation system responds."
    )

    product = st.selectbox(

        "Select Product",

        sorted(
            st.session_state.demo_stock.keys()
        )
    )

    with st.form("stock_update_form"):

        new_stock = st.number_input(

            "Current Stock Quantity",

            min_value=0,

            value=int(
                st.session_state.demo_stock[product]
            ),

            step=1
        )

        update = st.form_submit_button(
            "Update Simulated Stock"
        )

    if update:

        st.session_state.demo_stock[product] = int(
            new_stock
        )

        st.success(
            f"{product} stock updated to "
            f"{new_stock} units."
        )

    if st.button("Reset Demo Inventory"):

        st.session_state.demo_stock = {

            row["product_name"]: int(
                row["current_stock"]
            )

            for _, row in inventory.iterrows()
        }

        st.rerun()

    recommendations = generate_recommendations()

    reorder_count = (
        recommendations["Status"]
        .eq("REORDER")
        .sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Stock",
        recommendations["Current Stock"].sum()
    )

    c2.metric(
        "Reorder Alerts",
        reorder_count
    )

    c3.metric(
        "Recommended Order Units",
        recommendations["Reorder Quantity"].sum()
    )

    st.subheader("Inventory Recommendations")

    st.dataframe(

        recommendations,

        width="stretch",

        hide_index=True
    )

    st.subheader("Stock vs Reorder Point")

    chart_data = recommendations.melt(

        id_vars=["Product"],

        value_vars=[
            "Current Stock",
            "Reorder Point"
        ],

        var_name="Metric",

        value_name="Units"
    )

    chart = alt.Chart(
        chart_data
    ).mark_bar().encode(

        x=alt.X(
            "Product:N",
            title="Product"
        ),

        xOffset="Metric:N",

        y=alt.Y(
            "Units:Q",
            title="Units"
        ),

        color="Metric:N",

        tooltip=[
            "Product",
            "Metric",
            "Units"
        ]

    ).properties(height=350)

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.download_button(

        label="Download Inventory Report",

        data=recommendations.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="inventory_recommendations.csv",

        mime="text/csv"
    )


# ==================================================
# PAGE 4: ML MODEL PERFORMANCE
# ==================================================

elif page == "ML Model Performance":

    st.header("Machine Learning Model Evaluation")

    metric_values = metrics.set_index(
        "Metric"
    )["Value"].to_dict()

    st.success(

        "Selected Model: "

        + str(
            metric_values["Selected Model"]
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Model MAE",

        f"{float(metric_values['Model MAE']):.2f}"
    )

    c2.metric(

        "Model RMSE",

        f"{float(metric_values['Model RMSE']):.2f}"
    )

    c3.metric(

        "MAE Improvement",

        f"{float(metric_values['MAE Improvement (%)']):.2f}%"
    )

    c4.metric(

        "RMSE Improvement",

        f"{float(metric_values['RMSE Improvement (%)']):.2f}%"
    )

    st.subheader("Model Comparison")

    st.dataframe(

        comparison,

        width="stretch",

        hide_index=True
    )

    comparison_long = comparison.melt(

        id_vars=["Model"],

        value_vars=[
            "Average_MAE",
            "Average_RMSE"
        ],

        var_name="Metric",

        value_name="Error"
    )

    chart = alt.Chart(
        comparison_long
    ).mark_bar().encode(

        x=alt.X(
            "Error:Q",
            title="Average CV Error"
        ),

        y=alt.Y(
            "Model:N",
            title="Model"
        ),

        yOffset="Metric:N",

        color="Metric:N",

        tooltip=[
            "Model",
            "Metric",
            "Error"
        ]

    ).properties(height=350)

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.caption(
        "Lower MAE and RMSE indicate better "
        "predictive performance on the evaluation data."
    )

    st.subheader("Actual vs Predicted Demand")

    selected_product = st.selectbox(

        "Select Product",

        sorted(
            predictions["product_name"].unique()
        )
    )

    prediction_data = predictions[

        predictions["product_name"] == selected_product

    ].sort_values("date")

    prediction_long = prediction_data.melt(

        id_vars=["date"],

        value_vars=["Actual", "Predicted"],

        var_name="Series",

        value_name="Demand"
    )

    chart = alt.Chart(
        prediction_long
    ).mark_line().encode(

        x=alt.X("date:T", title="Date"),

        y=alt.Y(
            "Demand:Q",
            title="Units Sold"
        ),

        color="Series:N",

        tooltip=[
            "date",
            "Series",
            "Demand"
        ]

    ).properties(height=400)

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.subheader("Model Feature Coefficients")

    chart = alt.Chart(
        importance
    ).mark_bar(
        color="#7c3aed"
    ).encode(

        x=alt.X(
            "Importance:Q",
            title="Absolute Coefficient"
        ),

        y=alt.Y(
            "Feature:N",
            sort="-x",
            title="Feature"
        ),

        tooltip=[
            "Feature",
            "Importance"
        ]

    ).properties(height=300)

    st.altair_chart(
        chart,
        width="stretch"
    )

    st.caption(
        "Absolute Linear Regression coefficients are "
        "displayed. Their magnitudes are affected by "
        "feature scales and should not be interpreted "
        "as definitive feature importance."
    )


# ==================================================
# PAGE 5: ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.header("About This Project")

    st.markdown("""
    ### Smart Inventory Forecasting System

    This project demonstrates an end-to-end
    machine learning workflow for demand forecasting
    and inventory optimization.

    ### Technologies

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - MySQL
    - Streamlit
    - Altair
    - Joblib
    - Git and GitHub

    ### Project Highlights

    - 1,825 synthetic historical sales records
    - Five regression models compared
    - Time-series cross-validation
    - Linear Regression selected as the final model
    - 26.20% MAE improvement over the naive baseline
    - 26.61% RMSE improvement over the naive baseline
    - Safety stock and reorder recommendations
    - 13 verified unit tests

    ### Demo Limitations

    This public demo uses historical CSV files
    and a trained machine learning model.

    Inventory updates are simulated using
    Streamlit session state and are not stored
    permanently in MySQL.

    The original local application supports
    persistent MySQL inventory management.
    """)

    st.link_button(

        "View Complete GitHub Project",

        "https://github.com/aakanshakiroula/"
        "smart-inventory-forecasting"
    )


# 11. Dashboard footer

st.divider()

st.caption(
    "Smart Inventory Forecasting System | "
    "Data Science and Machine Learning Portfolio Project"
)
