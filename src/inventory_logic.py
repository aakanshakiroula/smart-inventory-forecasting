
import math


def calculate_reorder_point(
    predicted_demand,
    lead_time_days,
    safety_stock
):
    return (
        predicted_demand * lead_time_days
        + safety_stock
    )


def calculate_reorder_quantity(
    predicted_demand,
    lead_time_days,
    safety_stock,
    current_stock,
    coverage_days=7
):
    reorder_point = calculate_reorder_point(
        predicted_demand,
        lead_time_days,
        safety_stock
    )

    if current_stock > reorder_point:
        return 0

    target_stock = (
        predicted_demand * (lead_time_days + coverage_days)
        + safety_stock
    )

    return max(
        0,
        math.ceil(target_stock - current_stock)
    )


def get_stock_status(current_stock, reorder_point):
    if current_stock <= reorder_point:
        return "REORDER"

    return "STOCK AVAILABLE"
