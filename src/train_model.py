import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Step 1: Load processed dataset
df = pd.read_csv("data/processed/cleaned_inventory_sales.csv")

df["date"] = pd.to_datetime(df["date"])

# Step 2: Remove rows with missing lag values
df = df.dropna()

# Step 3: Sort data by date
df = df.sort_values("date")

# Step 4: Select input features
features = [
    "product_id",
    "month",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

X = df[features]
y = df["units_sold"]

# Step 5: Split data chronologically
unique_dates = sorted(df["date"].unique())
split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Split date:", split_date)

# Step 6: Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Make predictions
predictions = model.predict(X_test)

# Step 9: Evaluate model
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))




# Step 10: Compare with a simple baseline

baseline_predictions = X_test["lag_1"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5

print("\nBaseline Model Evaluation:")
print("Baseline MAE:", round(baseline_mae, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))

# Calculate model improvement over baseline

mae_improvement = (
    (baseline_mae - mae) / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rmse) / baseline_rmse
) * 100

print("MAE Improvement:", round(mae_improvement, 2), "%")
print("RMSE Improvement:", round(rmse_improvement, 2), "%")



# Step 10.1: Save model evaluation metrics

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

metrics = pd.DataFrame({
    "Metric": [
        "Model MAE",
        "Model RMSE",
        "Baseline MAE",
        "Baseline RMSE",
        "MAE Improvement (%)",
        "RMSE Improvement (%)"
    ],
    "Value": [
        mae,
        rmse,
        baseline_mae,
        baseline_rmse,
        mae_improvement,
        rmse_improvement
    ]
})

metrics.to_csv(
    output_dir / "model_metrics.csv",
    index=False
)

print("Model metrics saved successfully!")

# Step 10.2: Time-Series Cross-Validation

print("\n===== TIME-SERIES CROSS-VALIDATION =====")

# Get all unique dates in chronological order
unique_dates = sorted(
    df.loc[X_train.index, "date"].unique()
)

# Create 5 chronological splits
tscv = TimeSeriesSplit(n_splits=5)

cv_results = []

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(unique_dates), start=1
):

    # Select training and testing dates
    train_dates = [unique_dates[i] for i in train_idx]
    test_dates = [unique_dates[i] for i in test_idx]

    # Create masks using dates, keeping each date together
    train_fold_mask = df["date"].isin(train_dates)
    test_fold_mask = df["date"].isin(test_dates)

    X_train_fold = X[train_fold_mask]
    X_test_fold = X[test_fold_mask]

    y_train_fold = y[train_fold_mask]
    y_test_fold = y[test_fold_mask]

    # Train a fresh model for this fold
    fold_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    fold_model.fit(X_train_fold, y_train_fold)

    # Predict future-period sales
    fold_predictions = fold_model.predict(X_test_fold)

    # Calculate evaluation metrics
    fold_mae = mean_absolute_error(
        y_test_fold, fold_predictions
    )

    fold_rmse = mean_squared_error(
        y_test_fold, fold_predictions
    ) ** 0.5

    cv_results.append({
        "Fold": fold,
        "Training_Records": len(X_train_fold),
        "Testing_Records": len(X_test_fold),
        "MAE": fold_mae,
        "RMSE": fold_rmse
    })

    print(
        f"Fold {fold}: "
        f"MAE = {fold_mae:.2f}, "
        f"RMSE = {fold_rmse:.2f}"
    )

# Convert results into a DataFrame
cv_df = pd.DataFrame(cv_results)

# Calculate average performance across all folds
print("\nAverage Cross-Validation Performance:")

print("Average MAE:", round(cv_df["MAE"].mean(), 2))
print("Average RMSE:", round(cv_df["RMSE"].mean(), 2))

# Save cross-validation results
cv_df.to_csv(
    output_dir / "time_series_cv_results.csv",
    index=False
)

print("\nCross-validation results saved successfully!")


# Step 10.3: Compare multiple ML models

print("\n===== MODEL COMPARISON =====")

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Linear Regression": LinearRegression()
}

comparison_results = []

for model_name, algorithm in models.items():

    fold_mae_scores = []
    fold_rmse_scores = []

    for train_idx, test_idx in tscv.split(unique_dates):

        train_dates = [unique_dates[i] for i in train_idx]
        test_dates = [unique_dates[i] for i in test_idx]

        # Create separate masks
        train_comparison_mask = df["date"].isin(train_dates)
        test_comparison_mask = df["date"].isin(test_dates)

        X_train_fold = X[train_comparison_mask]
        X_test_fold = X[test_comparison_mask]

        y_train_fold = y[train_comparison_mask]
        y_test_fold = y[test_comparison_mask]

        # Train model
        algorithm.fit(X_train_fold, y_train_fold)

        # Make predictions
        fold_predictions = algorithm.predict(X_test_fold)

        # Calculate MAE
        fold_mae = mean_absolute_error(
            y_test_fold, fold_predictions
        )

        # Calculate RMSE
        fold_rmse = mean_squared_error(
            y_test_fold, fold_predictions
        ) ** 0.5

        fold_mae_scores.append(fold_mae)
        fold_rmse_scores.append(fold_rmse)

    # Calculate average after all folds
    comparison_results.append({
        "Model": model_name,
        "Average_MAE": sum(fold_mae_scores) / len(fold_mae_scores),
        "Average_RMSE": sum(fold_rmse_scores) / len(fold_rmse_scores)
    })

# Create comparison DataFrame
model_comparison = pd.DataFrame(comparison_results)

model_comparison = model_comparison.sort_values(
    by="Average_MAE"
).reset_index(drop=True)

model_comparison["Average_MAE"] = (
    model_comparison["Average_MAE"].round(2)
)

model_comparison["Average_RMSE"] = (
    model_comparison["Average_RMSE"].round(2)
)

print("\nModel Comparison Results:")
print(model_comparison.to_string(index=False))

# Save results
model_comparison.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

# Select the best model
best_model_name = model_comparison.iloc[0]["Model"]

print("\nBest Performing Model:", best_model_name)


# Step 10.4: Select and train the best model

print("\n===== FINAL MODEL SELECTION =====")

# Get the best model from cross-validation
model = models[best_model_name]

print("Selected Model:", best_model_name)

# Train selected model on the complete training dataset
model.fit(X_train, y_train)

# Predict on untouched test dataset
predictions = model.predict(X_test)

# Evaluate selected model
mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test, predictions
) ** 0.5

print("\nFinal Model Evaluation:")
print("Selected Model:", best_model_name)
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Compare with baseline again
baseline_predictions = X_test["lag_1"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5

mae_improvement = (
    (baseline_mae - mae) / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rmse) / baseline_rmse
) * 100

print("MAE Improvement:", round(mae_improvement, 2), "%")
print("RMSE Improvement:", round(rmse_improvement, 2), "%")

# Update final evaluation metrics
metrics = pd.DataFrame({
    "Metric": [
        "Selected Model",
        "Model MAE",
        "Model RMSE",
        "Baseline MAE",
        "Baseline RMSE",
        "MAE Improvement (%)",
        "RMSE Improvement (%)"
    ],
    "Value": [
        best_model_name,
        mae,
        rmse,
        baseline_mae,
        baseline_rmse,
        mae_improvement,
        rmse_improvement
    ]
})

metrics.to_csv(
    output_dir / "model_metrics.csv",
    index=False
)

print("\nFinal model metrics updated successfully!")

# Save selected model
joblib.dump(
    model,
    "models/inventory_demand_model.pkl"
)

print("Best model saved successfully!")


# Step 11: Save trained model
joblib.dump(model, "models/inventory_demand_model.pkl")

print("\nModel saved successfully!")


# Step 11.1: Feature Importance


# Step 11: Feature Importance

print("\n===== FEATURE IMPORTANCE =====")

# Check the selected model type
if hasattr(model, "feature_importances_"):

    importance_values = model.feature_importances_

elif hasattr(model, "coef_"):

    importance_values = abs(model.coef_)

else:

    importance_values = None

if importance_values is not None:

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance_values
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_importance.to_string(index=False))

    feature_importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False
    )

    print("\nFeature importance saved successfully!")

else:
    print("Feature importance is not available for this model.")

# Step 12: Save actual vs predicted sales

comparison = df.loc[
    test_mask,
    ["date", "product_id", "product_name", "units_sold"]
].copy()

comparison = comparison.rename(
    columns={"units_sold": "Actual"}
)

comparison["Predicted"] = predictions
comparison["Error"] = comparison["Actual"] - comparison["Predicted"]

comparison["Absolute_Error"] = comparison["Error"].abs()

comparison.to_csv(
    output_dir / "test_predictions.csv",
    index=False
)

print("\nActual vs Predicted Sales:")
print(comparison.head(15))

print("\nTest predictions saved successfully!")

print("\nActual vs Predicted Sales:")
print(comparison.head(15))

# Plot comparison
plt.figure(figsize=(12, 5))

plt.plot(
    comparison["Actual"].values[:50],
    label="Actual Sales",
    marker="o"
)

plt.plot(
    comparison["Predicted"].values[:50],
    label="Predicted Sales",
    marker="o"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Records")
plt.ylabel("Units Sold")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()