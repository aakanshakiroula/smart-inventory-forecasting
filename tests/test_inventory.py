
import unittest
import pandas as pd


class TestInventorySystem(unittest.TestCase):

    # Test 1: Predicted demand cannot be negative
    def test_negative_prediction(self):
        predictions = pd.Series([-5, 10, 15])
        predictions = predictions.clip(lower=0)

        self.assertTrue((predictions >= 0).all())

    # Test 2: Reorder point calculation
    def test_reorder_point(self):
        predicted_demand = 15
        lead_time_days = 4
        safety_stock = 10

        reorder_point = (
            predicted_demand * lead_time_days
            + safety_stock
        )

        self.assertEqual(reorder_point, 70)

    # Test 3: Stock below reorder point
    def test_reorder_status(self):
        current_stock = 20
        reorder_point = 70

        status = (
            "REORDER"
            if current_stock <= reorder_point
            else "STOCK AVAILABLE"
        )

        self.assertEqual(status, "REORDER")

    # Test 4: No reorder when stock is sufficient
    def test_sufficient_stock(self):
        current_stock = 100
        reorder_point = 70

        reorder_quantity = max(0, 90 - current_stock)

        if current_stock > reorder_point:
            reorder_quantity = 0

        self.assertEqual(reorder_quantity, 0)


if __name__ == "__main__":
    unittest.main()
