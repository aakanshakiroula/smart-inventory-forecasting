
import unittest

from src.inventory_logic import (
    calculate_reorder_point,
    calculate_reorder_quantity,
    get_stock_status
)


class TestInventoryLogic(unittest.TestCase):

    def test_reorder_point(self):
        result = calculate_reorder_point(15, 4, 10)
        self.assertEqual(result, 70)

    def test_reorder_quantity(self):
        result = calculate_reorder_quantity(
            predicted_demand=15,
            lead_time_days=4,
            safety_stock=10,
            current_stock=20
        )
        self.assertEqual(result, 155)

    def test_no_reorder_when_stock_sufficient(self):
        result = calculate_reorder_quantity(
            predicted_demand=15,
            lead_time_days=4,
            safety_stock=10,
            current_stock=100
        )
        self.assertEqual(result, 0)

    def test_low_stock_status(self):
        self.assertEqual(
            get_stock_status(20, 70),
            "REORDER"
        )

    def test_sufficient_stock_status(self):
        self.assertEqual(
            get_stock_status(100, 70),
            "STOCK AVAILABLE"
        )
    
    def test_zero_demand(self):
        quantity = calculate_reorder_quantity(
            predicted_demand=0,
            lead_time_days=4,
            safety_stock=0,
            current_stock=0
        )
        self.assertEqual(quantity, 0)

    def test_zero_stock(self):
        quantity = calculate_reorder_quantity(
            predicted_demand=10,
            lead_time_days=3,
            safety_stock=5,
            current_stock=0
        )
        self.assertEqual(quantity, 105)

    def test_stock_equal_to_reorder_point(self):
        reorder_point = calculate_reorder_point(
            predicted_demand=10,
            lead_time_days=3,
            safety_stock=5
        )
        self.assertEqual(reorder_point, 35)
        self.assertEqual(
            get_stock_status(35, reorder_point),
            "REORDER"
        )

    def test_round_up_order_quantity(self):
        quantity = calculate_reorder_quantity(
            predicted_demand=2.5,
            lead_time_days=2,
            safety_stock=1,
            current_stock=5
        )
        self.assertEqual(quantity, 19)


if __name__ == "__main__":
    unittest.main()
