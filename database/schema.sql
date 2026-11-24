
-- Smart Inventory Forecasting System
-- MySQL Database Schema

CREATE DATABASE IF NOT EXISTS smart_inventory_db;

USE smart_inventory_db;

-- Historical sales records
CREATE TABLE IF NOT EXISTS sales (
    id INT NOT NULL AUTO_INCREMENT,
    date DATE NOT NULL,
    product_id INT NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(100) DEFAULT NULL,
    units_sold INT NOT NULL,
    unit_price DECIMAL(12,2) NOT NULL,
    revenue DECIMAL(14,2) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_0900_ai_ci;

-- Current inventory information
CREATE TABLE IF NOT EXISTS inventory (
    product_id INT NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    current_stock INT NOT NULL,
    lead_time_days INT NOT NULL,
    last_updated TIMESTAMP NULL
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id)
) ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_0900_ai_ci;



-- Sample inventory records for initial setup
-- These are demonstration values, not live inventory.

INSERT INTO inventory
    (product_id, product_name, current_stock, lead_time_days)
VALUES
    (101, 'Laptop', 15, 5),
    (102, 'Mobile', 20, 4),
    (103, 'Tablet', 25, 3),
    (104, 'Mouse', 30, 3),
    (105, 'Keyboard', 18, 4)
ON DUPLICATE KEY UPDATE
    product_id = product_id;
