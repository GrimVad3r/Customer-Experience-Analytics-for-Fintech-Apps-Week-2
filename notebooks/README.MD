# Bank Reviews Data Engineering Project: PostgreSQL Schema

This document outlines the relational database schema designed for persistently storing cleaned and processed bank review data using **PostgreSQL**. The schema consists of two tables: `banks_table` and `reviews_table`, linked by a foreign key constraint.


## 1. Schema Overview

The design follows a basic star schema pattern where the `reviews_table` is the fact table containing individual review data, and the `banks_table` serves as a small dimension table for bank-specific information.

## 2. Table: `banks_table`

This table stores unique information about the banks being analyzed.

| Column Name | Data Type | Constraint | Description |
| :--- | :--- | :--- | :--- |
| **bank_id** | `VARCHAR(50)` | **PRIMARY KEY**, NOT NULL | A short, unique identifier for the bank (e.g., 'CBE', 'AWASH', 'BOA'). |
| **bank_name** | `VARCHAR(100)` | NOT NULL | The full, official name of the bank (e.g., 'Commercial Bank of Ethiopia'). |
| **app_name** | `VARCHAR(100)` | | The package name or identifier for the bank's mobile application. |

## 3. Table: `reviews_table`

This table stores the cleaned review records, including the sentiment analysis results. It links to the `banks_table` via the `bank_id` foreign key.

| Column Name | Data Type | Constraint | Description |
| :--- | :--- | :--- | :--- |
| **review_id** | `VARCHAR(50)` | **PRIMARY KEY**, NOT NULL | A unique identifier for the individual review (UUID format). |
| **bank_id** | `VARCHAR(50)` | **FOREIGN KEY** (references `banks_table`) | The ID of the bank the review is about. |
| **review_text** | `VARCHAR` | | The original, raw text of the customer review. |
| **rating** | `INTEGER` | | The numerical star rating given by the user (e.g., 1 to 5). |
| **review_date** | `DATE` | | The date the review was posted. |
| **sentiment_label** | `VARCHAR` | | The classified sentiment: 'positive' or 'negative'. |
| **sentiment_score** | `DOUBLE PRECISION` | | The model's confidence score for the sentiment classification. |
| **source** | `VARCHAR` | | The platform where the review was collected (e.g., 'Google Play'). |
