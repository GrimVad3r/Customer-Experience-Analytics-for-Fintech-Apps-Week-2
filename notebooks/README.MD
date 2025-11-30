# 📊 KIAM8: WEEK 2 - Task 1: Data Collection and Preprocessing

This document summarizes the process outlined in `task-1.ipynb` for collecting and cleaning customer reviews for key Ethiopian mobile banking applications from the Google Play Store.

---

## 🚀 Workflow Summary

The task successfully executes a two-stage pipeline:

1.  **Data Scraping**: Collects raw reviews, ratings, and associated metadata using the `scrapper.py` module.
2.  **Data Preprocessing**: Cleans and transforms the raw data into an analysis-ready format using the `Pre_Processor.py` module.

The code leverages external Python scripts (`scrapper.py` and `Pre_Processor.py`) and dynamically adds the `scripts` folder to the system path to enable module imports.


---

## 🎯 Target Applications

The scraping focused on three major Ethiopian commercial banks:

| Bank Name | App ID (Package Name) |
| :--- | :--- |
| **CBE** | `com.combanketh.mobilebanking` |
| **Awash Bank** | `com.sc.awashpay` |
| **Abyssinia Bank** | `com.boa.boaMobileBanking` |

---

## 1. Web Scraping (`scrapper.py`)

The `scrapper.py` script was used to fetch a minimum of **400 reviews per bank** (1,200 total target).

### ⚙️ Execution Details

* **Function Used**: `scrapper.run_scraper(APPS, count=500)`
    * *Note*: The `count` parameter was implicitly set to the default of 500, resulting in a total of **1500** reviews collected (500 per bank).
* **Sorting**: Reviews were collected using the `Sort.NEWEST` criterion.
* **Output File**: `../data/bank_reviews_raw.csv`

### 🔍 Raw Data Validation

The initial scraping process resulted in a DataFrame with **1500 total reviews**. The validation check revealed that `replyContent` and `repliedAt` columns were entirely null, and `reviewCreatedVersion` and `appVersion` had significant missing data, which is expected as bank staff rarely reply to reviews and version data is optional.

| Column | Missing Count | Note |
| :--- | :--- | :--- |
| `replyContent` | 1500 | Bank replies are not present for any scraped review. |
| `repliedAt` | 1500 | |
| `reviewCreatedVersion` | 335 | Version information is missing for some entries. |
| `appVersion` | 335 | |

---

## 2. Data Preprocessing (`Pre_Processor.py`)

The `Pre_Processor.py` script was applied to clean the raw DataFrame and prepare it for sentiment and text analysis.

### 🧹 Cleaning Steps

1.  **Column Selection & Rename**: Reduced the DataFrame to five essential columns and renamed them for clarity:
    * `content` $\rightarrow$ **`review`**
    * `score` $\rightarrow$ **`rating`**
    * `at` $\rightarrow$ **`date`**
    * `bank`
    * `source`
2.  **Date Normalization**: Converted the timestamp in the `date` column to a standard **YYYY-MM-DD** date format.
3.  **Missing Data Handling**: Rows with a null value in the **`review`** content were removed.
4.  **Deduplication**: Duplicate entries were dropped based on the combination of `review`, `date`, and `bank`.

### ✅ Cleaned Data Result

* **Function Used**: `preprocess_data(Scrapped_DF)`
* **Final Data Size**: **1479** rows and **5** columns.
    * *Note*: The reduction from 1500 to 1479 rows indicates **21 rows** were removed during the missing data and/or deduplication steps.
* **Final Columns**: `review`, `rating`, `date`, `bank`, `source`.
* **Missing Data Check**: The final DataFrame has **0 missing values** in any of the five final columns.
* **Output File**: `../data/bank_reviews_clean.csv`