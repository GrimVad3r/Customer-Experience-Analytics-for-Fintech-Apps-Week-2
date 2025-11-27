# Customer-Experience-Analytics-for-Fintech-Apps-Week-2
KIAM 8: Week 2 Challenge Repo

## 1. Project Overview
 
**Domain:** Fintech / Banking  
**Focus:** Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank

This project involves analyzing customer satisfaction with mobile banking apps to enhance user retention and service quality. By scraping user reviews from the Google Play Store, we aim to uncover satisfaction drivers and pain points, providing actionable insights to engineering and product teams

## 2. Business Objective

* Scrape user reviews to gather raw feedback[cite: 15].
* Analyze sentiment (positive/negative/neutral) and identify themes (e.g., "bugs", "UI")
* Identify specific satisfaction drivers (e.g., speed) and pain points (e.g., crashes).
* Persistently store cleaned data in a PostgreSQL database.
* Deliver a data-driven report with actionable recommendations.

## 3. Project Structure & Tasks

### Task 1: Data Collection & Preprocessing

* **Objective:** Scrape reviews from the Google Play Store and prepare the dataset.
* **Tools:** `google-play-scraper`, Pandas.
* **Requirements:**
    * Collect 400+ reviews per bank (1,200 total).
    * Fields: Review Text, Rating, Date, Bank Name, Source.
    * Preprocess data (remove duplicates, normalize dates).

### Task 2: Sentiment & Thematic Analysis

* **Objective:** Quantify sentiment and extract recurring themes.
* **Tools:** Hugging Face Transformers, Scikit-learn, spaCy.
* **Methodology:**
    * **Sentiment:** Used `distilbert-base-uncased-finetuned-sst-2-english` to compute sentiment scores.
    * **Thematic Analysis:** Extracted keywords using TF-IDF/spaCy and clustered them into themes (e.g., "Account Access Issues", "Transaction Performance").

### Task 3: Database Engineering (PostgreSQL)

* **Objective:** Design and implement a relational database to store the data[cite: 122].
* **Schema:**
    * **`banks` Table:** `bank_id`, `bank_name`, `app_name`.
    * **`reviews` Table:** `review_id`, `bank_id`, `review_text`, `rating`, `review_date`, `sentiment_label`, `sentiment_score`.
* **Implementation:** Data insertion via Python (`psycopg2` or `SQLAlchemy`).

### Task 4: Insights & Visualization

* **Objective:** Visualize trends and provide recommendations.
* **Tools:** Matplotlib, Seaborn.
* **Deliverables:**
    * Identification of key drivers and pain points.
    * Visualizations of sentiment trends and rating distributions.
    * Comparative analysis between CBE, BOA, and Dashen.

## 4. Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Web Scraping:** `google-play-scraper` 
* **NLP:** * Hugging Face (`transformers`) 
    * `scikit-learn` (TF-IDF) 
    * `spaCy` 
* **Database:** PostgreSQL, `psycopg2`, `SQLAlchemy` 
* **Visualization:** Matplotlib, Seaborn 

## 5. Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/GrimVad3r/Customer-Experience-Analytics-for-Fintech-Apps-Week-2
    cd Customer-Experience-Analytics-for-Fintech-Apps-Week-2
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Database Setup:**
    * Ensure PostgreSQL is installed and running[cite: 126].
    * Create a database named `bank_reviews`[cite: 129].
    * Update database connection credentials in the scripts.

4.  **Running the Pipeline:**
    * **Step 1:** Run the scraper script (Task 1) to generate the raw CSV.
    * **Step 2:** Run the analysis script (Task 2) to generate sentiment scores and themes.
    * **Step 3:** Run the database loader (Task 3) to populate the PostgreSQL tables.
    * **Step 4:** Launch the notebook/script for Task 4 to generate visualizations.

