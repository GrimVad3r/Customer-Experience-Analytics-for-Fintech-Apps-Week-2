import pandas as pd
from sqlalchemy import create_engine
import os # Imported for best practice, though connection details are hardcoded in the original notebook

# --- 1. Database Connection Details ---
# NOTE: Replace these with your actual credentials for execution
DB_HOST = "localhost"
DB_NAME = "bank_reviews"
DB_USER = "postgres"
DB_PASS = "12345678"
DB_PORT = "5432"

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create the SQLAlchemy Engine
try:
    engine = create_engine(DB_URL, echo=False)
    print("✅ SQLAlchemy Engine created.")
except Exception as e:
    print(f"❌ Error creating SQLAlchemy Engine: {e}")
    exit() # Exit if connection fails

# --- 2. Load and Prepare Data ---
try:
    # Assuming task2_results.csv is in the '../data/' directory as in the notebook
    df = pd.read_csv('../data/task2_results.csv')
    print(f"✅ Loaded raw data: {len(df)} rows.")
except FileNotFoundError:
    print("❌ Error: '../data/task2_results.csv' not found. Check file path.")
    exit()

# Define the function to normalize bank names to bank_id (CBE, AWASH, BOA)
def adjuster(bank):
    """Maps bank name to its corresponding bank_id for the foreign key constraint."""
    if bank == "Abyssinia Bank":
        return "BOA"
    elif bank == "Awash Bank":
        return "AWASH"
    else: # Defaulting to CBE if not one of the others, based on original logic
        return "CBE"

# Prepare the DataFrame for insertion
df_clean = df.copy()
df_clean['bank'] = df['bank'].apply(adjuster)

# Select and rename columns to match the 'reviews_table' schema
df_clean = df_clean[[
    'reviewId',
    'bank',
    'review',
    'rating',
    'date',
    'sentiment_label',
    'sentiment_score',
    'source'
]]

df_clean.rename(
    columns={
        'reviewId': 'review_id',
        'bank': 'bank_id',
        'review': 'review_text',
        'date': 'review_date'
    },
    inplace=True
)

print(f"✅ Data cleaned and prepared. Total rows to insert: {len(df_clean)}")

# --- 3. Insert Data into PostgreSQL ---
table_name = "reviews_table"

try:
    with engine.connect() as connection:
        with connection.begin():  # Starts a transaction block
            # Use to_sql to insert the DataFrame into the reviews_table
            df_clean.to_sql(
                name=table_name,
                con=connection,  # Use the connection object
                if_exists='append',
                index=False,
                method='multi'  # 'multi' is generally faster for large inserts
            )
            print(f"✅ Successfully loaded {len(df_clean)} rows into the '{table_name}'.")
    # Changes are committed upon successful exit of the second 'with' block.

except Exception as e:
    print(f"❌ An error occurred during data loading: {e}")

# --- 4. Verification Query ---
check_query = f"SELECT count(*) FROM {table_name}"

try:
    count_result = pd.read_sql(check_query, engine).iloc[0, 0]
    print(f"\n✅ Verification: Current row count in '{table_name}': {count_result}")
except Exception as e:
    print(f"❌ Error running verification query: {e}")