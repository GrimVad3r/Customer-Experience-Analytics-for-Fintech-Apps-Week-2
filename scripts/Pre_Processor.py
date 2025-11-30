import pandas as pd


def preprocess_data(df,output_file='../data/bank_reviews_clean.csv'):

    # Step 1
    """Cleans the raw dataframe."""
    print("Preprocessing data...")

    clean_df = df[['reviewId'.'content', 'score', 'at', 'bank', 'source']]
    clean_df.columns = ['reviewId','review', 'rating', 'date', 'bank', 'source']

    clean_df['date'] = pd.to_datetime(clean_df['date']).dt.date
    clean_df.dropna(subset=['review'], inplace=True)
    clean_df.drop_duplicates(subset=['review', 'date', 'bank'], keep='first', inplace=True)

    # Step 2 Saves the Cleanded Dtaframe object

    clean_df.to_csv(output_file, index=False)

    print(f"Task Complete: Cleaned Data is saved to '{output_file}'")

    return clean_df