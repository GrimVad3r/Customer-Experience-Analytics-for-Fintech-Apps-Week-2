import pandas as pd
from google_play_scraper import reviews, Sort
from datetime import datetime


def scrape_reviews(app_name, app_id, count=400):
    """Scrapes N reviews for a specific app."""
    print(f"Scraping {app_name}...")
    result, _ = reviews(
        app_id,
        lang='en',
        country='et',
        sort=Sort.NEWEST,
        count=count
    )

    df = pd.DataFrame(result)
    df['bank'] = app_name
    df['source'] = 'Google Play'
    return df


def run_scraper(APPS, count=500, output_file='../data/bank_reviews_raw.csv'):
    """
    Runs the scraping pipeline using a user-provided APPS dictionary.
    Example APPS input:
        {
            'CBE':'com.combanketh.mobilebanking',
            'Awash Bank': 'com.sc.awashpay',
            'Abyssinia Bank':'com.boa.boaMobileBanking'
        }
    """
    all_reviews = []

    # Step 1: Web Scraping
    for bank, app_id in APPS.items():
        try:
            bank_df = scrape_reviews(bank, app_id, count=count)
            all_reviews.append(bank_df)
        except Exception as e:
            print(f"Error scraping {bank}: {e}")

    # Step 2: Combine 
    full_df = pd.concat(all_reviews, ignore_index=True)


    # Step 3: Validation

    print(f"Total Reviews Collected: {len(full_df)}")
    print("Missing Data Count:")
    print(full_df.isnull().sum())

    # Step 4: Save

    full_df.to_csv(output_file, index=False)
    print(f"Task Complete: Raw Scapped Data saved to '{output_file}'")

    return full_df
