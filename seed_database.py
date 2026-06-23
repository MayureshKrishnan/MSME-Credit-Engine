import sqlite3
import pandas as pd

print("Initializing local bank database...")

# 1. Pull the raw source data from the web repository
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
raw_data = pd.read_csv(data_url, sep=r'\s+', header=None)

# 2. Establish a connection to create a local database file
conn = sqlite3.connect("central_bank_vault.db")

# 3. Export the raw data frame directly into a SQL table named 'msme_portfolio_ledger'
raw_data.to_sql("msme_portfolio_ledger", conn, if_exists="replace", index=False)

print("• SQL Table 'msme_portfolio_ledger' successfully seeded into 'central_bank_vault.db'!")
conn.close()