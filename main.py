import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# warnings.filterwarnings('ignore')

today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'

# Downloading BTC data
print("\nBTC data\n")
btc_df = yf.download('BTC-USD', start_date, today)
btc_df.reset_index(inplace=True)
print(btc_df.columns)
btc_df.drop(columns=['Open', 'Adj Close'], inplace=True)
print(btc_df.tail())
btc_df.info()

# Downloading S&P 500 data
print("\nS&P 500 data\n")
snp_df = yf.download('^GSPC', start_date, today)
snp_df.tail()
snp_df.reset_index(inplace=True)
print(snp_df.columns)
snp_df = snp_df[['Date', 'Close']]
snp_df.rename(columns={'Close': 'SNP'}, inplace=True)
print(snp_df.tail())
snp_df.info()

# Download dollar basket price data from https://www.investing.com/indices/usdollar-historical-data
# Load the dxy CSV file into a DataFrame
print("\nDXY data\n")
dxy_df = pd.read_csv('US Dollar Index Historical Data.csv')
# print(dxy_df.head())
dxy_df = dxy_df[['Date', 'Price']]
dxy_df.rename(columns={'Price': 'DXY'}, inplace=True)
dxy_df.info()

# Download gold price data from https://www.investing.com/currencies/xau-usd-historical-data
# Load the XAU CSV file into a DataFrame
print("\ngold data\n")
xau_df = pd.read_csv('XAU_USD Historical Data.csv')
# print(xau_df.head())
xau_df = xau_df[['Date', 'Price']]
xau_df['Price'] = xau_df['Price'].str.replace(',', '').astype(float)
xau_df.rename(columns={'Price': 'Gold'}, inplace=True)
xau_df.info()

# Download Fed base rate data from https://www.federalreserve.gov/datadownload/default.htm
# Load the Fed base rate CSV file into a DataFrame
print("\nFed interest rate data\n")
fed_df = pd.read_csv('FRB_H15.csv', skiprows=6, header=None, names=['Date', 'Rate'])
# print(fed_df.head())
fed_df.info()

# Download Bitcoin fear and greed index data
# Define the URL of the API
print("\nFear and Greed index data\n")
url = "https://api.alternative.me/fng/?limit=0&date_format=cn"

# Make a GET request to the API
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the content (CSV data)
    json_data = response.content.decode('utf-8')
else:
    print(f"Error: {response.status_code}")

parsed_data = json.loads(json_data)
fng_df = pd.DataFrame(parsed_data['data'])

print(fng_df.tail())
fng_df = fng_df[['value', 'timestamp']]
fng_df['value'] = fng_df['value'].astype(int)
fng_df.rename(columns={'value': 'FNG', 'timestamp': 'Date'}, inplace=True)
fng_df.info()

dataframes = [btc_df, snp_df, dxy_df, xau_df, fed_df, fng_df]

for df in dataframes:
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Format 'Date' column as 'dd-mm-yyyy'
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.info()
    print(df.tail())


for df in dataframes[1:]:
    # Perform a left join using the 'Date' column from the first DataFrame
    btc_df = pd.merge(btc_df, df, on='Date', how='left')

# Display the final merged DataFrame
# print(btc_df.head())
# print(btc_df.tail())

# Fill NaN values with previous non-NaN values in the same column
btc_df = btc_df.fillna(method='ffill')
btc_df["Tomorrow"] = btc_df['Close'].shift(-1)
btc_df["Target"] = (btc_df["Tomorrow"] > btc_df["Close"]).astype(int)
print(btc_df.tail())
btc_df.info()

# Creating moving averages

intervals = [10, 50, 200]

for interval in intervals:
    moving_average = f"{interval}-day_ma"
    btc_df[moving_average] = btc_df['Close'].rolling(interval, closed='left').mean()

print(btc_df.head())
print(btc_df.tail())
btc_df.info()

# transforming 'Date'

btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df['Year'] = btc_df['Date'].dt.year
btc_df['Month'] = btc_df['Date'].dt.month

btc_df['Day'] = btc_df['Date'].dt.day
btc_df['DayOfWeek'] = btc_df['Date'].dt.dayofweek
btc_df = btc_df.drop('Date', axis=1)

first_fng_index = btc_df['FNG'].first_valid_index()
print(first_fng_index)

btc_df_fng = btc_df.iloc[762:]
btc_df_fng.reset_index(drop=True, inplace=True)
print(btc_df_fng.head())
print(btc_df_fng.tail())
btc_df_fng.info()
print(btc_df_fng.dtypes)

btc_df_fng.drop(btc_df_fng.tail(1).index, inplace=True)

columns = btc_df_fng.columns

# Iterate through the column names
for column in columns:
    # Check for NaN values in the column
    nan_count = btc_df_fng[column].isna().sum()

    # Print the column name and the count of NaN values
    print(f'Column "{column}" has {nan_count} NaN values.')
# #models
# Define features and target variable
X = btc_df_fng.drop('Target', axis=1)
y = btc_df_fng['Target']

# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Time-based cross-validation
n = len(btc_df_fng)
train_size = int(0.8 * n)  # Use 80% of the data for training
predictions = []
actuals = []
for i in range(train_size, n):
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:i+1]  # Use the next data point for testing
    y_test = y[i:i+1]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    predictions.append(y_pred)
    actuals.append(y_test.values[0])

# Convert the lists to numpy arrays for easier handling
predictions = np.array(predictions)
actuals = np.array(actuals)

# Evaluate the model
accuracy = accuracy_score(actuals, predictions)
report = classification_report(actuals, predictions)

print(f'Overall Accuracy: {accuracy:.2f}')
print(f'Overall Classification Report:\n{report}')
