import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
import joblib
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.dates as mdates


# Set ticker input
ticker = input("Enter the stock ticker: ")

# Fetch data
data = yf.download(ticker, start='2010-01-01', end='2024-11-05')
data = data.reset_index()

data_2010 = data[data['Date'].dt.year >= 2010]

print("\nPerforming feature engineering...")
data_2010['Returns'] = data_2010['Close'].pct_change()
data_2010['Log_Returns'] = np.log(data_2010['Close'] / data_2010['Close'].shift(1)) #used for long term average values of stocks
data_2010['MA_5'] = data_2010['Close'].rolling(window=5).mean()  #used for short term average values of stocks
data_2010['MA_20'] = data_2010['Close'].rolling(window=20).mean()
data_2010['Volatility'] = data_2010['Returns'].rolling(window=20).std() # measures standard deviation, how much swings there are in stock prices in a day
data_2010['Price_Momentum'] = data_2010['Close'] / data_2010['Close'].shift(5) - 1 #the rate of stock prices in a 5-day period

data_2010 = data_2010.dropna()


# Define selected features and target attribute
features = ['Close', 'Open', 'Price_Momentum', 'Volatility']
target = "Close"


# Define start and end time for each period
train_end_date = pd.to_datetime("2022-12-31")
validate_start_date = pd.to_datetime("2023-01-01")
validate_end_date = pd.to_datetime("2023-12-31")
test_start_date = pd.to_datetime("2024-01-01")
test_end_date = pd.to_datetime("2024-10-27")

# Split dataset into training, validation, and testing
data_train = data_2010[data_2010["Date"] <= train_end_date][features]
data_train_dates = data_2010[data_2010["Date"] <= train_end_date]["Date"]
data_validate = data_2010[(data_2010["Date"] >= validate_start_date) & (data_2010["Date"] <= validate_end_date)][features]
data_validate_dates = data_2010[(data_2010["Date"] >= validate_start_date) & (data_2010["Date"] <= validate_end_date)]["Date"]
data_test = data_2010[(data_2010["Date"] >= test_start_date) & (data_2010["Date"] <= test_end_date)][features]
data_test_dates = data_2010[(data_2010["Date"] >= test_start_date) & (data_2010["Date"] <= test_end_date)]["Date"]

# Preprocessing
sc = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = sc.fit_transform(data_train)
data_validate_scaled = sc.fit_transform(data_validate)
data_test_scaled = sc.fit_transform(data_test)

scaler_model_location = r'C:\Users\Ayushman\Desktop\CODES\DIP_Scripts\scalers\\'
scaler_model_name = f'{ticker}_price_scaler'
scaler_model_ext = 'gz'
joblib.dump(sc, scaler_model_location + scaler_model_name + '.' + scaler_model_ext)

# Combine dates with each corresponding dataset
data_train_scaled_final = pd.DataFrame(data_train_scaled, columns=features, index=None)
data_train_scaled_final["Date"] = data_train_dates.values

data_validate_scaled_final = pd.DataFrame(data_validate_scaled, columns=features, index=None)
data_validate_scaled_final["Date"] = data_validate_dates.values

data_test_scaled_final = pd.DataFrame(data_test_scaled, columns=features, index=None)
data_test_scaled_final["Date"] = data_test_dates.values

data_train_scaled = data_train_scaled_final[features].values
data_validate_scaled = data_validate_scaled_final[features].values
data_test_scaled = data_test_scaled_final[features].values


sequence_size = 6  # Choose sequence size
forecast_days = 5

# Define function to create LSTM input sequences
def construct_lstm_data(data, sequence_size, target_attr_idx=0):
    data_X, data_y = [], []
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i, 0:data.shape[1]])
        data_y.append(data[i, target_attr_idx])
    return np.array(data_X), np.array(data_y)

X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size)

# Combine scaled datasets all together
data_all_scaled = np.concatenate([data_train_scaled, data_validate_scaled, data_test_scaled], axis=0)

# Calculate data size
train_size = len(data_train_scaled)
validate_size = len(data_validate_scaled)
test_size = len(data_test_scaled)

# Construct validation dataset
X_validate, y_validate = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size,:], sequence_size, 0)

# Construct testing dataset
X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):,:], sequence_size, 0)

# Initializing the model
regressor = Sequential()
# Add input layer
regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
regressor.add(LSTM(units = 100))
regressor.add(Dropout(rate = 0.2))
regressor.add(Dense(units = 1))

# Compiling the model
regressor.compile(optimizer = "adam", loss="mean_squared_error")


print(y_train.shape)
# Training the model
history = regressor.fit(
    x = X_train, 
    y = y_train, 
    validation_data=(X_validate, y_validate), 
    epochs= 10)


# Make predictions
y_train_predict = regressor.predict(X_train)
y_validate_predict = regressor.predict(X_validate)
y_test_predict = regressor.predict(X_test)


y_train_inv = sc.inverse_transform(np.column_stack((y_train, np.zeros((y_train.shape[0], 3)))))[:, 0]
y_validate_inv = sc.inverse_transform(np.column_stack((y_validate, np.zeros((y_validate.shape[0], 3)))))[:, 0]
y_test_inv = sc.inverse_transform(np.column_stack((y_test, np.zeros((y_test.shape[0], 3)))))[:, 0]

y_train_predict_inv = sc.inverse_transform(np.column_stack((y_train_predict, np.zeros((y_train_predict.shape[0], 3)))))[:, 0]
y_validate_predict_inv = sc.inverse_transform(np.column_stack((y_validate_predict, np.zeros((y_validate_predict.shape[0], 3)))))[:, 0]
y_test_predict_inv = sc.inverse_transform(np.column_stack((y_test_predict, np.zeros((y_test_predict.shape[0], 3)))))[:, 0]

# Define chart colors
train_actual_color = "cornflowerblue"
validate_actual_color = "orange"
test_actual_color = "green"
train_predicted_color = "lightblue"
validate_predicted_color = "peru"
test_predicted_color = "limegreen"


# Plot actual and predicted price
plt.figure(figsize=(18,6))
plt.plot(data_train_dates[sequence_size:,], y_train_inv, label="Training Data", color=train_actual_color)
plt.plot(data_train_dates[sequence_size:,], y_train_predict_inv, label="Training Predictions", linewidth=1, color=train_predicted_color)

plt.plot(data_validate_dates, y_validate_inv, label="Validation Data", color=validate_actual_color)
plt.plot(data_validate_dates, y_validate_predict_inv, label="Validation Predictions", linewidth=1, color=validate_predicted_color)

plt.plot(data_test_dates, y_test_inv, label="Testing Data", color=test_actual_color)
plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=1, color=test_predicted_color)

plt.title("AAPL Stock Price Predictions With LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.legend()
plt.grid(color="lightgray")

plt.savefig(fr"C:\Users\Ayushman\Desktop\CODES\DIP_Scripts\FIGURES\plot{ticker}.png")
plt.close()
