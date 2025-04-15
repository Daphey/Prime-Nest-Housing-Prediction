import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Load the dataset
df = pd.read_csv('apartment_price_data.csv')

# Separate features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error:{rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")


### Streamlit App
import streamlit as st
# import numpy as np

# Function to load the model
def load_model():
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions based on user input
def predict_price(model, num_rooms, parking_space, width, num_conveniences, building_age, distance_to_center, location):
    # Prepare the input data as a DataFrame with the same column names
    input_data = pd.DataFrame({
        'num_rooms': [num_rooms],
        'parking_space': [parking_space],
        'width': [width],
        'num_conveniences': [num_conveniences],
        'building_age': [building_age],
        'distance_to_center': [distance_to_center],
        'location': [location]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Calling the model function (load_model)
model = load_model()

## Web APP Interface Layout
st.title("Apartment Price Prediction")
st.subheader(" This app predicts the price of an apartment or building based on certain features")

# Input features for users to enter
num_rooms = st.slider("Number of Rooms", 1, 6, 3)
parking_space = st.selectbox("Parking Space", [0, 1, 2])
width   = st.slider("Width in square meters", 30, 300, 100)
num_conveniences = st.slider("Number of Conveniences", 1, 4, 2)
building_age = st.slider("Building Age in years", 1, 50, 10)
distance_to_center = st.slider("Distance in km", 1, 50, 5)
location = st.selectbox("Location (1 = City Center, 5 = Outskirts)", [1, 2, 3, 4, 5])


# Button to trigger the prediction
if st.button("Predict Price"):
    predicted_price = predict_price(model, num_rooms, parking_space, width, num_conveniences, building_age, distance_to_center, location)
    st.write(f"Predicted Price: ${predicted_price:,.2f}")

