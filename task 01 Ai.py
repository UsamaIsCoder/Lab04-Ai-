from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dataset
data = [
    ['Sunny', 'Hot', 'No'],
    ['Sunny', 'Hot', 'No'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rainy', 'Mild', 'Yes'],
    ['Rainy', 'Cool', 'Yes'],
    ['Rainy', 'Cool', 'No'],
    ['Overcast', 'Cool', 'Yes'],
    ['Sunny', 'Mild', 'No'],
    ['Sunny', 'Cool', 'Yes'],
    ['Rainy', 'Mild', 'Yes'],
    ['Sunny', 'Mild', 'Yes'],
    ['Overcast', 'Mild', 'Yes'],
    ['Overcast', 'Hot', 'Yes'],
    ['Rainy', 'Mild', 'No']
]

# Splitting features (Weather, Temperature) and target (Play)
X = [row[:2] for row in data]  # Features: Weather and Temperature
y = [row[2] for row in data]  # Target: Play

# Encode categorical data into numerical format
weather_encoder = LabelEncoder()
temp_encoder = LabelEncoder()
play_encoder = LabelEncoder()

# Encoding features and target
weather_encoded = weather_encoder.fit_transform([row[0] for row in X])
temp_encoded = temp_encoder.fit_transform([row[1] for row in X])
play_encoded = play_encoder.fit_transform(y)

# Combine encoded features into a single array
X_encoded = np.array(list(zip(weather_encoded, temp_encoded)))

# Create and train the Naïve Bayes model
model = CategoricalNB()
model.fit(X_encoded, play_encoded)

# Encoding the test case (Weather = Overcast, Temperature = Mild)
test_weather = weather_encoder.transform(['Overcast'])[0]
test_temp = temp_encoder.transform(['Mild'])[0]
test_data = np.array([[test_weather, test_temp]])

# Predicting the output
prediction = model.predict(test_data)
predicted_class = play_encoder.inverse_transform(prediction)

print(f"Prediction for Weather=Overcast and Temperature=Mild: {predicted_class[0]}")
