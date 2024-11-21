from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd

# Dataset with Weather and Temperature
data = [
    {"Weather": "Sunny", "Temperature": "Hot", "Play": "No"},
    {"Weather": "Sunny", "Temperature": "Mild", "Play": "No"},
    {"Weather": "Overcast", "Temperature": "Hot", "Play": "Yes"},
    {"Weather": "Overcast", "Temperature": "Cool", "Play": "Yes"},
    {"Weather": "Rainy", "Temperature": "Cool", "Play": "No"},
    {"Weather": "Rainy", "Temperature": "Mild", "Play": "Yes"},
    {"Weather": "Rainy", "Temperature": "Hot", "Play": "Yes"},
    {"Weather": "Sunny", "Temperature": "Cool", "Play": "No"},
    {"Weather": "Sunny", "Temperature": "Hot", "Play": "Yes"},
    {"Weather": "Rainy", "Temperature": "Cool", "Play": "Yes"},
    {"Weather": "Overcast", "Temperature": "Mild", "Play": "Yes"},
    {"Weather": "Sunny", "Temperature": "Mild", "Play": "Yes"},
    {"Weather": "Overcast", "Temperature": "Cool", "Play": "Yes"},
    {"Weather": "Rainy", "Temperature": "Hot", "Play": "No"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical data to numeric
encoded_columns = {}
for col in df.columns:
    df[col], unique_values = pd.factorize(df[col])
    encoded_columns[col] = unique_values

# Split features and target
X = df.drop(columns="Play")
y = df["Play"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = CategoricalNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No", "Yes"])

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", report)

# Prediction for Weather=Overcast and Temperature=Mild
test_instance = {"Weather": "Overcast", "Temperature": "Mild"}
test_instance_encoded = [
    list(encoded_columns[col]).index(test_instance[col]) for col in test_instance
]
test_prediction = model.predict([test_instance_encoded])
decoded_prediction = encoded_columns["Play"][test_prediction[0]]

print("\nPredicted Class for Overcast/Mild:", decoded_prediction)
