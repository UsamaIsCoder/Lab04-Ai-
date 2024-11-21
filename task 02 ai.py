from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd

# Dataset
data = [
    {"Age": "Youth", "Income": "High", "Student": "No", "Credit": "Fair", "Buys": "No"},
    {"Age": "Youth", "Income": "High", "Student": "No", "Credit": "Excellent", "Buys": "No"},
    {"Age": "Middle", "Income": "High", "Student": "No", "Credit": "Fair", "Buys": "Yes"},
    {"Age": "Senior", "Income": "Medium", "Student": "No", "Credit": "Fair", "Buys": "Yes"},
    {"Age": "Senior", "Income": "Low", "Student": "Yes", "Credit": "Fair", "Buys": "Yes"},
    {"Age": "Senior", "Income": "Low", "Student": "Yes", "Credit": "Excellent", "Buys": "No"},
    {"Age": "Middle", "Income": "Low", "Student": "Yes", "Credit": "Excellent", "Buys": "Yes"},
    {"Age": "Youth", "Income": "Medium", "Student": "No", "Credit": "Fair", "Buys": "No"},
    {"Age": "Youth", "Income": "Low", "Student": "Yes", "Credit": "Fair", "Buys": "Yes"},
    {"Age": "Senior", "Income": "Medium", "Student": "Yes", "Credit": "Fair", "Buys": "Yes"},
    {"Age": "Youth", "Income": "Medium", "Student": "Yes", "Credit": "Excellent", "Buys": "Yes"},
    {"Age": "Middle", "Income": "Medium", "Student": "No", "Credit": "Excellent", "Buys": "Yes"},
    {"Age": "Middle", "Income": "High", "Student": "Yes", "Credit": "Fair", "Buys": "Yes"},
    {"Age": "Senior", "Income": "Medium", "Student": "No", "Credit": "Excellent", "Buys": "No"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical data to numeric
encoded_columns = {}
for col in df.columns:
    df[col], unique_values = pd.factorize(df[col])
    encoded_columns[col] = unique_values

# Split features and target
X = df.drop(columns="Buys")
y = df["Buys"]

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

# Prediction for Age=Youth, Income=Medium, Student=Yes, Credit=Fair
test_instance = {"Age": "Youth", "Income": "Medium", "Student": "Yes", "Credit": "Fair"}
test_instance_encoded = [
    list(encoded_columns[col]).index(test_instance[col]) for col in test_instance
]
test_prediction = model.predict([test_instance_encoded])
decoded_prediction = encoded_columns["Buys"][test_prediction[0]]

print("\nPredicted Class for Youth/Medium/Yes/Fair:", decoded_prediction)
