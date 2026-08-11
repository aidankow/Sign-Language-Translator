import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
print("Loading dataset...")
data = pd.read_csv("dataset.csv", header=None)

# Separate features (columns 1 to 63) and labels (column 0)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 2. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Save the trained model to a file
joblib.dump(model, "model.joblib")
print("Model successfully saved as 'model.joblib'!")