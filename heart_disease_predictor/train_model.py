import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('Heart Disease.csv')

# Select 12 features for training
selected_features = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 
                     'PhysicalHealth', 'MentalHealth', 'DiffWalking', 
                     'AgeCategory', 'Diabetic', 'PhysicalActivity', 
                     'SleepTime', 'GenHealth']

# Encode categorical columns (Label encoding for simplicity)
label_encoders = {}
categorical_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
                    'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store label encoders for future use if needed

# Normalize the numerical columns
scaler = StandardScaler()
numerical_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split the data into features (X) and target (y)
X = data[selected_features]  # Features with only 12 selected columns
y = data['HeartDisease']  # Target

# Encode target variable (Yes/No to 1/0)
y = LabelEncoder().fit_transform(y)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(log_reg, 'heart_disease_model_12_features.pkl')
