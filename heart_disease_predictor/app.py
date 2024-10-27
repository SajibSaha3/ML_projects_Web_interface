from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (now with 12 features)
model = joblib.load('heart_disease_model_12_features.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form (12 features)
    features = [float(request.form['bmi']),
                float(request.form['smoking']),
                float(request.form['alcohol_drinking']),
                float(request.form['stroke']),
                float(request.form['physical_health']),
                float(request.form['mental_health']),
                float(request.form['diff_walking']),
                float(request.form['age_category']),
                float(request.form['diabetic']),
                float(request.form['physical_activity']),
                float(request.form['sleep_time']),
                float(request.form['gen_health'])]

    # Predict heart disease
    features_array = np.array([features])  # Ensure it matches model training
    prediction = model.predict(features_array)[0]
    result = "Positive" if prediction == 1 else "Negative"
    
    # Pass data back to the template
    return render_template('index.html', bmi=request.form['bmi'],
                           smoking=request.form['smoking'],
                           alcohol_drinking=request.form['alcohol_drinking'],
                           stroke=request.form['stroke'],
                           physical_health=request.form['physical_health'],
                           mental_health=request.form['mental_health'],
                           diff_walking=request.form['diff_walking'],
                           age_category=request.form['age_category'],
                           diabetic=request.form['diabetic'],
                           physical_activity=request.form['physical_activity'],
                           sleep_time=request.form['sleep_time'],
                           gen_health=request.form['gen_health'],
                           prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
