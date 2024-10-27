import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder= "template")

# Load the model
model = pickle.load(open("logistic_regression_model.pickle", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input from the form with the updated field names
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width = float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    petal_width = float(request.form.get('petal_width'))

    # Assuming the model needs these inputs in the same order
    features = [sepal_length, sepal_width, petal_length, petal_width]
    feature = [np.array(features)]
    
    prediction = model.predict(feature)
    output = prediction[0]

    return render_template("index.html", prediction_text=f"The predicted flower is: ${output}")

if __name__ == '__main__':
    app.run(debug=True)

