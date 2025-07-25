# necessary imports
from flask import Flask, request, jsonify
import joblib
import numpy as np

# initialize flask and load the model
app = Flask(__name__)
model = joblib.load("loan_model.pkl")

@app.route("/")
def home():
    return "Loan Approval Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    features = np.array([[
        data["Gender"],
        data["Married"],
        data["Dependents"],
        data["Education"],
        data["Self_Employed"],
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Loan_Amount_Term"],
        data["Credit_History"],
        data["Property_Area"]
    ]])

    prediction = model.predict(features)[0]
    result = "Approved" if prediction == 1 else "Denied"
    return jsonify({"loan_approval": result})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

