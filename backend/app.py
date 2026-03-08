from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json
    text = data["news"]

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)