from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', mothods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({f"Predicted Price": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
