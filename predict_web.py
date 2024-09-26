from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    # Get Data and convert to DataFrame
    data = request.json
    input_data = pd.DataFrame([data])

    # Prediction
    y_new_pred = model.predict(input_data)

    return jsonify({"predicted_price": y_new_pred[0]})


if __name__ == '__main__':
    app.run(debug=True)
