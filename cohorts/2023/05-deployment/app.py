from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction import DictVectorizer

app = Flask(__name__)

# Load the pickled DictVectorizer and LogisticRegression model
with open('dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)
    
with open('model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # Ensure that data is in the format of a list of dictionaries
        if not isinstance(data, list):
            raise ValueError("Data should be a list of dictionaries")
        
        # Vectorize the incoming data
        X = dv.transform(data)
        
        # Predict using the loaded model
        predictions = model.predict_proba(X)
        
        return jsonify(predictions.tolist())
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction."}), 500