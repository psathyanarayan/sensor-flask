import pickle
import numpy as np
from flask import Flask, jsonify, request
# Load the trained model from the pickle file
with open('modelKNN.pkl', 'rb') as f:
    model = pickle.load(f)
#X_train = np.load('t/X_train.npy',allow_pickle=True)
#y_train = np.load('t/y_train.npy',allow_pickle=True)


# Create a Flask app
app = Flask(__name__)
# Define an endpoint for making predictions
@app.route('/predict', methods=['GET','POST'])
def predict():
    # Get the input data from the user
    global prediction
    if request.method == 'POST':
        input_data = request.get_json()

        # Convert the input data to a numpy array
        input_array = np.array(list(input_data.values())).reshape(1, -1)

        # Make a prediction using the trained model
        prediction = model.predict(input_array)[0]
        print(prediction)
        # Convert prediction to a float
        # val = y_train[prediction-1]
        # prediction = (prediction)
        # print(prediction)
        # Return the prediction to the user in JSON format
        return jsonify({'prediction': prediction})
    if request.method == 'GET':
        return prediction





# Run the Flask app
if __name__ == '__main__':
    app.run()
