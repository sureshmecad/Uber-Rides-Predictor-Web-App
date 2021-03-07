import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model =  pickle.load(open('taxi.pkl','rb'))

@app.route("/")
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	int_feat = [int(x) for x in request.form.values()]
	final_feat = [np.array(int_feat)]
	prediction = model.predict(final_feat)
	output = round(prediction[0],2)
	return render_template('index.html',prediction_text = 'Number of weekly ride : {}'.format(math.floor(output)))

if __name__ == '__main__':
	app.run(host='127.0.0.1',port=5000)