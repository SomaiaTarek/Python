from flask import Flask, request, jsonify
import numpy as np
from model import predict_mobile_price_range

app = Flask(__name__)

@app.route('/predict_price_range', methods=['POST'])
def predict_price_range():
    if request.method == 'POST':
        user_input = request.json.get('user_input', None)
        if user_input is None:
            return jsonify({'error': 'No input data provided'}), 400

        user_input = np.array(user_input).reshape(1, -1)
        
        with app.test_request_context('/predict_price_range'):
            request.environ['SERVER_SOFTWARE'] = 'Gunicorn/19.9.0'
            request.environ['SERVER_NAME'] = '127.0.0.1'
            request.environ['SERVER_PORT'] = '5000'
            price_range = predict_mobile_price_range(user_input)

        return jsonify({'price_range': price_range})

if __name__ == '__main__':
    app.run(debug=True)
