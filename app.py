from multiprocessing import freeze_support
from flask import Flask, jsonify, request
from flask_cors import CORS
from src.model import TFTransformer

if __name__ == '__main__':

    app = Flask(__name__)
    CORS(app)

    freeze_support()
    
    model = TFTransformer()

    @app.route("/predict", methods=["POST"])
    def predict():
        print("DEBUG: predict called")

        req_data = request.get_json(force=True, silent=True)

        if req_data and req_data.get('longitude') and req_data.get('latitude'):
            longitude = float(req_data.get('longitude'))
            latitude = float(req_data.get('latitude'))
        else:
            return jsonify({"ok": False, "error": "Invalid input."}), 422

        results = model.predict_and_digest(
            longitude=longitude,
            latitude=latitude
        )

        return jsonify(results), 200
    
    print("Starting Flask app")
    print(f"Registered routes: {app.url_map}")
    app.run(debug=True)
