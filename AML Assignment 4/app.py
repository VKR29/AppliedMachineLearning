from flask import Flask, request, jsonify
import joblib
from score import score

# Initialize Flask app
app = Flask(__name__)

# Load the trained model once when the app starts sys.path.append(r'AML Assignment 4')


model = joblib.load(r"AML Assignment 4\model.pkl")  # ✅ raw string



@app.route("/", methods=["GET"])
def index():
    """Basic health check route."""
    return "Flask server is up and running!", 200

@app.route("/score", methods=["POST"])
def get_score():
    """Accepts POST request with input text and returns prediction and propensity."""
    request_data = request.get_json()

    if not request_data or "text" not in request_data:
        return jsonify({"error": "Input JSON must contain a 'text' field"}), 400

    input_text = request_data["text"]
    prediction, probability = score(input_text, model, threshold=0.5)

    return jsonify({
        "prediction": int(prediction),
        "propensity": float(probability)
    })

# Display registered routes for debugging purposes
with app.test_request_context():
    print(app.url_map)

# Run the app if this file is executed directly
if __name__ == "__main__":
    app.run(port=5000, debug=True)