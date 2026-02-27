import pytest  # Testing framework
import joblib  # Loads model
import sys
import os
import subprocess
import requests
import time
import signal
sys.path.append(r'AML Assignment 4')
from score import score  # Imports the function we are testing


@pytest.fixture
def load_model():
    """Loads the trained model for testing."""
    return joblib.load("model.pkl")

def test_score_smoke(load_model):
    """Test if the score function runs without crashing."""
    text = "This is a test message."
    threshold = 0.5
    assert score(text, load_model, threshold) is not None

def test_score_output_format(load_model):
    """Test if output format is correct (prediction: bool, propensity: float)."""
    text = "This is a test message."
    threshold = 0.5
    prediction, propensity = score(text, load_model, threshold)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

def test_score_prediction_range(load_model):
    """Test if prediction is always 0 or 1."""
    text = "This is a test message."
    threshold = 0.5
    prediction, _ = score(text, load_model, threshold)
    assert prediction in [True, False]

def test_score_propensity_range(load_model):
    """Test if propensity score is between 0 and 1."""
    text = "This is a test message."
    threshold = 0.5
    _, propensity = score(text, load_model, threshold)
    assert 0.0 <= propensity <= 1.0

def test_score_threshold_0(load_model):
    """Test if threshold = 0 always predicts spam."""
    text = "This is a test message."
    threshold = 0.0
    prediction, _ = score(text, load_model, threshold)
    assert prediction == True

def test_score_threshold_1(load_model):
    """Test if threshold = 1 always predicts not spam."""
    text = "This is a test message."
    threshold = 1.0
    prediction, _ = score(text, load_model, threshold)
    assert prediction == False

def test_score_spam_text(load_model):
    """Test if obvious spam text is classified as spam."""
    text = "Congratulations! You won $10,000. Click here to claim."
    threshold = 0.5
    prediction, _ = score(text, load_model, threshold)
    assert prediction == True

def test_score_non_spam_text(load_model):
    """Test if obvious non-spam text is classified as non-spam."""
    text = "Hey, let's meet up for coffee tomorrow."
    threshold = 0.5
    prediction, _ = score(text, load_model, threshold)
    assert prediction == False



@pytest.fixture
def load_model():
    return joblib.load("model.pkl")

# Existing tests unchanged...

def test_docker():
    """Test the Docker container by launching it, sending a request, and checking response."""
    image_name = "flask-spam-app"
    container_name = "flask_spam_test"
    test_text = "Win a free iPhone now!"
    expected_key_set = {"prediction", "probability"}

    # Step 1: Build the Docker image (skip if already built)
    subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

    # Step 2: Run the container in detached mode
    run_cmd = [
        "docker", "run", "--rm", "-d",
        "-p", "5000:5000",
        "--name", container_name,
        image_name
    ]
    subprocess.run(run_cmd, check=True)

    try:
        # Step 3: Wait for Flask to start
        time.sleep(5)

        # Step 4: Send a POST request
        url = "http://localhost:5000/score"
        response = requests.post(url, json={"text": test_text})

        # Step 5: Assert valid response
        assert response.status_code == 200
        json_data = response.json()
        assert set(json_data.keys()) == expected_key_set
        assert isinstance(json_data["prediction"], bool)
        assert 0.0 <= json_data["probability"] <= 1.0

    finally:
        # Step 6: Stop the container
        subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
