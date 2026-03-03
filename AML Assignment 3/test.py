import pytest  # Testing framework
import joblib  # Loads model
import sys
sys.path.append('AML Assignment 3')
from Scorre_card import score  # Imports the function we are testing


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
